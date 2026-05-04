import argparse
import re
from pathlib import Path

import mido


def _numeric_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 10**18


def midi_to_abs_events(mid: mido.MidiFile):
    """
    Merge all tracks into a single stream of events with absolute tick times.
    Returns (ticks_per_beat, events_abs) where events_abs = [(abs_tick, msg), ...]
    """
    merged = mido.merge_tracks(mid.tracks)
    abs_t = 0
    events = []
    for msg in merged:
        abs_t += msg.time
        events.append((abs_t, msg))
    return mid.ticks_per_beat, events


def scale_ticks(events_abs, scale: float):
    if scale == 1.0:
        return events_abs
    out = []
    for t, msg in events_abs:
        # round to int ticks
        t2 = int(round(t * scale))
        out.append((t2, msg))
    return out


def abs_events_to_track(events_abs):
    """
    events_abs must be sorted by abs tick.
    Convert back to delta times in a single track.
    """
    events_abs = sorted(events_abs, key=lambda x: x[0])
    track = mido.MidiTrack()
    prev = 0
    for t, msg in events_abs:
        dt = t - prev
        prev = t
        track.append(msg.copy(time=dt))
    return track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Folder containing 0000.mid, 0001.mid, ...")
    ap.add_argument("--out_mid", type=str, required=True)
    ap.add_argument("--keep_meta_from_all", action="store_true",
                    help="Keep tempo/time_signature/etc from all segments (default: keep all anyway).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    midi_files = sorted(in_dir.glob("*.mid"), key=_numeric_key)
    if not midi_files:
        raise SystemExit(f"No .mid found in {in_dir}")

    # Base ticks_per_beat = first file
    base_mid = mido.MidiFile(midi_files[0])
    base_tpb, base_events = midi_to_abs_events(base_mid)

    all_events = []
    offset = 0
    # Add first segment
    for t, msg in base_events:
        all_events.append((t + offset, msg))
    seg_len = max((t for t, _ in base_events), default=0)
    offset += seg_len

    # Add subsequent segments
    for mf in midi_files[1:]:
        mid = mido.MidiFile(mf)
        tpb, events = midi_to_abs_events(mid)

        # If ticks_per_beat differs, scale tick times so musical length matches roughly.
        # (This assumes same tempo structure; if not, you should normalize in DAW.)
        scale = base_tpb / tpb
        events = scale_ticks(events, scale)

        for t, msg in events:
            # Optionally: you could drop duplicated meta messages here.
            # We'll keep everything by default.
            all_events.append((t + offset, msg))

        seg_len = max((t for t, _ in events), default=0)
        offset += seg_len

    # Ensure end_of_track exists at end
    all_events.append((offset + 1, mido.MetaMessage("end_of_track", time=0)))

    out = mido.MidiFile(ticks_per_beat=base_tpb)
    out.tracks.append(abs_events_to_track(all_events))
    out.save(args.out_mid)
    print(f"Wrote: {args.out_mid}  (segments={len(midi_files)}, tpb={base_tpb})")


if __name__ == "__main__":
    main()

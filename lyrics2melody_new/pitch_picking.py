# choose_svs_singer_octave.py
from __future__ import annotations
import argparse, io, os, tempfile
import numpy as np
import pretty_midi as pm
from dataclasses import dataclass
from mido import MidiFile, MidiTrack, MetaMessage, bpm2tempo

# ---------- Singer Profiles (adjust to your singers) ----------
@dataclass
class SingerProfile:
    name: str
    comfy_low: int
    comfy_high: int
    usable_low: int
    usable_high: int
    passaggio_low: int
    passaggio_high: int

MALE = SingerProfile(
    name="male",
    comfy_low=48,  comfy_high=73,   # C3–A4
    usable_low=45, usable_high=72,  # A2–C5
    passaggio_low=64, passaggio_high=66,  # E4–F#4
)
FEMALE = SingerProfile(
    name="female",
    comfy_low=55,  comfy_high=73,   # C3–A4
    usable_low=52, usable_high=76,  # A2–C5
    passaggio_low=68, passaggio_high=70,  # E4–F#4
)

PITCH_STEPS = list(range(-24, 25))  # every semitone from -2 to +2 octaves

# ---------- Melody helpers ----------
def get_single_melody(pm_obj: pm.PrettyMIDI):
    """Return (instrument, notes[(pitch,start,end)]) for the first instrument with notes."""
    if not pm_obj.instruments:
        raise ValueError("MIDI has no instruments.")
    inst = next((i for i in pm_obj.instruments if i.notes), pm_obj.instruments[0])
    notes = [(n.pitch, n.start, n.end) for n in inst.notes if n.end > n.start]
    if not notes:
        raise ValueError("No notes in the first instrument.")
    notes.sort(key=lambda x: x[1])
    return inst, notes

def transpose_in_place(inst: pm.Instrument, semitones: int):
    for n in inst.notes:
        n.pitch = int(np.clip(n.pitch + semitones, 0, 127))


# ---------- Scoring ----------
def score_for_profile(notes, profile: SingerProfile):
    """Lower is better. Heavy penalty outside usable; medium outside comfy; passaggio & sustains penalized."""
    total = sum(max(1e-6, e - s) for _, s, e in notes)
    out_usable = out_comfy = pass_time = sust_high = 0.0
    for p, s, e in notes:
        d = max(1e-6, e - s)
        if p < profile.usable_low:
            out_usable += (profile.usable_low - p) * d
        elif p > profile.usable_high:
            out_usable += (p - profile.usable_high) * d
        if p < profile.comfy_low:
            out_comfy += (profile.comfy_low - p) * d
        elif p > profile.comfy_high:
            out_comfy += (p - profile.comfy_high) * d
        if profile.passaggio_low <= p <= profile.passaggio_high:
            pass_time += d
        if p >= profile.comfy_high - 1 and d >= 1.0:
            sust_high += d * (p - (profile.comfy_high - 1) + 1)
    comfy_center = 0.5 * (profile.comfy_low + profile.comfy_high)
    wavg = sum(p * max(1e-6, e - s) for p, s, e in notes) / total
    center_dev = abs(wavg - comfy_center)
    # weights
    return (10.0 * out_usable / total
            + 2.5 * out_comfy / total
            + 1.0 * (pass_time / total)
            + 1.5 * (sust_high / total)
            + 0.5 * center_dev)

def evaluate_shifts(notes, pitch_steps=PITCH_STEPS, fixed_singer: str | None = None):
    """Return list of dicts with male/female cost for each semitone shift.
    If fixed_singer is 'male' or 'female', ranking uses only that singer's cost."""
    results = []
    for st in pitch_steps:
        shifted = [(p + st, s, e) for (p, s, e) in notes]
        male_cost = score_for_profile(shifted, MALE)
        female_cost = score_for_profile(shifted, FEMALE)
        if fixed_singer == "male":
            best, best_cost = "male", male_cost
        elif fixed_singer == "female":
            best, best_cost = "female", female_cost
        else:
            best = "male" if male_cost < female_cost else "female"
            best_cost = min(male_cost, female_cost)
        results.append(dict(shift=st, male_cost=male_cost, female_cost=female_cost,
                            best=best, best_cost=best_cost))
    results.sort(key=lambda r: (r["best_cost"], abs(r["shift"])))
    return results

# ---------- Tempo rewrite (single global BPM) ----------
def write_pm_with_bpm(pm_obj: pm.PrettyMIDI, out_path: str, bpm: float | None):
    """
    Save PrettyMIDI to out_path and, if bpm is provided, enforce a single global tempo.
    Implemented by re-opening with mido and replacing any set_tempo metas.
    """
    # 1) Save current pm to a temp file
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name
    pm_obj.write(tmp_path)

    # 2) If no BPM requested, just move it
    if bpm is None:
        # just rename/move
        os.replace(tmp_path, out_path)
        return

    # 3) Overwrite tempo metas using mido
    mid = MidiFile(tmp_path)
    target_tempo = bpm2tempo(bpm)  # microseconds per beat

    new_tracks = []
    for ti, track in enumerate(mid.tracks):
        nt = MidiTrack()
        # remove any set_tempo in this track
        for msg in track:
            if not (msg.is_meta and msg.type == "set_tempo"):
                nt.append(msg.copy(time=msg.time))
        new_tracks.append(nt)
    mid.tracks = new_tracks

    # Prepend a single set_tempo at the beginning of track 0
    mid.tracks[0].insert(0, MetaMessage('set_tempo', tempo=target_tempo, time=0))

    mid.save(out_path)
    os.remove(tmp_path)

# ---------- Main ----------
def run(midi_path: str, out_dir: str | None, meta_json_path: str | None, fixed_singer: str | None = None):
    import json

    pm_obj = pm.PrettyMIDI(midi_path)
    inst, notes = get_single_melody(pm_obj)

    # Evaluate all semitone shifts; if fixed_singer is set, rank by that singer only
    results = evaluate_shifts(notes, PITCH_STEPS, fixed_singer=fixed_singer)
    top = results[0]
    best_singer = top["best"]
    best_shift = top["shift"]

    # ✅ Transpose IN PLACE on the original object (lyrics & meta preserved)
    transpose_in_place(inst, best_shift)

    # save shifted MIDI as a new file alongside the original
    base, ext = os.path.splitext(midi_path)
    out_path = f"{base}{ext}"

    # load/save meta json (your --bpm arg is actually a path)
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    singer_dict = {'male': '5', 'female': '6'}
    meta['singer'] = singer_dict[best_singer]
    meta['pitch_shift'] = best_shift
    target_bpm = float(meta["bpm"])

    # write with enforced BPM; other meta (including lyrics) stays intact
    write_pm_with_bpm(pm_obj, out_path, target_bpm)

    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # logs ...


    # Diagnostics
    print(f"[Input] {midi_path}")
    print(f"Target BPM: {'(leave original)' if target_bpm is None else target_bpm}")
    print("Semitone evaluation (lower cost is better, top 5):")
    for r in results[:5]:
        print(f"  shift={r['shift']:>+3}  male={r['male_cost']:.3f}  female={r['female_cost']:.3f}  best={r['best']}")
    print("\n[Recommendation]")
    print(f"  Singer: {best_singer}")
    print(f"  Pitch shift: {best_shift:+d} semitones")
    print(f"  Wrote: {out_path}")
    return dict(singer=best_singer, shift_semitones=best_shift, midi_out=out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Choose SVS singer (male/female) and octave shift; optionally set BPM.")
    ap.add_argument("midi", help="Path to single-line melody MIDI")
    ap.add_argument("--bpm", type=str, default=None, help="Target BPM (e.g., 134). Omit to keep original tempo.")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: MIDI folder).")
    ap.add_argument("--singer", type=str, default=None, choices=["male", "female"],
                    help="Fix the singer to 'male' or 'female'. Omit to auto-select.")
    args = ap.parse_args()
    run(args.midi, args.out_dir, args.bpm, fixed_singer=args.singer)

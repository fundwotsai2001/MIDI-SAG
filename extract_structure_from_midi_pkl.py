#!/usr/bin/env python3
"""
Emit STRUCTURE_STARTS / STRUCTURE_TAGS bash arrays for a song by combining:

  * ``data/struct/<name>.pkl`` - tuple (section_letters, lyric_strings)
    where ``section_letters`` is one letter per line (A/B/C ... typically
    A=verse, B=chorus, C=bridge).

  * ``data/Midi/<name>.mid`` - the vocal MIDI with lyric events. Each
    lyric whose text contains ``.`` marks a phrase (line) end. Line i's
    start time is the first note-on at or after line (i-1)'s end.

Consecutive lines with the same letter are merged into one section;
the emitted start time is the first note of that section's first line.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mido

DEFAULT_TAG_MAP = {"A": "verse", "B": "chorus", "C": "verse",
                   "D": "bridge", "E": "outro"}


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def midi_line_starts(midi_path: Path) -> list[float]:
    """Return start time (s) of each lyric line using the '.'-terminated
    lyric events as phrase boundaries."""
    mid = mido.MidiFile(str(midi_path))
    t = 0.0
    note_ons: list[float] = []
    line_ends: list[float] = []
    for msg in mid:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            note_ons.append(t)
        elif msg.is_meta and msg.type == "lyrics" and "." in msg.text:
            line_ends.append(t)

    note_ons.sort()
    line_ends.sort()

    starts: list[float] = []
    prev_end = 0.0
    for le in line_ends:
        window = [n for n in note_ons if prev_end <= n <= le + 1e-6]
        starts.append(window[0] if window else prev_end)
        prev_end = le
    return starts


def group_consecutive(letters: list[str]) -> list[tuple[int, str]]:
    sections: list[tuple[int, str]] = []
    for i, letter in enumerate(letters):
        if not sections or sections[-1][1] != letter:
            sections.append((i, letter))
    return sections


def format_bash_arrays(starts: list[float], tags: list[str]) -> str:
    starts_str = "  ".join(f"{s:.2f}" for s in starts)
    tags_str = "   ".join(tags)
    return (
        f"STRUCTURE_STARTS=( {starts_str} )\n"
        f"STRUCTURE_TAGS=(   {tags_str} )\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("struct_pkl", type=Path,
                    help="Path to data/struct/<name>.pkl")
    ap.add_argument("midi", type=Path,
                    help="Path to data/Midi/<name>.mid")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output .txt (default: <midi>.structure.txt).")
    ap.add_argument("--raw_letters", action="store_true",
                    help="Emit A/B/C letters instead of verse/chorus/bridge.")
    args = ap.parse_args()

    letters, lyrics = load_pkl(args.struct_pkl)
    line_starts = midi_line_starts(args.midi)

    if len(letters) != len(line_starts):
        raise SystemExit(
            f"Line count mismatch: struct has {len(letters)} letters, "
            f"MIDI has {len(line_starts)} lyric lines"
        )

    sections = group_consecutive(letters)
    starts = [round(line_starts[i], 2) for i, _ in sections]
    if args.raw_letters:
        tags = [letter for _, letter in sections]
    else:
        tags = [DEFAULT_TAG_MAP.get(letter, letter.lower())
                for _, letter in sections]

    # Merge adjacent sections that map to the same tag (e.g. A and C both → verse).
    merged_starts: list[float] = []
    merged_tags: list[str] = []
    for s, tag in zip(starts, tags):
        if merged_tags and merged_tags[-1] == tag:
            continue
        merged_starts.append(s)
        merged_tags.append(tag)
    starts, tags = merged_starts, merged_tags

    # Prepend an intro starting at 0 if the song doesn't already start there.
    if starts and starts[0] > 0.0:
        starts = [0.0] + starts
        tags = ["intro"] + tags

    out = args.output or args.midi.with_suffix(".structure.txt")
    text = format_bash_arrays(starts, tags)
    out.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\n[info] lines={len(line_starts)}  sections={len(sections)}")
    print(f"[info] Wrote {out}")


if __name__ == "__main__":
    main()

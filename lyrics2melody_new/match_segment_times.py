#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_segment_times.py

Usage:
  python -u match_segment_times.py ORIGINAL_LYRIC_PATH TIME_LYRIC_PATH STRUCT_time_LIST STRUCT_label_LIST

Inputs
  ORIGINAL_LYRIC_PATH : Traditional Chinese lyrics that include structure tags like <Chorus>, <Verse>, ...
  TIME_LYRIC_PATH     : Simplified Chinese time-stamped lyrics.
                        Supported formats:
                          1) TSV:  start\tend\ttext   (e.g., "0:07.164\t0:10.000\t副歌第一句")
                          2) LRC-like: "[mm:ss.xx] lyric text" (first timestamp on the line is used)

Outputs
  STRUCT_time_LIST    : JSON list of start times (seconds) for each structure in order.
  STRUCT_label_LIST   : JSON list of structure labels (lowercased) aligned with the times.

Notes
  • This script is designed to work with ANY pair of inputs; **no song-specific hardcoding**.
  • To maximize matching accuracy between Traditional (orig) and Simplified (timed), it will
    optionally use OpenCC (if installed) or hanziconv to convert Traditional→Simplified on the fly.
    If neither library is present, it will continue without conversion (and still work when the
    two inputs already share similar character sets). For best results across arbitrary pairs,
    install one of:
        pip install opencc-python-reimplemented
        # or
        pip install hanziconv

  • Matching uses a robust fuzzy approach (SequenceMatcher + partial-window ratio) with
    monotonic timestamp constraints to keep section order realistic.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Optional

# ---- Optional Traditional → Simplified conversion -------------------------------------------
_CONVERTER = None

def _try_load_converter():
    global _CONVERTER
    # Try OpenCC first
    try:
        from opencc import OpenCC  # type: ignore
        _CONVERTER = OpenCC('t2s')
        return
    except Exception:
        pass
    # Try hanziconv next
    try:
        from hanziconv import HanziConv  # type: ignore
        class _HZC:
            def convert(self, s: str) -> str:
                return HanziConv.toSimplified(s)
        _CONVERTER = _HZC()
        return
    except Exception:
        pass

_try_load_converter()


def to_simplified(s: str) -> str:
    """Convert Traditional Chinese to Simplified if possible; otherwise return input unchanged."""
    if _CONVERTER is None:
        return s
    try:
        # OpenCC and _HZC both expose .convert
        return _CONVERTER.convert(s)  # type: ignore[attr-defined]
    except Exception:
        return s


# ---- Utilities --------------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[\s，。,．、！？!?,；;“”\"'：:—\-（）()\[\]【】·…#·`~@$%^&*_+=|\\]")

def clean_text_zh(s: str) -> str:
    """Remove whitespace/punctuation; keep letters/numbers/CJK for fuzzy matching."""
    s = _PUNCT_RE.sub("", s)
    return s


def parse_time_str(t: str) -> Optional[float]:
    """Parse time strings like mm:ss(.ms), hh:mm:ss(.ms). Returns seconds or None."""
    t = t.strip()
    if not t:
        return None
    try:
        parts = [float(x) for x in t.split(":")]
        if len(parts) == 2:
            mm, ss = parts
            return mm * 60.0 + ss
        elif len(parts) == 3:
            hh, mm, ss = parts
            return hh * 3600.0 + mm * 60.0 + ss
    except Exception:
        pass
    # Also support plain seconds (e.g., "7.164")
    try:
        return float(t)
    except Exception:
        return None


@dataclass
class TimedLine:
    start: float
    text: str
    clean: str


def read_timed_lines(path: Path) -> List[TimedLine]:
    lines: List[TimedLine] = []

    # Heuristic 1: TSV format: start\tend\ttext
    def _try_tsv(raw: str) -> Optional[TimedLine]:
        parts = raw.rstrip("\n").split("\t")
        if len(parts) < 3:
            return None
        s = parse_time_str(parts[0])
        if s is None:
            return None
        text = "\t".join(parts[2:])  # preserve tabs in lyric text
        simp = to_simplified(text)
        return TimedLine(start=s, text=text, clean=clean_text_zh(simp))

    # Heuristic 2: LRC-like: [mm:ss.xx] lyric text (use FIRST bracket)
    LRC_RE = re.compile(r"^\s*\[(?P<t>\d{1,2}:\d{2}(?:\.\d{1,3})?)\][\s]*(?P<txt>.*)")
    def _try_lrc(raw: str) -> Optional[TimedLine]:
        m = LRC_RE.match(raw)
        if not m:
            return None
        s = parse_time_str(m.group("t"))
        if s is None:
            return None
        text = m.group("txt").strip()
        simp = to_simplified(text)
        return TimedLine(start=s, text=text, clean=clean_text_zh(simp))

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            item = _try_tsv(raw)
            if item is None:
                item = _try_lrc(raw)
            if item is None:
                # As a last resort, try "start end text" (space-separated)
                parts = raw.split()
                if len(parts) >= 3:
                    s = parse_time_str(parts[0])
                    if s is not None:
                        text = " ".join(parts[2:])
                        simp = to_simplified(text)
                        item = TimedLine(start=s, text=text, clean=clean_text_zh(simp))
            if item is not None:
                lines.append(item)

    # Ensure ascending by start time
    lines.sort(key=lambda x: x.start)
    return lines


@dataclass
class Section:
    label: str
    anchor_line: str  # first non-empty lyric line under the label


def read_sections_with_tags(path: Path) -> List[Section]:
    sections: List[Section] = []
    current_label: Optional[str] = None
    buffer: List[str] = []

    TAG_RE = re.compile(r"^\s*<\s*([^>]+?)\s*>\s*$", re.IGNORECASE)

    def _flush():
        nonlocal current_label, buffer
        if current_label is not None:
            # take the first non-empty non-tag line as the anchor
            anchor = next((ln for ln in buffer if ln.strip()), "")
            if anchor:
                sections.append(Section(label=current_label.strip(), anchor_line=anchor))
        buffer = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            m = TAG_RE.match(line)
            if m:
                _flush()
                current_label = m.group(1)
            else:
                buffer.append(line)
    _flush()

    if not sections:
        print(
            "[WARN] No <...> structure tags detected in ORIGINAL_LYRIC_PATH; output will be empty.",
            file=sys.stderr,
        )
    return sections


# ---- Fuzzy matching --------------------------------------------------------------------------

def partial_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # windowed best-of subsequence match
    if len(a) <= len(b):
        shorter, longer = a, b
    else:
        shorter, longer = b, a
    n = len(shorter)
    best = 0.0
    for i in range(0, max(1, len(longer) - n + 1)):
        window = longer[i : i + n]
        best = max(best, SequenceMatcher(None, shorter, window).ratio())
    return best


def composite_score(q: str, cand: str) -> float:
    s1 = SequenceMatcher(None, q, cand).ratio()
    s2 = partial_ratio(q, cand)
    # prefer similar lengths a bit
    length_penalty = abs(len(q) - len(cand)) / (len(q) + 1e-6)
    return max(s1, s2) - 0.10 * length_penalty


def match_sections_to_times(
    sections: List[Section], timed: List[TimedLine], min_gap: float = 0.5
) -> Tuple[List[float], List[str]]:
    times: List[float] = [0]
    labels: List[str] = ['intro']

    last_time = -1e9
    for sec in sections:
        q = clean_text_zh(to_simplified(sec.anchor_line))
        best_i = -1
        best_score = -1.0
        for i, tl in enumerate(timed):
            if tl.start < last_time - min_gap:
                continue  # keep monotonicity
            score = composite_score(q, tl.clean)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0:
            chosen = timed[best_i]
            times.append(round(chosen.start, 3))
            labels.append(sec.label.strip().lower())
            last_time = chosen.start
        else:
            # If no candidate (shouldn't happen), append sentinel and label anyway
            print(
                f"[WARN] Could not match section '{sec.label}' (anchor='{sec.anchor_line[:40]}...')",
                file=sys.stderr,
            )
            times.append(-1.0)
            labels.append(sec.label.strip().lower())

    return times, labels


# ---- Post-processing: split long structures ---------------------------------------------------

def enforce_max_segment_length(times, labels, max_len: float = 40.0):
    """Split any structure segment longer than max_len seconds by inserting
    evenly spaced boundaries and repeating the structure tag for each split.

    Example:
      times  = [0, 10, 60, 80]
      labels = ["intro", "chorus", "verse", "chorus"]
      max_len = 40
      -> [0, 10, 35, 60, 80], ["intro", "chorus", "chorus", "verse", "chorus"]
    """
    if not times or len(times) != len(labels):
        return times, labels

    new_times = [times[0]]
    new_labels = [labels[0]]

    for i in range(len(times) - 1):
        start = times[i]
        end = times[i + 1]
        lab = labels[i + 1]  # label boundary semantics: label[i+1] starts at times[i+1]

        seg_len = end - start
        if seg_len > max_len:
            # number of equal segments such that each <= max_len
            import math
            n_segments = int(math.ceil(seg_len / max_len))
            # we already have the segment starting at 'start'; insert (n_segments-1) internal splits
            step = seg_len / n_segments
            # Generate internal split points
            splits = [round(start + step * k, 3) for k in range(1, n_segments)]
            for sp in splits:
                if sp > new_times[-1]:
                    new_times.append(sp)
                    # Repeat previous structure tag (the one active until next boundary)
                    new_labels.append(new_labels[-1])
        # Finally append the original boundary and its next label
        if end > new_times[-1]:
            new_times.append(end)
            new_labels.append(lab)

    return new_times, new_labels


# ---- CLI -------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Match structure tags to timed lyrics")
    ap.add_argument("original_lyric_path", type=Path,
                    help="Traditional Chinese lyrics with <structure> tags")
    ap.add_argument("time_lyric_path", type=Path,
                    help="Simplified Chinese time-stamped lyrics (TSV or LRC-like)")
    ap.add_argument("struct_time_list", type=Path,
                    help="Output JSON file for structure start times (seconds)")
    ap.add_argument("struct_label_list", type=Path,
                    help="Output JSON file for structure labels (lowercased)")
    args = ap.parse_args()

    if not args.original_lyric_path.exists():
        sys.exit(f"ORIGINAL_LYRIC_PATH not found: {args.original_lyric_path}")
    if not args.time_lyric_path.exists():
        sys.exit(f"TIME_LYRIC_PATH not found: {args.time_lyric_path}")

    sections = read_sections_with_tags(args.original_lyric_path)
    timed = read_timed_lines(args.time_lyric_path)

    if not sections:
        # write empty arrays to be explicit
        args.struct_time_list.write_text("[]", encoding="utf-8")
        args.struct_label_list.write_text("[]", encoding="utf-8")
        print("[INFO] No sections found; wrote empty lists.")
        return
    if not timed:
        sys.exit("No timed lyric lines parsed from TIME_LYRIC_PATH. Check the input format.")

    times, labels = match_sections_to_times(sections, timed)

    # Enforce a maximum segment length of 40 seconds by splitting long spans
    times, labels = enforce_max_segment_length(times, labels, max_len=40.0)

    # Save outputs
    args.struct_time_list.write_text(json.dumps(times, ensure_ascii=False), encoding="utf-8")
    args.struct_label_list.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    # Brief stdout message for convenience
    print(json.dumps({
        "STRUCT_time_LIST": times,
        "STRUCT_label_LIST": labels
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

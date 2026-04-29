
from __future__ import annotations
from pathlib import Path
import pickle
import re
import math
import argparse
import json
from typing import List, Tuple, Optional, Dict, Any

# ---------- Utilities ----------

STRUCT_DICT = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

def normalize_punct(s: str) -> str:
    # Remove most punctuation & whitespace for char-count in CJK
    return re.sub(r"[\s\p{P}\p{S}]+", "", s, flags=re.UNICODE)

def is_cjk(text: str) -> bool:
    # Heuristic: consider it CJK if no spaces and any CJK codepoints
    if " " in text:
        return False
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf]", text))

def token_count(line: str) -> int:
    line = line.strip()
    if not line:
        return 0
    if is_cjk(line):
        # Count visible CJK chars (exclude punctuation/symbols/spaces)
        # Using a simpler filter if regex classes \p{P} not supported
        cleaned = re.sub(r"[\s，。、《》；；：：「」『』？！,.!?;:()\[\]{}“”\"'·…—-]", "", line)
        return len(cleaned)
    else:
        # Whitespace token count
        return len(line.split())

def structure_to_ints(struct_events: List[str]) -> List[int]:
    # Convert A/B/C/D/E to 0..4 with unknowns mapped to -1
    out = []
    for s in struct_events:
        out.append(STRUCT_DICT.get(s, -1))
    return out

def l1_profile_distance(a: List[int], b: List[int]) -> float:
    # Align by length: pad shorter with its median to be fair
    if not a and not b:
        return 0.0
    if not a:
        return float(sum(abs(x) for x in b)) / (len(b) or 1)
    if not b:
        return float(sum(abs(x) for x in a)) / (len(a) or 1)
    la, lb = len(a), len(b)
    if la < lb:
        pad = a[len(a)//2] if a else 0
        a = a + [pad] * (lb - la)
    elif lb < la:
        pad = b[len(b)//2] if b else 0
        b = b + [pad] * (la - lb)
    # mean absolute error
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

def hamming_with_len_penalty(a: List[int], b: List[int]) -> float:
    # Hamming distance for overlap + penalty per extra element
    la, lb = len(a), len(b)
    overlap = min(la, lb)
    if overlap == 0:
        return float(max(la, lb))
    mismatches = sum(1 for i in range(overlap) if a[i] != b[i])
    extra = abs(la - lb)
    return float(mismatches + extra)

def sentence_count_distance(a_len: int, b_len: int) -> float:
    return float(abs(a_len - b_len))

def safe_load_pickle(path: Path) -> Optional[Tuple[List[str], List[str]]]:
    """
    Try to load (struct_events, lyrics_list). Accepts:
      - Tuple[List[str], List[str]]
      - Dict with keys 'struct'/'struct_events' and 'lyrics'/'lines'
    Returns None if not parseable.
    """
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None

    struct_events = None
    lyrics = None

    if isinstance(obj, tuple) and len(obj) >= 2:
        a, b = obj[0], obj[1]
        if isinstance(a, list) and all(isinstance(x, str) for x in a):
            struct_events = a
        if isinstance(b, list) and all(isinstance(x, str) for x in b):
            lyrics = b

    if struct_events is None or lyrics is None:
        if isinstance(obj, dict):
            # Try common keys
            for k in ["struct", "struct_events", "structure", "sections"]:
                v = obj.get(k)
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    struct_events = v
                    break
            for k in ["lyrics", "lines", "sentences", "lyric_lines"]:
                v = obj.get(k)
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    lyrics = v
                    break

    if struct_events is None or lyrics is None:
        return None
    return struct_events, lyrics

def score_candidate(
    target_lines: List[str],
    cand_lines: List[str],
    target_struct_ints: Optional[List[int]],
    cand_struct_ints: Optional[List[int]],
    weights=(0.4, 0.4, 0.2),
) -> Dict[str, float]:
    """
    Compute a similarity score (lower is better).
    Components:
      - sentence count diff (normalized by max(len))
      - L1 distance between per-sentence token counts (normalized by max token)
      - structure pattern mismatch (normalized by max len) if target_struct provided
    """
    w_sent, w_profile, w_struct = weights

    # 1) sentence count distance
    a_len, b_len = len(target_lines), len(cand_lines)
    if b_len < a_len:
        # Hard constraint: candidate must have at least as many sentences as the target
        sent_norm = float("inf")
    else:
        sent_dist_raw = sentence_count_distance(a_len, b_len)
        sent_norm = sent_dist_raw / max(1, max(a_len, b_len))


    # 2) token count profile distance
    tgt_profile = [token_count(x) for x in target_lines]
    cand_profile = [token_count(x) for x in cand_lines]
    profile_raw = l1_profile_distance(tgt_profile, cand_profile)
    # normalize by max of medians to stabilize scaling
    denom = max(1.0, float(max(tgt_profile + cand_profile) or 1))
    profile_norm = profile_raw / denom

    # 3) structure distance
    if target_struct_ints is not None and cand_struct_ints is not None:
        struct_raw = hamming_with_len_penalty(target_struct_ints, cand_struct_ints)
        struct_norm = struct_raw / max(1, max(len(target_struct_ints), len(cand_struct_ints)))
    else:
        struct_norm = 0.0
        w_struct = 0.0  # effectively ignore

    total = w_sent * sent_norm + w_profile * profile_norm + w_struct * struct_norm
    return {
        "total": total,
        "sent_norm": sent_norm,
        "profile_norm": profile_norm,
        "struct_norm": struct_norm,
    }

def find_best_matches(
    folder: str,
    target_struct_events: Optional[List[str]],
    target_lyrics: List[str],
    topk: int = 10,
    weights=(0.2, 0.6, 0.2),
) -> List[Dict[str, Any]]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    target_struct_ints = target_struct_events

    results = []
    for p in folder_path.glob("*.pkl"):
        parsed = safe_load_pickle(p)
        if parsed is None:
            continue
        cand_struct, cand_lines = parsed
        cand_struct_ints = structure_to_ints(cand_struct) if cand_struct else None
        parts = score_candidate(
            target_lyrics, cand_lines, target_struct_ints, cand_struct_ints, weights=weights
        )
        results.append({
            "path": str(p),
            "score_total": parts["total"],
            "score_breakdown": parts,
            "cand_num_sentences": len(cand_lines),
            "tgt_num_sentences": len(target_lyrics),
            "cand_token_profile": [len(x.strip()) for x in cand_lines],  # quick glance length
            "cand_struct": cand_struct,
        })
    results.sort(key=lambda x: x["score_total"])
    best = results[0]['path']
    return best

def main():
    parser = argparse.ArgumentParser(description="Find similar lyrics/structure pickles.")
    parser.add_argument("--folder", required=True, help="Folder containing .pkl files")
    parser.add_argument("--target_json", help="Path to JSON with keys: 'struct_events' (optional), 'lyrics' (required)")
    parser.add_argument("--topk", type=int, default=10, help="Number of matches to show")
    parser.add_argument("--weights", type=str, default="0.4,0.4,0.2", help="Weights: w_sent,w_profile,w_struct")
    args = parser.parse_args()

    w = tuple(float(x) for x in args.weights.split(","))
    if len(w) != 3:
        raise ValueError("weights must be three comma-separated floats")

    if not args.target_json:
        raise SystemExit("Provide --target_json pointing to a JSON file with 'lyrics' and optional 'struct_events'.")

    with open(args.target_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_lyrics = data.get("lyrics")
    if not isinstance(target_lyrics, list) or not all(isinstance(x, str) for x in target_lyrics):
        raise ValueError("target_json['lyrics'] must be a list of strings")

    target_struct = data.get("struct_events", None)
    if target_struct is not None and (not isinstance(target_struct, list) or not all(isinstance(x, str) for x in target_struct)):
        raise ValueError("target_json['struct_events'] must be a list of strings (e.g., ['0','1',...])")

    results = find_best_matches(args.folder, target_struct, target_lyrics, topk=args.topk, weights=w)
    # Pretty print
    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=False)
    for i, r in enumerate(results, 1):
        print(f"#{i}  score={r['score_total']:.6f}  file={r['path']}  sentences: tgt={r['tgt_num_sentences']}, cand={r['cand_num_sentences']}")
        print(f"     parts={r['score_breakdown']}")
        # Optional: show first few structure tags
        if r.get("cand_struct"):
            print(f"     cand_struct_head={r['cand_struct'][:10]}")
        print()

if __name__ == "__main__":
    # main()
    folder = "/volume/fundwo-test/lyrics2melody/sentence_struct_1000"
    target_struct = ['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1']
    target_lyrics = ['我不会再让你哭', '我只想给你幸福', '前方未知的陌路', '有我陪你就不孤独', '我不会再让你哭', '我已许下了赌注', '不管未来有多苦', '只要有你陪我', '我就不在乎', '你说爱到了尽头', '真情无法再挽留', '孤独的夜里你独自泪流', '握着发黄的相片', '往事沥沥在心头', '那一夜看见你哭', '多想轻轻把你抱住', '为你抚平爱的伤口', '听你诉说心里的酸楚', '让我今生爱上了你', '也许是前世的赌注', '你让我心有所属', '我不会再让你哭', '我只想给你幸福', '前方未知的陌路', '有我陪你就不孤独', '我不会再让你哭', '我已许下了赌注', '不管未来有多苦', '只要有你陪我', '我就不在乎']

    matches = find_best_matches(folder, target_struct, target_lyrics)
    print(matches)
    # for m in matches:
    #     print(m["path"], m["score_total"], m["score_breakdown"])

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _numeric_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 10**18


def _to_channels(x: np.ndarray, channels: int) -> np.ndarray:
    # soundfile returns shape (n,) or (n, c)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == channels:
        return x
    if x.shape[1] == 1 and channels == 2:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] == 2 and channels == 1:
        return x.mean(axis=1, keepdims=True)
    # generic: pad or truncate
    if x.shape[1] < channels:
        pad = np.zeros((x.shape[0], channels - x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)
    return x[:, :channels]


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    # resample_poly expects (n,) or (n,c) ok; do per-channel
    if x.ndim == 1:
        return resample_poly(x, sr_out, sr_in)
    y = []
    for c in range(x.shape[1]):
        y.append(resample_poly(x[:, c], sr_out, sr_in))
    # different lengths can happen by 1 sample; trim to min
    min_len = min(len(ch) for ch in y)
    y = np.stack([ch[:min_len] for ch in y], axis=1)
    return y


def load_gaps_from_meta(meta_json: Path, wav_files: list[Path]) -> list[float]:
    """
    Optional helper:
    If you have a meta.json that contains per-segment start/end times,
    you can compute silence gaps (in seconds) between consecutive segments.

    Because meta.json schema varies by dataset, this function is a conservative
    "best effort". If it can't parse, it returns zeros.
    """
    try:
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
    except Exception:
        return [0.0] * max(0, len(wav_files) - 1)

    # Try common patterns: meta might be a dict keyed by utt_id, with "start"/"end" (seconds)
    # Or a list of entries.
    def find_entry(stem: str):
        if isinstance(meta, dict):
            return meta.get(stem) or meta.get(stem + ".wav") or meta.get(stem + ".mid")
        if isinstance(meta, list):
            for e in meta:
                if not isinstance(e, dict):
                    continue
                if e.get("utt") == stem or e.get("utt_id") == stem or e.get("id") == stem:
                    return e
        return None

    starts, ends = [], []
    for wf in wav_files:
        e = find_entry(wf.stem)
        if not e:
            starts.append(None); ends.append(None); continue
        s = e.get("start") or e.get("begin") or e.get("start_sec")
        t = e.get("end") or e.get("finish") or e.get("end_sec")
        starts.append(float(s) if s is not None else None)
        ends.append(float(t) if t is not None else None)

    gaps = []
    for i in range(len(wav_files) - 1):
        if ends[i] is None or starts[i + 1] is None:
            gaps.append(0.0)
        else:
            gaps.append(max(0.0, starts[i + 1] - ends[i]))
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Folder containing 0000.wav, 0001.wav, ...")
    ap.add_argument("--out_wav", type=str, required=True)
    ap.add_argument("--target_sr", type=int, default=None, help="Force output sample rate (else uses first file SR)")
    ap.add_argument("--target_channels", type=int, default=None, help="Force channels (1 or 2; else uses first file)")
    ap.add_argument("--meta_json", type=str, default=None, help="Optional meta.json to insert gaps")
    ap.add_argument("--gap_scale", type=float, default=1.0, help="Scale computed gaps (for debugging)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    wav_files = sorted(in_dir.glob("*.wav"), key=_numeric_key)
    if not wav_files:
        raise SystemExit(f"No .wav found in {in_dir}")

    # First file defines default format
    x0, sr0 = sf.read(wav_files[0], always_2d=False)
    if x0.ndim == 1:
        ch0 = 1
    else:
        ch0 = x0.shape[1]
    target_sr = args.target_sr or sr0
    target_ch = args.target_channels or ch0

    # Optional gaps
    gaps_sec = [0.0] * (len(wav_files) - 1)
    if args.meta_json:
        gaps_sec = load_gaps_from_meta(Path(args.meta_json), wav_files)
        gaps_sec = [g * args.gap_scale for g in gaps_sec]

    chunks = []
    for idx, wf in enumerate(wav_files):
        x, sr = sf.read(wf, always_2d=False)
        x = _resample(x, sr, target_sr)
        x = _to_channels(x, target_ch)
        chunks.append(x.astype(np.float32))

        # insert gap after this segment (except last)
        if idx < len(wav_files) - 1 and gaps_sec[idx] > 0:
            n_sil = int(round(gaps_sec[idx] * target_sr))
            chunks.append(np.zeros((n_sil, target_ch), dtype=np.float32))

    y = np.concatenate(chunks, axis=0)
    sf.write(args.out_wav, y, target_sr)
    print(f"Wrote: {args.out_wav}  (sr={target_sr}, ch={target_ch}, segments={len(wav_files)})")


if __name__ == "__main__":
    main()

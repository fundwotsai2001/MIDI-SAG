import argparse
import os
import builtins
import numpy as np
import torch
import soundfile as sf
import librosa

# Work around a broken local madmom install where DBNBeatTrackingProcessor
# defaults reference undefined min_bpm/max_bpm names during import.
if not hasattr(builtins, 'min_bpm'):
    builtins.min_bpm = 60.0
if not hasattr(builtins, 'max_bpm'):
    builtins.max_bpm = 160.0
if not hasattr(builtins, 'MIN_BPM'):
    builtins.MIN_BPM = 60.0
if not hasattr(builtins, 'MAX_BPM'):
    builtins.MAX_BPM = 160.0

from madmom.features import DBNBeatTrackingProcessor

from wav_lm import WAV_LM
from distil_hubert import DISTILHUBERT
from log_spect import LOG_SPECT
from matplotlib import pyplot as plt

# ----------------------------
# Your model loader (unchanged)
# ----------------------------
def load_model(model_path, device='cuda'):
    import dill
    model_bytes = torch.load(model_path, map_location='cpu')

    _real_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return _real_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    try:
        model = dill.loads(model_bytes)
    finally:
        torch.load = _real_torch_load

    model = model.to(device)
    model.eval()
    return model


# ----------------------------
# Beat conversion (your logic)
# ----------------------------
def predictions_to_beat_times(
    preds,
    method='DBN',
    threshold=0.5,
    sample_rate=16000,
    hop_length=320,
    dbn_min_bpm=None,
    dbn_max_bpm=None
):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    if len(preds.shape) > 1:
        preds = preds[0] if preds.shape[0] == 1 else preds.squeeze()

    preds = preds.flatten()

    if method == 'DBN':
        dbn_kwargs = {'fps': 50}
        if dbn_min_bpm is not None:
            dbn_kwargs['min_bpm'] = dbn_min_bpm
        if dbn_max_bpm is not None:
            dbn_kwargs['max_bpm'] = dbn_max_bpm
        dbn_processor = DBNBeatTrackingProcessor(**dbn_kwargs)
        beat_times = dbn_processor.process_offline(preds)
    elif method == 'threshold':
        beat_frames = np.argwhere(preds >= threshold).flatten()
        beat_times = librosa.frames_to_time(frames=beat_frames, sr=sample_rate, hop_length=hop_length)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'DBN' or 'threshold'")
    return beat_times


# ----------------------------
# Audio loading / resampling
# ----------------------------
def load_audio_mono_keep_original(path: str, target_sr: int = 16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio_mono = audio.mean(axis=1)
    else:
        audio_mono = audio

    audio_mono = audio_mono.astype(np.float32)

    if sr != target_sr:
        audio_16k = librosa.resample(audio_mono, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    else:
        audio_16k = audio_mono

    return audio_mono, sr, audio_16k, target_sr


# ----------------------------
# Reusable embedding processor
# ----------------------------
def make_processor(model_type: str, device: str):
    if model_type == 'wavlm':
        return WAV_LM(device=device)
    if model_type == 'distilhubert':
        return DISTILHUBERT(device=device)
    if model_type == 'log_spec':
        return LOG_SPECT(device=device)
    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------
# Inference on an audio segment (16k mono numpy)
# ----------------------------
@torch.no_grad()
def infer_segment(model, processor, audio_16k_segment: np.ndarray):
    device = next(model.parameters()).device

    embeddings = processor.process_audio(audio_16k_segment)

    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float()
    elif isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach()
    else:
        raise TypeError(f"Unexpected embeddings type: {type(embeddings)}")

    # force shape to (1, seq_len, feat_dim)
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)  # (1, T)
    elif embeddings.ndim == 3:
        if embeddings.shape[0] == 1:
            embeddings = embeddings.squeeze(0)
        elif embeddings.shape[-1] == 1:
            embeddings = embeddings.squeeze(-1)

    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)  # (1, F, T)

    embeddings = embeddings.transpose(1, 2).to(device)  # (1, T, F)

    preds = model(embeddings)
    if isinstance(preds, tuple):
        preds_beat, preds_downbeat = model.final_pred(preds[0], preds[1])
        return preds_beat, preds_downbeat
    else:
        preds_beat = model.final_pred(preds)
        return preds_beat, None


# ----------------------------
# Silero VAD intervals
# ----------------------------
def vad_intervals_silero(
    audio_16k: np.ndarray,
    sr: int = 16000,
    merge_gap: float = 3.0,
    min_len: float = 0.25,
    vad_device: str = "cpu",
    vad_threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100
):
    # load silero once here
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )
    model.to(vad_device).eval()

    (get_speech_timestamps, _, _, _, _) = utils

    wav = torch.from_numpy(audio_16k).to(vad_device).float()

    ts = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms
    )

    intervals = [(t["start"] / sr, t["end"] / sr) for t in ts]
    if not intervals:
        return []

    # merge small gaps
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # filter short segments
    merged = [(s, e) for s, e in merged if (e - s) >= min_len]
    return merged


# ----------------------------
# Beat stitching + silence filling
# ----------------------------
def estimate_period_from_beats(beats: np.ndarray, k: int = 8):
    """Return median beat period (sec) from last/first k diffs, or None."""
    if beats is None or len(beats) < 2:
        return None
    diffs = np.diff(beats)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return None
    if len(diffs) > k:
        diffs = diffs[-k:]
    period = float(np.median(diffs))
    # sanity: 40-240 BPM -> period ~ [0.25, 1.5]
    if not (0.25 <= period <= 1.5):
        return None
    return period


def choose_gap_period(pre_beats, post_beats, k=8):
    pre_p = estimate_period_from_beats(pre_beats[-(k+1):] if pre_beats is not None else None, k=k)
    post_p = None
    if post_beats is not None and len(post_beats) >= 2:
        post_p = estimate_period_from_beats(post_beats[: (k+1)], k=k)

    if pre_p is None and post_p is None:
        return None
    if pre_p is None:
        return post_p
    if post_p is None:
        return pre_p

    # if close, average; else take the one closer to a typical mid tempo (optional)
    if abs(pre_p - post_p) / max(pre_p, post_p) <= 0.15:
        return 0.5 * (pre_p + post_p)
    # otherwise prefer pre (continuity) — you can flip this if you want
    return pre_p


def fill_beat_sequence_gaps(
    beats: np.ndarray,
    k: int = 8,
    gap_threshold: float = 1.5,
    edge_margin: float = 0.02
) -> tuple:
    """
    Fill unusually large inter-beat gaps within the detected beat sequence itself
    (not VAD gaps). Any consecutive pair whose interval > gap_threshold * local_period
    gets interpolated beats inserted.
    Returns (filled_beats, extra_inferred_beats).
    """
    if beats is None or len(beats) < 2:
        return (beats if beats is not None else np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))

    extra = []
    for i in range(len(beats) - 1):
        gap = beats[i + 1] - beats[i]
        pre  = beats[max(0, i - k): i + 1]
        post = beats[i + 1: min(len(beats), i + 1 + k + 1)]
        period = _choose_gap_period(pre, post, k=k)
        if period is None:
            continue
        if gap > gap_threshold * period:
            t = beats[i] + period
            while t < beats[i + 1] - edge_margin:
                if t > beats[i] + edge_margin:
                    extra.append(t)
                t += period

    if len(extra) == 0:
        return beats, np.array([], dtype=np.float64)

    extra_arr = np.array(sorted(set(extra)), dtype=np.float64)
    # remove extras that collide with existing beats (within 20 ms)
    keep = [t for t in extra_arr if np.min(np.abs(beats - t)) > 0.02]
    extra_arr = np.array(keep, dtype=np.float64)

    filled = np.sort(np.unique(np.concatenate([beats, extra_arr]))).astype(np.float64)
    return filled, extra_arr


def fill_silence_with_beats(
    detected_beats: np.ndarray,
    intervals: list,
    audio_duration: float,
    k: int = 8,
    edge_margin: float = 0.02
):
    """
    Fill gaps between consecutive non-silence intervals using tempo from nearby beats.
    Returns inferred_beats (np.ndarray).
    """
    if detected_beats is None:
        detected_beats = np.array([], dtype=np.float64)
    detected_beats = np.array(sorted(detected_beats), dtype=np.float64)

    inferred = []

    # helper: beats within a time window
    def beats_between(a, b):
        mask = (detected_beats >= a) & (detected_beats <= b)
        return detected_beats[mask]

    # gaps: before first interval, between intervals, after last interval
    gaps = []

    if len(intervals) == 0:
        # no speech intervals; nothing to fill safely
        return np.array([], dtype=np.float64)

    # leading gap
    if intervals[0][0] > 0:
        gaps.append((0.0, intervals[0][0]))

    # between intervals
    for (s1, e1), (s2, e2) in zip(intervals[:-1], intervals[1:]):
        if s2 > e1:
            gaps.append((e1, s2))

    # trailing gap
    if intervals[-1][1] < audio_duration:
        gaps.append((intervals[-1][1], audio_duration))

    for g0, g1 in gaps:
        if g1 - g0 <= 1e-3:
            continue

        # get nearby beats: last beats before gap, first beats after gap
        pre = detected_beats[detected_beats < g0]
        post = detected_beats[detected_beats > g1]

        period = choose_gap_period(pre, post, k=k)
        if period is None:
            continue

        # anchor:
        # - if we have a pre beat, start from it forward
        # - else if only post, go backwards from post
        if len(pre) > 0:
            t = float(pre[-1] + period)
            while t < g1 - edge_margin:
                if t > g0 + edge_margin:
                    inferred.append(t)
                t += period
        elif len(post) > 0:
            t = float(post[0] - period)
            while t > g0 + edge_margin:
                if t < g1 - edge_margin:
                    inferred.append(t)
                t -= period

    if len(inferred) == 0:
        return np.array([], dtype=np.float64)

    inferred = np.array(sorted(set(inferred)), dtype=np.float64)

    # remove inferred that collide with detected (within 20ms)
    if len(detected_beats) > 0:
        keep = []
        for t in inferred:
            if np.min(np.abs(detected_beats - t)) > 0.02:
                keep.append(t)
        inferred = np.array(keep, dtype=np.float64)

    return inferred


def detect_beats_per_interval(
    model,
    processor,
    audio_16k: np.ndarray,
    intervals: list,
    method: str,
    threshold: float,
    pad_sec: float = 0.25,
    keep_only_inside_interval: bool = True
):
    beats_all = []
    per_interval = []

    n = len(audio_16k)
    sr = 16000

    for (s, e) in intervals:
        # pad for better boundary stability
        ps = max(0.0, s - pad_sec)
        pe = min(e + pad_sec, n / sr)

        i0 = int(ps * sr)
        i1 = int(pe * sr)
        seg = audio_16k[i0:i1]

        if len(seg) < int(0.5 * sr):
            per_interval.append(np.array([], dtype=np.float64))
            continue

        preds_beat, _ = infer_segment(model, processor, seg)
        bt = predictions_to_beat_times(preds_beat, method=method, threshold=threshold)

        # shift to global time
        bt_global = bt + ps

        if keep_only_inside_interval:
            bt_global = bt_global[(bt_global >= s) & (bt_global <= e)]

        bt_global = np.array(sorted(bt_global), dtype=np.float64)

        per_interval.append(bt_global)
        beats_all.append(bt_global)

    if len(beats_all) == 0:
        return np.array([], dtype=np.float64), per_interval

    beats_all = np.concatenate(beats_all) if any(len(x) > 0 for x in beats_all) else np.array([], dtype=np.float64)
    beats_all = np.array(sorted(beats_all), dtype=np.float64)

    # de-dup within 20ms
    if len(beats_all) > 1:
        cleaned = [beats_all[0]]
        for t in beats_all[1:]:
            if t - cleaned[-1] > 0.02:
                cleaned.append(t)
        beats_all = np.array(cleaned, dtype=np.float64)

    return beats_all, per_interval


# ----------------------------
# Save beat lists (detected + inferred)
# ----------------------------
def save_beats(output_dir: str, base_name: str, detected: np.ndarray, inferred: np.ndarray):
    os.makedirs(output_dir, exist_ok=True)

    detected = np.array(sorted(detected), dtype=np.float64) if detected is not None else np.array([], dtype=np.float64)
    inferred = np.array(sorted(inferred), dtype=np.float64) if inferred is not None else np.array([], dtype=np.float64)

    combined = np.array(sorted(np.concatenate([detected, inferred]) if (len(detected)+len(inferred)) > 0 else []),
                        dtype=np.float64)

    # Save 3 files
    np.save(os.path.join(output_dir, f"{base_name}_beats_detected.npy"), detected)
    np.save(os.path.join(output_dir, f"{base_name}_beats_inferred.npy"), inferred)
    np.save(os.path.join(output_dir, f"{base_name}_beats_all.npy"), combined)

    # Human-readable txt with a label
    txt_path = os.path.join(output_dir, f"{base_name}_beats_all.txt")
    with open(txt_path, "w") as f:
        f.write("# time_sec\tlabel\n")
        di = 0
        ii = 0
        while di < len(detected) or ii < len(inferred):
            next_d = detected[di] if di < len(detected) else None
            next_i = inferred[ii] if ii < len(inferred) else None

            if next_i is None or (next_d is not None and next_d <= next_i):
                f.write(f"{next_d:.6f}\tdetected\n")
                di += 1
            else:
                f.write(f"{next_i:.6f}\tinferred\n")
                ii += 1

    print(f"Saved: {txt_path}")
    return combined

import math

def get_vad_intervals_silero_from_audio(audio_16k, sr=16000,
                                       merge_gap=3.0, min_len=0.25,
                                       vad_threshold=0.5,
                                       min_speech_ms=250, min_silence_ms=100,
                                       vad_device="cpu"):
    """
    Run Silero VAD on a mono 16k waveform (numpy array), return merged intervals in seconds.
    """
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )
    model = model.to(vad_device).eval()
    (get_speech_timestamps, _, _, _, _) = utils

    wav = torch.from_numpy(audio_16k).float().to(vad_device)

    ts = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms
    )

    intervals = [(t["start"] / sr, t["end"] / sr) for t in ts]
    if not intervals:
        return []

    # merge small gaps
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # filter short segments
    merged = [(s, e) for s, e in merged if (e - s) >= min_len]
    return merged


def build_processor(model_type, device):
    if model_type == 'wavlm':
        return WAV_LM(device=device)
    elif model_type == 'distilhubert':
        return DISTILHUBERT(device=device)
    elif model_type == 'log_spec':
        return LOG_SPECT(device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def infer_segment_preds(model, processor, audio_16k_segment):
    """
    audio_16k_segment: numpy mono float waveform at 16k
    returns preds_beat (torch.Tensor), preds_downbeat (optional or None)
    """
    device = next(model.parameters()).device

    embeddings = processor.process_audio(audio_16k_segment)

    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float()
    elif isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach()
    else:
        raise TypeError(f"Unexpected embeddings type: {type(embeddings)}")

    # ensure (1, F, T)
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    elif embeddings.ndim == 3:
        if embeddings.shape[0] == 1:
            embeddings = embeddings.squeeze(0)
        elif embeddings.shape[-1] == 1:
            embeddings = embeddings.squeeze(-1)

    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)  # (1, F, T)

    embeddings = embeddings.transpose(1, 2).to(device)  # (1, T, F)

    preds = model(embeddings)
    if isinstance(preds, tuple):
        preds_beat, preds_downbeat = model.final_pred(preds[0], preds[1])
        return preds_beat, preds_downbeat
    else:
        preds_beat = model.final_pred(preds)
        return preds_beat, None


def _median_period(beats, k=8):
    if beats is None or len(beats) < 2:
        return None
    diffs = np.diff(beats)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return None
    diffs = diffs[-k:] if len(diffs) > k else diffs
    p = float(np.median(diffs))
    # sanity: 40-240 BPM -> [0.25, 1.5] sec
    if not (0.25 <= p <= 1.5):
        return None
    return p


def _choose_gap_period(pre_beats, post_beats, k=8):
    pre_p = _median_period(pre_beats[-(k+1):] if pre_beats is not None else None, k=k) if (pre_beats is not None and len(pre_beats) > 0) else None
    post_p = _median_period(post_beats[:(k+1)] if post_beats is not None else None, k=k) if (post_beats is not None and len(post_beats) > 0) else None

    if pre_p is None and post_p is None:
        return None
    if pre_p is None:
        return post_p
    if post_p is None:
        return pre_p

    # if close, average; else prefer continuity (pre)
    if abs(pre_p - post_p) / max(pre_p, post_p) <= 0.15:
        return 0.5 * (pre_p + post_p)
    return pre_p


def infer_beats_vad_stitch_fill(
    model,
    audio_path,
    model_type,
    method='DBN',
    threshold=0.5,
    lock_first_bpm=True,
    first_bpm_margin=10.0,
    use_vad=True,
    vad_threshold=0.5,
    vad_merge_gap=3.0,
    vad_min_len=0.25,
    vad_min_speech_ms=250,
    vad_min_silence_ms=100,
    vad_device="cpu",
    interval_pad=0.25,
    vad_beat_buffer=0.1,
    fill_silence=True,
    fill_k=8,
    fps=50
):
    """
    Returns:
      beat_times_all (np.ndarray): detected + inferred beats in global seconds
      global_preds (np.ndarray): full-length prediction curve at fps (50), zeros in silence

    vad_beat_buffer: extend each VAD interval by this many seconds on each side
        before slicing audio and filtering beats. Prevents edge beats from being
        clipped by tight VAD boundaries. Default 0.1s.
    """
    # ---- load original (for save_results) ----
    audio, sr = sf.read(audio_path)
    original_audio = audio.copy()
    original_sr = sr

    if audio.ndim > 1:
        audio_mono = audio.mean(axis=1)
    else:
        audio_mono = audio

    # ---- resample to 16k for VAD + model ----
    if sr != 16000:
        audio_16k = librosa.resample(audio_mono.astype(np.float32), orig_sr=sr, target_sr=16000).astype(np.float32)
    else:
        audio_16k = audio_mono.astype(np.float32)

    dur_sec = len(audio_16k) / 16000.0

    # ---- intervals ----
    if use_vad:
        intervals = get_vad_intervals_silero_from_audio(
            audio_16k, sr=16000,
            merge_gap=vad_merge_gap,
            min_len=vad_min_len,
            vad_threshold=vad_threshold,
            min_speech_ms=vad_min_speech_ms,
            min_silence_ms=vad_min_silence_ms,
            vad_device=vad_device
        )
    else:
        intervals = [(0.0, dur_sec)]

    # ---- extend VAD intervals by vad_beat_buffer on each side ----
    if vad_beat_buffer > 0 and use_vad:
        intervals = [(max(0.0, s - vad_beat_buffer), min(dur_sec, e + vad_beat_buffer))
                     for s, e in intervals]

    # ---- processor once ----
    device = next(model.parameters()).device
    processor = build_processor(model_type, device=device)

    # ---- global prediction array at 50 fps ----
    total_frames = int(math.ceil(dur_sec * fps))
    global_preds = np.zeros((total_frames,), dtype=np.float32)

    detected_beats = []

    # ---- Pass 1: detect beats for every segment (unconstrained) ----
    # Store per-segment preds + beats so we can selectively re-detect in pass 2
    # without re-running the neural network.
    seg_results = []  # list of dicts: {s, e, ps, pe, preds_beat, bt_global, bpm}

    for (s, e) in intervals:
        ps = max(0.0, s - interval_pad)
        pe = min(dur_sec, e + interval_pad)

        i0 = int(ps * 16000)
        i1 = int(pe * 16000)
        seg = audio_16k[i0:i1]

        if len(seg) < int(0.5 * 16000):
            seg_results.append({'s': s, 'e': e, 'ps': ps, 'pe': pe,
                                'preds_beat': None, 'bt_global': np.array([], dtype=np.float64),
                                'bpm': None})
            continue

        preds_beat, _ = infer_segment_preds(model, processor, seg)

        # unconstrained beat detection
        bt = predictions_to_beat_times(preds_beat, method=method, threshold=threshold)
        bt_global = bt + ps
        bt_global = bt_global[(bt_global >= s) & (bt_global <= e)]
        bt_global = np.array(sorted(bt_global), dtype=np.float64)

        # estimate per-segment BPM
        seg_bpm = None
        if len(bt_global) >= 2:
            period = _median_period(bt_global, k=fill_k)
            if period is not None:
                seg_bpm = 60.0 / period

        seg_results.append({'s': s, 'e': e, 'ps': ps, 'pe': pe,
                            'preds_beat': preds_beat, 'bt_global': bt_global,
                            'bpm': seg_bpm})

    # ---- Determine reference BPM from the smallest per-segment BPM ----
    valid_bpms = [r['bpm'] for r in seg_results if r['bpm'] is not None]

    if valid_bpms:
        min_bpm = min(valid_bpms)
        print(f"[bpm_lock] per-segment BPMs: {[f'{b:.1f}' for b in valid_bpms]}")
        print(f"[bpm_lock] smallest BPM = {min_bpm:.2f}")
    else:
        min_bpm = None

    # ---- Pass 2: re-detect segments whose BPM >= 1.2 * min_bpm ----
    for r in seg_results:
        if r['preds_beat'] is None:
            continue

        needs_redetect = (
            method == 'DBN'
            and lock_first_bpm
            and min_bpm is not None
            and r['bpm'] is not None
            and r['bpm'] >= 1.2 * min_bpm
        )

        if needs_redetect:
            dbn_min = max(1.0, min_bpm - first_bpm_margin)
            dbn_max = min_bpm + first_bpm_margin
            print(f"[bpm_lock] re-detecting segment [{r['s']:.2f}-{r['e']:.2f}]: "
                  f"original {r['bpm']:.1f} BPM >= 1.2*{min_bpm:.1f}={1.2*min_bpm:.1f}, "
                  f"constraining to [{dbn_min:.1f}, {dbn_max:.1f}]")

            bt = predictions_to_beat_times(
                r['preds_beat'], method=method, threshold=threshold,
                dbn_min_bpm=dbn_min, dbn_max_bpm=dbn_max
            )
            bt_global = bt + r['ps']
            bt_global = bt_global[(bt_global >= r['s']) & (bt_global <= r['e'])]
            r['bt_global'] = np.array(sorted(bt_global), dtype=np.float64)

            # update BPM
            if len(r['bt_global']) >= 2:
                period = _median_period(r['bt_global'], k=fill_k)
                if period is not None:
                    r['bpm'] = 60.0 / period
                    print(f"[bpm_lock]   -> re-detected BPM: {r['bpm']:.1f}")

        if len(r['bt_global']) > 0:
            detected_beats.append(r['bt_global'])

        # ---- paste predictions into global_preds (keep only frames inside [s,e]) ----
        preds_np = r['preds_beat'].detach().cpu().numpy().reshape(-1)
        frame_times = r['ps'] + (np.arange(len(preds_np)) / fps)

        inside = (frame_times >= r['s']) & (frame_times <= r['e'])
        if np.any(inside):
            ft = frame_times[inside]
            pv = preds_np[inside]
            idx = np.clip(np.round(ft * fps).astype(int), 0, total_frames - 1)
            np.maximum.at(global_preds, idx, pv.astype(np.float32))

    if len(detected_beats) > 0:
        detected_beats = np.sort(np.concatenate(detected_beats).astype(np.float64))
        # de-dup within 20 ms
        cleaned = [detected_beats[0]]
        for t in detected_beats[1:]:
            if t - cleaned[-1] > 0.02:
                cleaned.append(t)
        detected_beats = np.array(cleaned, dtype=np.float64)
    else:
        detected_beats = np.array([], dtype=np.float64)

    # ---- snapshot: beats before any gap-filling (vocal segments only) ----
    detected_beats_raw = detected_beats.copy()

    inferred_beats = np.array([], dtype=np.float64)

    # ---- fill intra-sequence gaps (missed beats within vocal segments) ----
    if len(detected_beats) > 1:
        detected_beats, seq_inferred = fill_beat_sequence_gaps(detected_beats, k=fill_k)
        inferred_beats = np.concatenate([inferred_beats, seq_inferred])
        print(f"[fill_seq_gaps] filled {len(seq_inferred)} intra-sequence beats")

    print(f"[fill_silence] VAD intervals ({len(intervals)}): {intervals[:5]}{'...' if len(intervals)>5 else ''}")
    print(f"[fill_silence] detected_beats: {len(detected_beats)} beats, fill_silence={fill_silence}")

    # ---- fill silent gaps (non-vocal regions) ----
    if fill_silence and len(intervals) > 0 and len(detected_beats) > 1:
        intervals_sorted = sorted(intervals, key=lambda x: x[0])

        gaps = []
        if intervals_sorted[0][0] > 0:
            gaps.append((0.0, intervals_sorted[0][0]))
        for (s1, e1), (s2, e2) in zip(intervals_sorted[:-1], intervals_sorted[1:]):
            if s2 > e1:
                gaps.append((e1, s2))
        if intervals_sorted[-1][1] < dur_sec:
            gaps.append((intervals_sorted[-1][1], dur_sec))

        print(f"[fill_silence] gaps to fill: {gaps}")

        inferred = []
        edge_margin = 0.02

        for g0, g1 in gaps:
            if g1 - g0 <= 1e-3:
                continue

            pre = detected_beats[detected_beats < g0]
            post = detected_beats[detected_beats > g1]
            period = _choose_gap_period(pre, post, k=fill_k)
            print(f"[fill_silence] gap ({g0:.2f},{g1:.2f}): pre={len(pre)} beats, post={len(post)} beats, period={period}")
            if period is None:
                continue

            if len(pre) > 0:
                t = float(pre[-1] + period)
                while t < g1 - edge_margin:
                    if t > g0 + edge_margin:
                        inferred.append(t)
                    t += period
            elif len(post) > 0:
                t = float(post[0] - period)
                while t > g0 + edge_margin:
                    if t < g1 - edge_margin:
                        inferred.append(t)
                    t -= period

        print(f"[fill_silence] inferred {len(inferred)} beats before dedup/collision filter")

        if len(inferred) > 0:
            inferred = np.array(sorted(set(inferred)), dtype=np.float64)
            # remove inferred too close to detected
            keep = []
            for t in inferred:
                if np.min(np.abs(detected_beats - t)) > 0.02:
                    keep.append(t)
            inferred_beats = np.array(keep, dtype=np.float64)
        print(f"[fill_silence] final inferred_beats: {len(inferred_beats)}")

    # ---- combine beats ----
    beat_times_all = np.sort(
        np.unique(np.concatenate([detected_beats, inferred_beats]) if (len(detected_beats) + len(inferred_beats)) > 0 else np.array([]))
    ).astype(np.float64)

    # De-dup near-duplicates: fill_beat_sequence_gaps and fill_silence can both
    # fill the same gap with slightly different periods, creating interleaved
    # beats only 0.02-0.09s apart. Merge any pair closer than 30% of the median
    # IBI, keeping the one that was detected (or the first if both inferred).
    if len(beat_times_all) > 1:
        med_ibi = float(np.median(np.diff(beat_times_all)))
        min_gap = 0.3 * med_ibi
        cleaned = [beat_times_all[0]]
        for t in beat_times_all[1:]:
            if t - cleaned[-1] < min_gap:
                # keep whichever is a detected beat; if both or neither, keep first
                prev_is_det = len(detected_beats_raw) > 0 and np.min(np.abs(detected_beats_raw - cleaned[-1])) < 0.025
                curr_is_det = len(detected_beats_raw) > 0 and np.min(np.abs(detected_beats_raw - t)) < 0.025
                if curr_is_det and not prev_is_det:
                    cleaned[-1] = t  # replace inferred with detected
                # else keep the one already in cleaned
            else:
                cleaned.append(t)
        n_deduped = len(beat_times_all) - len(cleaned)
        if n_deduped > 0:
            print(f"[dedup] removed {n_deduped} near-duplicate beats (min_gap={min_gap:.3f}s)")
        beat_times_all = np.array(cleaned, dtype=np.float64)

    # clip to ORIGINAL duration for safety
    orig_dur = (len(audio_mono) / float(original_sr)) if original_sr > 0 else dur_sec
    beat_times_all = beat_times_all[(beat_times_all >= 0.0) & (beat_times_all <= orig_dur + 1e-3)]

    return beat_times_all, global_preds, original_audio, original_sr, detected_beats_raw
def plot_predictions(preds, output_path, sample_rate=16000, hop_length=320, fps=50):
    """
    Plot raw model predictions (probabilities) over time.
    
    Parameters
    ----------
    preds : torch.Tensor or np.ndarray
        Model predictions (probabilities per frame)
    output_path : str
        Path to save the plot
    sample_rate : int
        Audio sample rate (default: 16000)
    hop_length : int
        Hop length used in feature extraction (default: 320)
    fps : int
        Frames per second for predictions (default: 50)
    """
    # Convert to numpy and handle batch dimension
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    
    # Handle batch dimension - take first batch if present
    if len(preds.shape) > 1:
        preds = preds[0] if preds.shape[0] == 1 else preds.squeeze()
    
    # Ensure 1D
    preds = preds.flatten()
    
    # Create time axis for predictions
    # Using fps=50: each frame represents 1/50 seconds
    num_frames = len(preds)
    time_axis = np.arange(num_frames) / fps
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot predictions
    ax.plot(time_axis, preds, linewidth=1.5, color='blue', label='Beat Probability', alpha=0.8)
    
    # Add threshold line if needed (optional, can be commented out)
    # ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.5)')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Beat Probability', fontsize=12)
    ax.set_title('Raw Model Predictions (Before Beat Time Conversion)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set y-axis limits to show full probability range
    ax.set_ylim(-0.05, 1.05)
    
    # Set x-axis limits
    ax.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Predictions plot saved to: {output_path}")
def plot_audio_with_beats(audio, sample_rate, beat_times, output_path):
    """
    Plot audio waveform with beat predictions marked as vertical dashed lines.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sample_rate : int
        Sample rate of the audio
    beat_times : np.ndarray
        Beat times in seconds
    output_path : str
        Path to save the plot
    """
    # Ensure audio is mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Create time axis
    duration = len(audio) / sample_rate
    time_axis = np.linspace(0, duration, len(audio))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Plot waveform
    ax.plot(time_axis, audio, linewidth=0.5, alpha=0.7, color='blue', label='Audio')
    
    # Plot beat predictions as vertical dashed lines
    for beat_time in beat_times:
        if beat_time <= duration:  # Only plot beats within audio duration
            ax.axvline(x=beat_time, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Audio Waveform with Beat Predictions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set x-axis limits
    ax.set_xlim(0, duration)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
def generate_metronome_click(sample_rate, duration=0.01, frequency=1000, volume=0.3):
    """
    Generate a metronome click sound.
    
    Parameters
    ----------
    sample_rate : int
        Sample rate
    duration : float
        Duration of click in seconds (default: 0.01)
    frequency : float
        Frequency of the click tone in Hz (default: 1000)
    volume : float
        Volume/amplitude of the click (default: 0.3)
    
    Returns
    -------
    click : np.ndarray
        Metronome click sound
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # Generate a click with a quick attack and decay envelope
    # Using a combination of sine wave and envelope
    click = np.sin(2 * np.pi * frequency * t)
    
    # Apply envelope: quick attack, exponential decay
    envelope = np.exp(-t * 50)  # Exponential decay
    click = click * envelope
    
    # Normalize and scale by volume
    click = click * volume
    
    return click

def add_metronome_to_audio(audio, sample_rate, beat_times, click_duration=0.01, click_frequency=1000, click_volume=0.3):
    """
    Add metronome clicks to audio at beat times.
    
    Parameters
    ----------
    audio : np.ndarray
        Original audio waveform
    sample_rate : int
        Sample rate of the audio
    beat_times : np.ndarray
        Beat times in seconds
    click_duration : float
        Duration of each click in seconds (default: 0.01)
    click_frequency : float
        Frequency of click tone in Hz (default: 1000)
    click_volume : float
        Volume of clicks (default: 0.3)
    
    Returns
    -------
    audio_with_metronome : np.ndarray
        Audio with metronome clicks added
    """
    # Ensure audio is mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Make a copy to avoid modifying original
    audio_with_metronome = audio.copy()
    
    # Generate click sound
    click = generate_metronome_click(sample_rate, click_duration, click_frequency, click_volume)
    click_length = len(click)
    
    # Add clicks at each beat time
    for beat_time in beat_times:
        # Convert beat time to sample index
        beat_sample = int(beat_time * sample_rate)
        
        # Make sure the click fits within the audio
        if beat_sample + click_length <= len(audio_with_metronome):
            # Mix the click into the audio (simple addition)
            audio_with_metronome[beat_sample:beat_sample + click_length] += click
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio_with_metronome))
    if max_val > 1.0:
        audio_with_metronome = audio_with_metronome / max_val
    
    return audio_with_metronome
def save_results(audio, sample_rate, beat_times, predictions, audio_path, output_dir, 
                 metronome_volume=0.3, metronome_frequency=1000, metronome_duration=0.01):
    """
    Save audio, beat predictions, and plot to output directory.
    
    Parameters
    ----------
    audio : np.ndarray
        Original audio waveform
    sample_rate : int
        Sample rate of the audio
    beat_times : np.ndarray
        Beat times in seconds
    predictions : np.ndarray or torch.Tensor
        Raw model predictions (probabilities)
    audio_path : str
        Original audio file path (for naming)
    output_dir : str
        Output directory to save results
    metronome_volume : float
        Volume of metronome clicks (0.0-1.0, default: 0.3)
    metronome_frequency : float
        Frequency of metronome click tone in Hz (default: 1000)
    metronome_duration : float
        Duration of each metronome click in seconds (default: 0.01)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename from audio path
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Save audio (copy original or save processed)
    audio_output_path = os.path.join(output_dir, f"{base_name}_audio.wav")
    # Ensure audio is mono and in correct format
    if audio.ndim > 1:
        audio_to_save = audio.mean(axis=1)
    else:
        audio_to_save = audio
    sf.write(audio_output_path, audio_to_save, sample_rate)
    print(f"Audio saved to: {audio_output_path}")
    
    # Save beat times as text file
    beat_times_path = os.path.join(output_dir, f"{base_name}_beat_times.txt")
    with open(beat_times_path, 'w') as f:
        f.write("# Beat times in seconds\n")
        for beat_time in beat_times:
            f.write(f"{beat_time:.6f}\n")
    print(f"Beat times saved to: {beat_times_path}")
    
    # Save beat times as numpy array
    beat_times_npy_path = os.path.join(output_dir, f"{base_name}_beat_times.npy")
    np.save(beat_times_npy_path, beat_times)
    print(f"Beat times (numpy) saved to: {beat_times_npy_path}")
    
    # Save raw predictions
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.detach().cpu().numpy()
    else:
        predictions_np = predictions
    predictions_path = os.path.join(output_dir, f"{base_name}_predictions.npy")
    np.save(predictions_path, predictions_np)
    print(f"Raw predictions saved to: {predictions_path}")
    
    # Create and save predictions plot (before conversion to beat times)
    predictions_plot_path = os.path.join(output_dir, f"{base_name}_predictions_plot.png")
    plot_predictions(predictions, predictions_plot_path, sample_rate=sample_rate)
    
    # Create and save audio with beats plot
    plot_path = os.path.join(output_dir, f"{base_name}_plot.png")
    plot_audio_with_beats(audio_to_save, sample_rate, beat_times, plot_path)
    
    # Add metronome clicks and save
    audio_with_metronome = add_metronome_to_audio(
        audio_to_save, 
        sample_rate, 
        beat_times,
        click_duration=metronome_duration,
        click_frequency=metronome_frequency,
        click_volume=metronome_volume
    )
    metronome_output_path = os.path.join(output_dir, f"{base_name}_with_metronome.wav")
    sf.write(metronome_output_path, audio_with_metronome, sample_rate)
    print(f"Audio with metronome saved to: {metronome_output_path}")
    
    return output_dir
def main():
    parser = argparse.ArgumentParser(description='Inference for Singing-Vocal-Beat-Tracking + VAD + Fill Silence (uses original save_results)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_type', type=str, default='distilhubert', choices=['wavlm','distilhubert','log_spec'], help='Type of the model')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    # beat extraction
    parser.add_argument('--method', type=str, default='DBN', choices=['DBN', 'threshold'],
                        help='Method to extract beats: DBN or threshold')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for beat detection when method=threshold')
    parser.add_argument('--lock_first_bpm', dest='lock_first_bpm', action='store_true',
                        help='When using DBN, constrain later segments to the first valid segment BPM +/- margin')
    parser.add_argument('--no_lock_first_bpm', dest='lock_first_bpm', action='store_false',
                        help='Disable BPM locking across VAD segments')
    parser.set_defaults(lock_first_bpm=True)
    parser.add_argument('--first_bpm_margin', type=float, default=10.0,
                        help='Allowed BPM deviation around the first valid segment when BPM locking is enabled')

    # output
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Output directory to save results (default: predictions)')

    # metronome (same as before)
    parser.add_argument('--metronome_volume', type=float, default=0.3)
    parser.add_argument('--metronome_frequency', type=float, default=1000)
    parser.add_argument('--metronome_duration', type=float, default=0.01)

    # VAD options
    parser.add_argument('--use_vad', action='store_true', help='Enable silero VAD to remove silence')
    parser.add_argument('--vad_device', type=str, default='cpu')
    parser.add_argument('--vad_threshold', type=float, default=0.5)
    parser.add_argument('--vad_merge_gap', type=float, default=0.5)
    parser.add_argument('--vad_min_len', type=float, default=0.25)
    parser.add_argument('--vad_min_speech_ms', type=int, default=250)
    parser.add_argument('--vad_min_silence_ms', type=int, default=100)

    # interval padding
    parser.add_argument('--interval_pad', type=float, default=0.25)
    parser.add_argument('--vad_beat_buffer', type=float, default=0.1,
                        help='Extend each VAD interval by this many seconds on each side so edge beats are not clipped (default: 0.1)')

    # fill silence
    parser.add_argument('--fill_silence', action='store_true', help='Infer beats during silent gaps using nearby tempo')
    parser.add_argument('--fill_k', type=int, default=8)

    args = parser.parse_args()

    model = load_model(args.model_path, args.device)

    beat_times_all, global_preds, original_audio, original_sr, detected_beats_raw = infer_beats_vad_stitch_fill(
        model=model,
        audio_path=args.audio_path,
        model_type=args.model_type,
        method=args.method,
        threshold=args.threshold,
        lock_first_bpm=args.lock_first_bpm,
        first_bpm_margin=args.first_bpm_margin,
        use_vad=args.use_vad,
        vad_threshold=args.vad_threshold,
        vad_merge_gap=args.vad_merge_gap,
        vad_min_len=args.vad_min_len,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_device=args.vad_device,
        interval_pad=args.interval_pad,
        vad_beat_buffer=args.vad_beat_buffer,
        fill_silence=args.fill_silence,
        fill_k=args.fill_k,
        fps=50
    )

    print(f"Total beats (detected+inferred): {len(beat_times_all)}")
    print(f"Beat times preview: {beat_times_all[:10]}..." if len(beat_times_all) > 10 else beat_times_all)

    # same output structure as your original script
    audio_basename = os.path.splitext(os.path.basename(args.audio_path))[0]
    output_dir = os.path.join(args.output_dir, audio_basename)
    os.makedirs(output_dir, exist_ok=True)

    # Save detected-only beats (vocal segments, before any gap-filling)
    detected_raw_path = os.path.join(output_dir, f"{audio_basename}_beat_times_detected_only.txt")
    with open(detected_raw_path, 'w') as f:
        f.write("# Beat times in seconds (vocal segments only, before gap-filling)\n")
        for t in detected_beats_raw:
            f.write(f"{t:.6f}\n")
    print(f"Detected-only beat times saved to: {detected_raw_path}")

    # ✅ use your original save_results exactly
    save_results(
        original_audio,
        original_sr,
        beat_times_all,
        global_preds,          # <- full-length 50fps predictions (0 in silence)
        args.audio_path,
        output_dir,
        metronome_volume=args.metronome_volume,
        metronome_frequency=args.metronome_frequency,
        metronome_duration=args.metronome_duration
    )

    print(f"\nAll results saved to: {output_dir}")
    return beat_times_all

if __name__ == '__main__':
    main()

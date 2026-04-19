from btc_chords import Chords
import bisect
import torch
import torch.nn.functional as F
import numpy as np
from torchaudio import transforms as T
import torchaudio
from utils.stable_audio_dataset_utils import Stereo, PhaseFlipper

def load_audio_file(filename, target_sr=44100, target_samples=2097152, segment_starts=0):
    try:
        audio, in_sr = torchaudio.load(filename)    
        # Resample if necessary
        if in_sr != target_sr:
            resampler = T.Resample(in_sr, target_sr)
            audio = resampler(audio)
        augs = torch.nn.Sequential(
            PhaseFlipper(),
        )
        audio = augs(audio)
        audio = audio.clamp(-1, 1)
        encoding = torch.nn.Sequential(
            Stereo(),
        )
        audio = encoding(audio)
        audio = audio[:, int(segment_starts*44100):int(segment_starts*44100)+target_samples]
        return audio
    except RuntimeError:
        print(f"Failed to decode audio file: {filename}")
        return None
def _rms(x: torch.Tensor, eps: float = 1e-12):
    # x: shape [T] or [C, T]
    return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

def _to_dbfs(rms: torch.Tensor, eps: float = 1e-12):
    return 20.0 * torch.log10(torch.clamp(rms, min=eps))

def _from_db(db: float):
    return 10.0 ** (db / 20.0)

def loudness_match(x: torch.Tensor, target_dbfs: float = -18.0):
    """
    RMS-loudness normalize to target dBFS per-channel.
    x: [T] or [C, T] float32 in [-1, 1]
    """
    mono = (x.dim() == 1)
    if mono:
        x = x.unsqueeze(0)  # [1, T]

    rms = _rms(x)                      # [C, 1]
    cur_db = _to_dbfs(rms)             # [C, 1]
    gain_db = target_dbfs - cur_db     # [C, 1]
    gain = _from_db(gain_db)           # [C, 1]
    y = x * gain

    return y.squeeze(0) if mono else y

def peak_normalize(x: torch.Tensor, peak_dbfs: float = -1.0, eps: float = 1e-12):
    """
    Peak-normalize so the absolute peak hits (peak_dbfs).
    """
    peak = torch.max(torch.abs(x))
    if peak < eps:
        return x  # silence stays silence
    target_peak = _from_db(peak_dbfs)
    return x * (target_peak / peak)

def pad_or_trim(a: torch.Tensor, b: torch.Tensor):
    """
    Make both tensors same length along last dim by padding end with zeros.
    Supports [T] or [C, T]. Assumes same #channels; handle beforehand if not.
    """
    Ta, Tb = a.shape[-1], b.shape[-1]
    T = max(Ta, Tb)
    def pad(x, T):
        if x.shape[-1] == T:
            return x
        pad_len = T - x.shape[-1]
        pad_shape = list(x.shape[:-1]) + [pad_len]
        return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=-1)
    return pad(a, T), pad(b, T)

def mix_audio(a: torch.Tensor,
              b: torch.Tensor,
              target_dbfs: float = -18.0,
              out_peak_dbfs: float = -1.0):
    """
    Mix two audio tensors safely.

    a, b: [T] mono or [C, T] multi-channel, float32 in [-1, 1]
    target_dbfs: per-track RMS loudness target before mixing (e.g., -18 dBFS)
    out_peak_dbfs: peak ceiling for the final mix (e.g., -1 dBFS)
    """
    # 1) Basic checks (dtype/range are caller’s responsibility; shown below)
    assert a.dim() in (1,2) and b.dim() in (1,2), "Use [T] or [C, T]"
    # If channel counts differ (e.g., mono vs stereo), upmix mono to stereo:
    if a.dim() == 1 and b.dim() == 2:
        a = a.unsqueeze(0).expand(b.shape[0], -1)
    if b.dim() == 1 and a.dim() == 2:
        b = b.unsqueeze(0).expand(a.shape[0], -1)
    # Now channels must match
    if a.dim() == 2 and b.dim() == 2:
        assert a.shape[0] == b.shape[0], "Channel count mismatch"

    # 2) Make same length
    a, b = pad_or_trim(a, b)

    # 3) Loudness-match each stem
    a_n = loudness_match(a, target_dbfs=target_dbfs)
    b_n = loudness_match(b, target_dbfs=target_dbfs)

    # 4) Mix (simple sum). If you want a 50/50 “equal-power” style, divide by sqrt(2).
    mix = a_n + b_n

    # 5) Peak-normalize with headroom
    mix = peak_normalize(mix, peak_dbfs=out_peak_dbfs)

    # 6) Safety clamp
    mix = torch.clamp(mix, -1.0, 1.0)
    return mix



def pad_to_match(a: torch.Tensor, b: torch.Tensor, len=0):
    """
    Pad the shorter tensor (along the last dimension) with zeros
    so that a and b have the same length.
    """
    len_a, len_b = a.shape[-1], b.shape[-1]
    if len_a < len:
        pad = (0, len - len_a)  # (left, right)
        a = F.pad(a, pad)
    if len_b < len:
        pad = (0, len - len_b)  # (left, right)
        b = F.pad(b, pad)
    if len < len_a:
        a = a[:, :len]
    if len < len_b:
        b = b[:, :len]
    return a, b
def extract_chords_lab(chord_path, segment_starts=0):
    CHORDS = Chords()
    with open(chord_path, 'r') as f:
        chord_infos = f.read().splitlines()
    chroma = np.zeros((12, 2097152))
    segment_ends = segment_starts + 2097152 / 44100
    for info in chord_infos:
        s, t, chord = info.split(' ')
        s = float(s) - segment_starts
        t = float(t) - segment_starts
        if t >  2097152 / 44100 and s < 2097152 / 44100:
            t = 2097152 / 44100
        elif t >=  2097152 / 44100 and s > 2097152 / 44100:
            continue
        elif t < 0 and s < 0:
            continue
        elif t > 0 and s < 0:
            s = 0
        mhot = CHORDS.chord(chord)
        final_vec = np.roll(mhot[2], mhot[0])
        final_vec = final_vec[..., None]  # shape (12, 1)
        chroma[:, int(float(s)*44100): int(float(t)*44100)] = final_vec
    s, end_time, chord = chord_infos[-1].split(' ')
    chroma = torch.from_numpy(chroma).unsqueeze(0).float().cuda()  # shape (1, 12, 2097152)
    chroma = F.interpolate(chroma, size=1024, mode='linear', align_corners=False)  # shape (1, 12, 4756)
    return chroma, end_time
def extract_chords_lab_full(chord_path, segment_starts=0, sr=44100, frames_per_sec=None):
    """
    Like extract_chords_lab but processes the full-length chord file without
    the 2097152-sample cap. The output length is determined by the last chord's
    end time.

    Args:
        chord_path:      path to chord txt (format: "start end chord" per line)
        segment_starts:  global offset to subtract from all timestamps (seconds)
        sr:              sample rate used to convert seconds -> samples (default 44100)
        frames_per_sec:  if set, interpolate output to int(duration * frames_per_sec)
                         frames; if None, keep one frame per sample (full resolution)

    Returns:
        chroma:    torch.Tensor on cuda, shape (1, 12, T)
        end_time:  str, end time of last chord entry
    """
    CHORDS = Chords()
    with open(chord_path, 'r') as f:
        chord_infos = [l for l in f.read().splitlines() if l.strip()]

    # determine full duration from last entry
    _, end_time, _ = chord_infos[-1].split(' ')
    duration_sec = float(end_time) - segment_starts
    total_samples = int(duration_sec * sr) + 1  # +1 to avoid off-by-one

    chroma = np.zeros((12, total_samples), dtype=np.float32)

    for info in chord_infos:
        s, t, chord = info.split(' ')
        s = float(s) - segment_starts
        t = float(t) - segment_starts
        if t < 0 or s < 0 and t < 0:
            continue
        s = max(s, 0.0)
        t = min(t, duration_sec)
        if s >= t:
            continue
        mhot = CHORDS.chord(chord)
        final_vec = np.roll(mhot[2], mhot[0]).astype(np.float32)  # (12,)
        chroma[:, int(s * sr): int(t * sr)] = final_vec[..., None]

    chroma_t = torch.from_numpy(chroma).unsqueeze(0).float().cuda()  # (1, 12, total_samples)

    if frames_per_sec is not None:
        out_frames = max(1, int(duration_sec * frames_per_sec))
        chroma_t = F.interpolate(chroma_t, size=out_frames, mode='linear', align_corners=False)

    return chroma_t, end_time


def sublist_between(arr, a, b, eps=1e-6):
    """Return arr elements in [a, b) using indices (fast; arr must be sorted)."""
    lo = bisect.bisect_left(arr, a - eps)
    hi = bisect.bisect_left(arr, b - eps)   # half-open: up to but not including b
    return arr[lo:hi]
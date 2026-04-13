import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class Resample(nn.Module):
    """Pure-torch linear resampler (replaces torchaudio.transforms.Resample)."""
    def __init__(self, orig_freq: int, new_freq: int, **kwargs):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        orig_len = waveform.shape[-1]
        new_len = int(round(orig_len * self.new_freq / self.orig_freq))
        x = waveform.reshape(1, 1, -1) if waveform.dim() == 1 else waveform.unsqueeze(0)
        out = F.interpolate(x.float(), size=new_len, mode='linear', align_corners=False)
        return out.reshape(waveform.shape[:-1] + (new_len,))

from .constants import *
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0


def resample_align_curve(f0: np.ndarray, orig_time_step: float, target_time_step: float, length: int) -> np.ndarray:
    """Resample an F0 curve from orig_time_step to target_time_step, aligned to `length` frames."""
    if orig_time_step == target_time_step:
        if len(f0) >= length:
            return f0[:length].astype(np.float32)
        return np.pad(f0, (0, length - len(f0)), mode='constant').astype(np.float32)
    orig_times = np.arange(len(f0)) * orig_time_step
    target_times = np.arange(length) * target_time_step
    return np.interp(target_times, orig_times, f0).astype(np.float32)


def collate_1d(tensors: list, pad_value: float = 0.0) -> torch.Tensor:
    """Pad a list of 1-D tensors to the same length and stack."""
    max_len = max(t.shape[0] for t in tensors)
    result = torch.stack([
        F.pad(t, (0, max_len - t.shape[0]), value=pad_value) for t in tensors
    ])
    return result


class RMVPE:
    def __init__(self, model_path, hop_length=160, device=None):
        self.resample_kernel = {}
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = E2E0(4, 1, (2, 2)).eval().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)
        self.hop_length = hop_length

    @torch.no_grad()
    def mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0, cents = to_viterbi_f0(hidden, thred=thred)
        else:
            f0, cents = to_local_average_f0(hidden, thred=thred)
        return f0, cents

    def postprocess(self, f0, fmin=50, fmax=1000, min_gap=2):
        f0[f0 < fmin] = 0
        f0[f0 > fmax] = 0
        # Eliminate short glitches
        for idx in range(f0.shape[0] - min_gap - 1):
            if f0[idx] == 0 and f0[idx + min_gap + 1] == 0 and np.sum(f0[idx: idx + min_gap + 2]) > 0:
                f0[idx: idx + min_gap + 2] = 0
        return f0

    def _get_resampler(self, sample_rate):
        key = str(sample_rate)
        if key not in self.resample_kernel:
            self.resample_kernel[key] = Resample(sample_rate, 16000, lowpass_filter_width=128).to(self.device)
        return self.resample_kernel[key]

    @torch.no_grad()
    def infer_from_audio(self, audio: np.ndarray, sample_rate=16000, thred=0.03, use_viterbi=False):
        """Extract F0 from a single waveform. Returns (f0, cents) each shape (T,)."""
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        if sample_rate != 16000:
            audio_t = self._get_resampler(sample_rate)(audio_t)
        mel = self.mel_extractor(audio_t, center=True)
        hidden = self.mel2hidden(mel)
        f0, cents = self.decode(hidden, thred=thred, use_viterbi=use_viterbi)
        return f0[0], cents[0]  # squeeze batch dim -> (T,)

    def get_pitch(self, waveform: np.ndarray, sample_rate: int, hop_size: int, length: int,
                  interp_uv=False, fmin=50, fmax=1000):
        """Extract and resample F0 to target (sample_rate, hop_size, length)."""
        f0, _ = self.infer_from_audio(waveform, sample_rate=sample_rate)
        f0 = self.postprocess(f0, fmin, fmax)
        uv = f0 == 0
        time_step = hop_size / sample_rate
        f0_res = resample_align_curve(f0, 0.01, time_step, length)
        uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
        if not interp_uv:
            f0_res[uv_res] = 0
        return f0_res, uv_res

    @torch.no_grad()
    def infer_from_audio_batch(self, audios: list, sample_rate=16000, thred=0.03, use_viterbi=False):
        """Batch F0 extraction. Returns (f0s, indexs, cents) — each a list of (T_i,) arrays."""
        tensors = [torch.from_numpy(a).float() for a in audios]
        sizes = [math.ceil(a.shape[0] / self.hop_length) for a in audios]
        audios_t = collate_1d(tensors, 0.0).to(self.device)
        if sample_rate != 16000:
            audios_t = self._get_resampler(sample_rate)(audios_t)
            sizes = [math.ceil(s * (16000 / sample_rate)) for s in sizes]
        mels = self.mel_extractor(audios_t, center=True)
        hiddens = self.mel2hidden(mels)
        f0_batch, cent_batch = self.decode(hiddens, thred=thred, use_viterbi=use_viterbi)
        index_batch = hiddens.argmax(dim=2).cpu().numpy()

        f0s, cents, indexs = [], [], []
        for i, sz in enumerate(sizes):
            f0s.append(f0_batch[i, :sz])
            cents.append(cent_batch[i, :sz])
            indexs.append(index_batch[i, :sz])
        return f0s, indexs, cents

    def get_pitch_batch(self, waveforms: list, sample_rate: int, hop_size: int, lengths: list,
                        interp_uv=False, fmin=50, fmax=1000):
        """Batch F0 extraction resampled to target (sample_rate, hop_size)."""
        f0s_raw, indexs, cents = self.infer_from_audio_batch(waveforms, sample_rate=sample_rate)
        f0s_res, uvs_res = [], []
        time_step = hop_size / sample_rate
        for i, f0 in enumerate(f0s_raw):
            f0 = self.postprocess(f0, fmin, fmax, min_gap=6)
            uv = f0 == 0
            length = lengths[i]
            if time_step != 0.01:
                f0_res = resample_align_curve(f0, 0.01, time_step, length)
                uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
            else:
                f0_res = f0[:length] if len(f0) >= length else np.pad(f0, (0, length - len(f0)))
                uv_res = uv[:length] if len(uv) >= length else np.pad(uv, (0, length - len(uv)))
            if not interp_uv:
                f0_res[uv_res] = 0
            f0s_res.append(f0_res)
            uvs_res.append(uv_res)
        return f0s_res, indexs, cents

    def release_cuda(self):
        self.model = self.model.cpu()
        self.mel_extractor = self.mel_extractor.cpu()
        torch.cuda.empty_cache()

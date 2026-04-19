import sys
import math
import torch
import soundfile as sf
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
    StableAudioAttnProcessor2_0_rotary_double,
    StableAudioAttnProcessor2_0_rotary_free,
)
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file  # Import safetensors
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from config_inference_full_song_ablation import get_config
import argparse
import json
from utils.extract_conditions import compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm, calculate_beats_and_downbeats, create_activations_from_timestamps, compute_rhythm_beatnet

# ── RMVPE / F0 melody encoder ─────────────────────────────────────────
_MUSECTRLLITE_DIR = os.path.dirname(os.path.abspath(__file__))
from rmvpe import RMVPE  # noqa: E402

RMVPE_CKPT     = os.path.join(_MUSECTRLLITE_DIR, "SongEcho/rmvpe_model.pt")
F0_MELODY_CKPT = os.path.join(_MUSECTRLLITE_DIR, "SongEcho/melody_encoder.pt")
SR_RMVPE   = 16000
HOP_RMVPE  = 160      # 10 ms frames
FMIN_RMVPE = 50
FMAX_RMVPE = 900
_F0_MIN_LOG = 3.912023005   # ln(50)
_F0_MAX_LOG = 6.802394763   # ln(900)


def _melody_preprocess(f0: np.ndarray, device) -> torch.Tensor:
    """f0: (T,) Hz  →  tensor (1, T, 2) [normalized_log_f0, uv_flag]"""
    f0_t = torch.from_numpy(f0).float().to(device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    voiced = f0_t > 0
    log_f0 = torch.zeros_like(f0_t)
    log_f0[voiced] = torch.log(f0_t[voiced])
    norm_f0 = (log_f0 - _F0_MIN_LOG) / (_F0_MAX_LOG - _F0_MIN_LOG)
    norm_f0[~voiced] = 0.0
    uv_flag = voiced.float()
    return torch.cat([norm_f0, uv_flag], dim=-1)  # (1, T, 2)

from utils.stable_audio_dataset_utils import Stereo, PhaseFlipper
# from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
import random
from torchaudio import transforms as T
from tqdm import tqdm
import torchaudio
# from BeatNet.BeatNet import BeatNet
import subprocess, re, tempfile, shutil
from sklearn.metrics import f1_score
import bisect
from btc_chords import Chords
import mido

KEY_LINE_FULL = re.compile(r"^[A-G](?:[#♯b♭])?(?:m)?$")
KEY_TOKEN_IN_LINE = re.compile(r"\b([A-G](?:[#♯b♭])?(?:m)?)\b")
import torch

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
def sublist_between(arr, a, b, eps=1e-6):
    """Return arr elements in [a, b) using indices (fast; arr must be sorted)."""
    lo = bisect.bisect_left(arr, a - eps)
    hi = bisect.bisect_left(arr, b - eps)   # half-open: up to but not including b
    return arr[lo:hi]
def load_attn1_qkv_into_pipeline(pipeline, qkv_path, dtype=torch.float32, strict=False):
    """
    Loads attn1.to_{q,k,v} weights back into StableAudio pipeline's transformer.
    """
    # If you used Accelerate and prepared the model, unwrap first:
    core = getattr(pipeline, "transformer")
    sd = load_file(qkv_path, device="cpu")

    # (optional) cast tensors to desired dtype
    for k in list(sd.keys()):
        sd[k] = sd[k].to(dtype)

    # Will fill matching keys; keeps others unchanged
    incompatible = core.load_state_dict(sd, strict=strict)
    print("Unexpected:", incompatible.unexpected_keys)
class Rhythm_extractor(nn.Module):
    def __init__(self):
        super(Rhythm_extractor, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)  
        self.conv1d_2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(64, 176, kernel_size=3, padding=1)  
    def forward(self, x):
        x = self.conv1d_1(x)# shape: (batchsize, 128, 4756)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 256, 2378)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 256, 2378)
        x = F.silu(x)
        return x
class Structure_extractor(nn.Module):
    def __init__(self):
        super(Structure_extractor, self).__init__()
        self.emb = nn.Embedding(num_embeddings=8, embedding_dim=176, padding_idx=0)
    def forward(self, x):
        x = self.emb(x)    
        return x
class F0MelodyEncoder(nn.Module):
    """Encodes (B, T, 2) f0+uv features into (B, 256, T) melody embeddings."""
    def __init__(self, input_dim=2, hidden_dim=256, kernel_size=5):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
        )

    def forward(self, normalized_f0):
        # normalized_f0: (B, T, 2) → transpose → (B, 2, T)
        x = normalized_f0.transpose(1, 2)
        return self.conv_stack(x)  # (B, 256, T)


class MelodyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Four Conv1d layers, each with kernel_size=3, padding=1:
        self.conv1 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 176, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(176, 176, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = F.silu(x)
        x = self.conv3(x)
        x = F.silu(x)
        return x
class Chord_extractor(nn.Module):
    def __init__(self):
        super(Chord_extractor, self).__init__()
        self.conv1d_1 = nn.Conv1d(12, 64, kernel_size=3, padding=1)  
        self.conv1d_2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(64, 176, kernel_size=3, padding=1)  
    def forward(self, x):
        x = self.conv1d_1(x)# shape: (batchsize, 128, 4756)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 256, 2378)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 256, 2378)
        x = F.silu(x)
        return x
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
        # audio.shape is [channels, samples]
        # num_samples = audio.shape[-1]

        # if num_samples < target_samples:
        #     # Pad if it's too short
        #     pad_amount = target_samples - num_samples
        #     # Zero-pad at the end (or randomly if you prefer)
        #     audio = F.pad(audio, (0, pad_amount)) 
        #     print(f"pad {pad_amount}")
        # else:
        audio = audio[:, int(segment_starts*44100):]
        return audio
    except RuntimeError:
        print(f"Failed to decode audio file: {filename}")
        return None

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    generator = torch.Generator().manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    score_dynamics = []
    score_rhythm = []
    score_melody = []
    output_dir = config["output_dir"] + f"text_{config['guidance_scale_text']}_con_{config['guidance_scale_con']}_{'_'.join(config['condition_type'])}_{config['sigma_min']}_{config['sigma_max']}"
    os.makedirs(output_dir, exist_ok=True)
    weight_dtype = torch.float32
    struct_emb_extractor = Structure_extractor().to("cuda").float()
    melody_emb_extractor = MelodyEncoder().to("cuda").float()
    chord_extractor = Chord_extractor().to("cuda").float()
    rhythm_extractor = Rhythm_extractor().to("cuda").float()
    if config["checkpoint_path"]:
        config["self_attention_ckpt"] = os.path.join(config["checkpoint_path"], "attn1_qkv.safetensors")
        config["transformer_ckpt"] = os.path.join(config["checkpoint_path"], "attn_procs.safetensors")
        config["rhythm_emb_ckpt"] = os.path.join(config["checkpoint_path"], "rhythm_cnn.safetensors")
        config["struct_emb_ckpt"] = os.path.join(config["checkpoint_path"], "struct_emb.safetensors")
        config["melody_emb_ckpt"] = os.path.join(config["checkpoint_path"], "melody_emb.safetensors")
        config["chord_cnn_ckpt"] = os.path.join(config["checkpoint_path"], "chord_cnn.safetensors")
    else:
        config["self_attention_ckpt"] = None
        config["transformer_ckpt"] = None
        config["rhythm_emb_ckpt"] = None
        config["struct_emb_ckpt"] = None
        config["melody_emb_ckpt"] = None
        config["chord_cnn_ckpt"] = None
        config["chord_cnn_ckpt"] = None

    if config['chord_cnn_ckpt'] is not None:
        state_dict = load_file(config['chord_cnn_ckpt'])
        # Check keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        chord_extractor.load_state_dict(new_state_dict)
        print("load chord_extractor")
    if config['rhythm_emb_ckpt'] is not None:
        state_dict = load_file(config['rhythm_emb_ckpt'])
        # Check keys
        print(f"Loaded {len(state_dict)} tensors:")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        rhythm_extractor.load_state_dict(new_state_dict)
        print("load rhythm_extractor")
    if config['struct_emb_ckpt'] is not None:
        state_dict = load_file(config['struct_emb_ckpt'])
        # Check keys
        print(f"Loaded {len(state_dict)} tensors:")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        struct_emb_extractor.load_state_dict(new_state_dict)
        print("load struct_emb_extractor")
    if config['melody_emb_ckpt'] is not None:
        state_dict = load_file(config['melody_emb_ckpt'])
        # Check keys
        print(f"Loaded {len(state_dict)} tensors:")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        melody_emb_extractor.load_state_dict(new_state_dict)
        print("load melody_emb_extractor")

    # Load RMVPE and F0MelodyEncoder for vocal melody extraction
    print(f"Loading RMVPE from {RMVPE_CKPT} ...")
    rmvpe_model = RMVPE(RMVPE_CKPT, hop_length=HOP_RMVPE, device="cuda")
    print(f"Loading F0MelodyEncoder from {F0_MELODY_CKPT} ...")
    f0_melody_enc = F0MelodyEncoder().to("cuda").eval()
    _f0_state = torch.load(F0_MELODY_CKPT, map_location="cuda")
    if isinstance(_f0_state, dict) and "model" in _f0_state:
        _f0_state = _f0_state["model"]
    f0_melody_enc.load_state_dict(_f0_state)
    print("F0MelodyEncoder loaded.")

    if config["weight_dtype"] == "fp16":
        weight_dtype = torch.float16
    elif config["weight_dtype"] == "bp16":
        weight_dtype = torch.bfloat16
    if config["apadapter"]:
        from pipeline.stable_audio_multi_cfg_pipe_free import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
        if config['self_attention_ckpt'] is not None:
            load_attn1_qkv_into_pipeline(pipe, config['self_attention_ckpt'], dtype=torch.float32, strict=False)
        pipe.scheduler.config.sigma_max = config["sigma_max"]
        pipe.scheduler.config.sigma_min = config["sigma_min"]
        transformer = pipe.transformer
        attn_procs = {}
        processor_classes = {
            "rotary": StableAudioAttnProcessor2_0_rotary,
            "rotary_double": StableAudioAttnProcessor2_0_rotary_double,
            "rotary_free": StableAudioAttnProcessor2_0_rotary_free,
        }
        # Get the processor classes based on the type
        attn_processor = processor_classes.get(config["attn_processor_type"], None)
        for name in transformer.attn_processors.keys():
            if name.endswith("attn1.processor"):
                attn_procs[name] = StableAudioAttnProcessor2_0()
            else:
                attn_procs[name] = attn_processor(
                    layer_id = name.split(".")[1],
                    hidden_size=768,
                    name=name,
                    cross_attention_dim=768,
                    scale=config['ap_scale'],
                ).to("cuda", dtype=torch.float)
        if config["transformer_ckpt"] is not None:
            if "bin" in config["transformer_ckpt"]:
                state_dict = torch.load(config["transformer_ckpt"])
            elif "safetensors" in config["transformer_ckpt"]:
                state_dict = load_file(config["transformer_ckpt"], device="cuda")
            for name, processor in attn_procs.items():
                if isinstance(processor, attn_processor):
                    weight_name_v = name + ".to_v_ip.weight"
                    weight_name_k = name + ".to_k_ip.weight"
                    conv_out_weight = name + ".conv_out.weight"
                    processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].to(torch.float32))
                    processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].to(torch.float32))
                    processor.conv_out.weight = torch.nn.Parameter(state_dict[conv_out_weight].to(torch.float32))
                    print(f"load {name}")
        transformer.set_attn_processor(attn_procs)
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipe.transformer(*args, **kwargs)
        transformer = _Wrapper(pipe.transformer.attn_processors)
    else:
        from diffusers import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
        pipe.scheduler.config.sigma_max = config["sigma_max"]
        pipe.scheduler.config.sigma_min = config["sigma_min"]

    pipe = pipe.to("cuda")
    negative_text_prompt = config["negative_text_prompt"]
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    modes = ['', 'm']  # or ['maj','min'] if you prefer
    idx2key = [''] + [f'{n}{m}' for n in notes for m in modes]  # len = 25
    key2idx = {k: i for i, k in enumerate(idx2key)}
    structure2id = {
            'intro': 0,
            'outro': 1,
            'break': 2,
            'bridge': 3,
            'inst': 4,
            'solo': 5,
            'verse': 6,
            'chorus': 7,
        }
    print("structure2id", structure2id)
    print("key2idx", key2idx)
    # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
    total_seconds = 2097152/44100
    if config['use_audio_mask']:
        audio_mask_start = int(config["audio_mask_start_seconds"] / total_seconds * 1024) # 1024 is the latent length for 2097152/44100 seconds
        audio_mask_end = int(config["audio_mask_end_seconds"] / total_seconds * 1024)
    elif config['use_musical_attribute_mask']:
        musical_attribute_mask_start = int(config["musical_attribute_mask_start_seconds"] / total_seconds * 1024)
        musical_attribute_mask_end = int(config["musical_attribute_mask_end_seconds"] / total_seconds * 1024)
    with torch.no_grad():
        _SEGMENT_KEYS      = ['intro', 'verse', 'chorus', 'outro']
        _STRUCTURE_EN_KEYS = ['Intro', 'Verse', 'Chorus', 'Outro']
        _dataset_folder = config['dataset_folder']
        with open(config['text_file'], 'r', encoding="utf-8") as f:
            file_text = json.load(f)
        score_chord = []
        score_rhythm = []
        for song in range(len(file_text)):
            json_file = file_text[song]['json_file']
            print(f"file:{json_file}")
            with open(os.path.join(_dataset_folder, json_file), 'r', encoding='utf-8') as _f:
                _structure_en = json.load(_f)['structure_en']
            _song_local_prompts = {k: _structure_en[K] for k, K in zip(_SEGMENT_KEYS, _STRUCTURE_EN_KEYS)}
            id = int(json_file.split('.')[0])
            if config.get('id_filter') and id not in config['id_filter']:
                print(f"Skipping id {id} (not in id_filter)")
                continue
            if os.path.exists(os.path.join(output_dir, f"mixed_{id}.wav")):
                print(f"Skipping id {id} (mixed_{id}.wav already exists)")
                continue
            info_path = f"{config['midi_folder']}/lyrics_{id}"
            config['structure_tag'] = os.path.join(info_path, "sample01_struct_label.json")
            config['structure_start_seconds'] = os.path.join(info_path, "sample01_struct_time.json")
            config['chord_info'] = os.path.join(info_path, "chord/chord_btc_txt/sample01_shifted_chord_gen_filled_empty_bars.txt")
            gt_chord_path = config['chord_info']
            config['melody_midi'] = os.path.join(info_path, "sample01_shifted.mid")
            config['vocal_audio_files'] = os.path.join(info_path, "sample01_shifted.wav")
            transformer.eval()
            melody_midi = config["melody_midi"]
            midi = mido.MidiFile(melody_midi)
            print(f"Duration: {midi.length:.2f} seconds")
            with open(config['structure_tag'], "r", encoding="utf-8") as f:
                config['structure_tag'] = json.load(f)
            with open(config['structure_start_seconds'], "r", encoding="utf-8") as f:
                config['structure_start_seconds'] = json.load(f)
            structure_starts_seconds = config['structure_start_seconds']
            print("structure_starts_seconds", structure_starts_seconds)
            _vocal_duration = librosa.get_duration(path=config['vocal_audio_files'])
            _beat_times_tmp, _ = calculate_beats_and_downbeats(melody_midi)
            _avg_beat_dur = np.mean(np.diff(_beat_times_tmp)) if len(_beat_times_tmp) > 1 else 0.5
            config['structure_ends_seconds'] = config['structure_start_seconds'][1:] + [_vocal_duration]
            print("structure_ends_seconds", config['structure_ends_seconds'])
            config['key_conditions'] = []
            for c in range(len(structure_starts_seconds)):
                config['key_conditions'].append('C')
            key_ids = [key2idx[k] for k in config['key_conditions']]
            structures_ids = [structure2id[s] for s in config['structure_tag']]
            keys_ids_expand = []
            structures_ids_expand = []
            total_duration = 2097152 / 44100
            slice_len = total_duration / 1024
            latent_length = int((structure_starts_seconds[-1] + 2097152 / 44100) / (2097152 / 44100) * 1024)
            for k in range(latent_length):
                t = (k + 0.5) * slice_len  # midpoint of this slice
                # find which segment t falls into
                j = 0
                while j + 1 < len(structure_starts_seconds) and t >= structure_starts_seconds[j + 1]:
                    j += 1
                keys_ids_expand.append(key_ids[j])
                structures_ids_expand.append(structures_ids[j])
            key_ids = torch.tensor(keys_ids_expand)
            structures_ids = torch.tensor(structures_ids_expand)
            print("key_ids", key_ids)
            print("structures_ids", structures_ids)
            config['structure_start_seconds'].append(_vocal_duration)
            # for i, prompt_texts in enumerate(config['text']):
            backing_audio = torch.empty(2, 0)
            segments_list = list(range(len(config['structure_tag'])))
            tensors = {name: torch.empty(0) for name in config['structure_tag']}
            print("segments_list", segments_list)
            for s, segments in enumerate(segments_list):
                if s > 0:
                    _current_struct = config['structure_tag'][s-1]
                    _next_struct = config['structure_tag'][min(s, len(config['structure_tag']) - 1)]
                else:
                    _current_struct = config['structure_tag'][s]
                    _next_struct = config['structure_tag'][min(s + 1, len(config['structure_tag']) - 1)]
                prompt_1_text = _song_local_prompts.get(_current_struct, _song_local_prompts[_SEGMENT_KEYS[-1]])
                prompt_2_text = _song_local_prompts.get(_next_struct, _song_local_prompts[_SEGMENT_KEYS[-1]])
                print(f"Generating segment {segments + 1}/{len(config['structure_tag'])} for prompt {id}/{len(file_text)}")
                print("_song_local_prompts", _song_local_prompts)
                print("prompt_1", prompt_1_text)
                print("prompt_2", prompt_2_text)
                print("structure_tag", config['structure_tag'][segments])
                if config["apadapter"]:
                    gt_vocal_audio_file = config["vocal_audio_files"]
                    if config["no_text"] is True:
                        prompt_1_text = ""
                        prompt_2_text = ""
                    description_path = os.path.join(output_dir, "description.txt")
                    _window_start_s = config['structure_start_seconds'][segments]
                    _ref_end = None
                    beat_times_all, downbeat_times_all = calculate_beats_and_downbeats(melody_midi)
                    if "audio" in config["condition_type"] and s != 0:
                        print("config['structure_start_seconds']", config['structure_start_seconds'])
                        _ref_end   = int(config['structure_start_seconds'][segments]*44100)
                        _ref_start = max(0, int((config['structure_ends_seconds'][segments] - 2097152 / 44100) * 44100))
                        _window_start_s = _ref_start / 44100
                        audio = backing_audio[:, _ref_start:_ref_end].unsqueeze(0).to(weight_dtype).cuda()
                        print(f"{_ref_start/44100} ~ {_ref_end/44100} seconds will be reference audio")
                        audio_condition = torch.zeros((1, 64, 1024), device="cuda")
                        audio_condition_ref = pipe.vae.encode(audio).latent_dist.sample()
                        print("audio_condition_ref", audio_condition_ref.shape)
                        audio_condition[:,:,:audio_condition_ref.shape[2]] = audio_condition_ref
                        extracted_audio_condition = audio_condition
                        masked_extracted_audio_condition = torch.zeros_like(extracted_audio_condition)
                    else:
                        extracted_audio_condition = torch.zeros((1, 64, 1024), device="cuda")
                        masked_extracted_audio_condition = extracted_audio_condition
                    if "structure" in config['condition_type']:
                        structures_ids_start = int(_window_start_s / (2097152/44100) * 1024)
                        # print("structures_ids_start", structures_ids_start)
                        # print("structure_start_seconds", _window_start_s)
                        structures_ids_segment = structures_ids[structures_ids_start:structures_ids_start + 1024]
                        extracted_struct_condition = struct_emb_extractor(structures_ids_segment.cuda().unsqueeze(0)).transpose(1,2)
                        # print("structure_condition", extracted_struct_condition.shape)
                        masked_extracted_struct_condition = torch.zeros_like(extracted_struct_condition)
                    else: 
                        extracted_struct_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_struct_condition = extracted_struct_condition
                    # For single conition, we can utilize the full cross-attention dimension 768, instead of 768/4 in MuseControlLite_inference_on_the_fly_all.py
                    if "melody" in config["condition_type"]:
                        _seg_start = _window_start_s
                        _wav, _ = librosa.load(
                            gt_vocal_audio_file, sr=SR_RMVPE, mono=True,
                            offset=_seg_start, duration=2097152 / 44100,
                        )
                        _length = math.ceil(len(_wav) / HOP_RMVPE)
                        _f0, _ = rmvpe_model.get_pitch(
                            _wav, SR_RMVPE, HOP_RMVPE, _length,
                            fmin=FMIN_RMVPE, fmax=FMAX_RMVPE,
                        )
                        _f0_input = _melody_preprocess(_f0, "cuda")  # (1, T, 2)
                        with torch.no_grad():
                            melody_condition = f0_melody_enc(_f0_input)  # (1, 256, T)
                        extracted_melody_condition = melody_emb_extractor(melody_condition)
                        # print("melody_condition", extracted_melody_condition.shape)
                        masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
                        extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                        masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                    else:
                        extracted_melody_condition = torch.zeros((1, 176, 1024), device="cuda")
                        masked_extracted_melody_condition = extracted_melody_condition
                    if "chord" in config["condition_type"]:
                        chord_condition, end_time = extract_chords_lab(config['chord_info'], segment_starts = _window_start_s)
                        extracted_chord_condition = chord_extractor(chord_condition)
                        # print("chord_condition", extracted_chord_condition.shape)
                        # extracted_melody_condition = condition_extractors["melody"](melody_condition.to(torch.float32))
                        masked_extracted_chord_condition = torch.zeros_like(extracted_chord_condition)
                        # extracted_chord_condition = F.interpolate(extracted_chord_condition, size=1024, mode='linear', align_corners=False)
                        # masked_extracted_chord_condition = F.interpolate(masked_extracted_chord_condition, size=1024, mode='linear', align_corners=False)
                    else:
                        chord_condition, end_time = extract_chords_lab(config['chord_info'], segment_starts = _window_start_s)
                        extracted_chord_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_chord_condition = extracted_chord_condition
                    if "rhythm" in config["condition_type"]:
                        beat_times_all += [beat_times_all[-1] + (beat_times_all[-1] - beat_times_all[-2]) * k for k in range(1, 200)] 
                        downbeat_times_all += [downbeat_times_all[-1] + (downbeat_times_all[-1] - downbeat_times_all[-2]) * k for k in range(1, 40)] 
                        beat_times = sublist_between(beat_times_all, _window_start_s, _window_start_s + 2097152/44100)
                        downbeat_times = sublist_between(downbeat_times_all, _window_start_s, _window_start_s + 2097152/44100)
                        # print("beat_times: ", beat_times)
                        # print("downbeat_times: ", downbeat_times)
                        beat_times = [x - _window_start_s for x in beat_times]
                        downbeat_times = [x - _window_start_s for x in downbeat_times]
                        # print("beat_times: ", beat_times)
                        rhythm_condition = create_activations_from_timestamps(beat_times, downbeat_times)
                        # print("rhythm_condition", rhythm_condition.shape)
                        extracted_rhythm_condition = rhythm_extractor(torch.from_numpy(rhythm_condition).cuda().unsqueeze(0).float()) 
                        # print("rhythm_condition", extracted_rhythm_condition.shape)
                        masked_extracted_rhythm_condition = torch.zeros_like(extracted_rhythm_condition)
                        extracted_rhythm_condition = F.interpolate(extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
                        masked_extracted_rhythm_condition = F.interpolate(masked_extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
                    else: 
                        extracted_rhythm_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_rhythm_condition = extracted_rhythm_condition
                    
                    # Use multiple cfg
                    # print(extracted_rhythm_condition.shape, extracted_melody_condition.shape, extracted_struct_condition.shape, extracted_audio_condition.shape, extracted_chord_condition.shape)
                    extracted_condition = torch.concat((extracted_rhythm_condition, extracted_melody_condition, extracted_struct_condition, extracted_audio_condition, extracted_chord_condition), dim=1)
                    masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_melody_condition, masked_extracted_struct_condition, masked_extracted_audio_condition, masked_extracted_chord_condition), dim=1)
                    extracted_condition = torch.concat((masked_extracted_condition, masked_extracted_condition, extracted_condition), dim=0)
                    extracted_condition = extracted_condition.transpose(1, 2)
                    audio_mid_s = (_ref_end - _ref_start) / 44100 if _ref_end is not None else structure_starts_seconds[s+1]
                    print("audio_mid_s, ", audio_mid_s)
                    waveform = pipe(
                        extracted_condition = extracted_condition, 
                        prompt_1=prompt_1_text,
                        prompt_2=prompt_2_text,
                        negative_prompt=negative_text_prompt,
                        num_inference_steps=config["denoise_step"],
                        guidance_scale_text=config["guidance_scale_text"],
                        guidance_scale_con=config["guidance_scale_con"],
                        num_waveforms_per_prompt=1,
                        audio_end_in_s=2097152 / 44100,
                        audio_mid_s = audio_mid_s,
                        generator=generator,
                    ).audios 
                    # print(f"{i}")      
                    
                    # output = waveform[0]
                    tensors[config['structure_tag'][segments]] = waveform[0]
                    # print("output", output.shape)
                    # print("backing_audio", backing_audio.shape)
                    print(f"Generate {config['structure_tag'][segments]} segment")
                    if s == 0:
                        backing_audio = torch.cat((backing_audio, tensors[config['structure_tag'][segments]][:, :int(44100 * (config['structure_start_seconds'][segments + 1] - config['structure_start_seconds'][segments]))].cpu()), dim=1)
                        print(f"generate {config['structure_start_seconds'][segments]} ~ {config['structure_start_seconds'][segments + 1]} seconds")
                    else:
                        backing_audio = torch.cat((backing_audio, tensors[config['structure_tag'][segments]][:, int(44100*audio_mid_s):int(44100*audio_mid_s) + int(44100 * (config['structure_start_seconds'][segments + 1] - config['structure_start_seconds'][segments]))].cpu()), dim=1)
                        print(f"generate {config['structure_start_seconds'][segments]} ~ {config['structure_start_seconds'][segments + 1]} seconds")
                    # else:
                    #     backing_audio = torch.cat((backing_audio, tensors[config['structure_tag'][segments]][:, :int(config['structure_start_seconds'] - audio_mid_s)].cpu()), dim=1)
                    #     print(f"generate {_ref_start/44100} ~ {_ref_start/44100 + 2087152/44100} seconds")
                    # save_segments = os.path.join(output_dir, f"{config['structure_start_seconds'][segments]}_{config['structure_start_seconds'][segments + 1]}.wav")
                    # sf.write(save_segments, tensors[config['structure_tag'][segments]][:, :int(44100 * (config['structure_start_seconds'][segments + 1] - config['structure_start_seconds'][segments]))].T.float().cpu().numpy(), pipe.vae.sampling_rate)
                    
                    print(f"backing audio length {backing_audio.shape[1]/44100} seconds")
                    print("===============================")
                    
                else:
                    audio = pipe(
                        prompt=prompt_texts,
                        negative_prompt=negative_text_prompt,
                        num_inference_steps=config["denoise_step"],
                        guidance_scale=config["guidance_scale_text"],
                        num_waveforms_per_prompt=1,
                        audio_end_in_s=2097152/44100,
                        generator=generator,
                    ).audios
                    output = audio[0].T.float().cpu().numpy()
                    file_path = os.path.join(output_dir, f"{prompt_texts}.wav")
                    sf.write(file_path, output, pipe.vae.sampling_rate)    
            
            waveform_vocal = load_audio_file(gt_vocal_audio_file, segment_starts= config['structure_start_seconds'][0])
            # waveform_vocal = torch.cat([waveform_vocal_mono, waveform_vocal_mono], dim=0)
            print("waveform_vocal", waveform_vocal.shape[1]/44100)
            print("backing_audio", backing_audio.shape[1]/44100)

            _min_len = min(waveform_vocal.shape[-1], backing_audio.shape[-1])
            waveform_vocal, backing_audio = pad_to_match(waveform_vocal, backing_audio, len=_min_len)
            waveform_vocal = waveform_vocal.to(torch.float32)
            backing_audio = backing_audio.to(torch.float32)

            mix = mix_audio(waveform_vocal, backing_audio, target_dbfs=-18.0, out_peak_dbfs=-1.0)
            mixed_file_path = os.path.join(output_dir, f"mixed_{id}.wav")
            sf.write(mixed_file_path, mix.T.float().cpu().numpy(), pipe.vae.sampling_rate)

        #     estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
        #     generated_timestamps = estimator.process(mixed_file_path)
        #     audio_duration = backing_audio.shape[1] / 44100
        #     beat_times_eval = np.array([t for t in beat_times_all if t <= audio_duration])
        #     gen_timestamps_eval = generated_timestamps[generated_timestamps[:, 0] <= audio_duration]
        #     print("beat_times_eval", beat_times_eval)
        #     print("gen_timestamps_eval", gen_timestamps_eval)
        #     precision, recall, f1 = evaluate_f1_rhythm(beat_times_eval, gen_timestamps_eval)
        #     score_rhythm.append(f1)
        #     print(f"rhythm {id}: {f1:.4f}")
        #     if "chord" in config["condition_type"]:
        #         # Use a temp dir with only the current song so BTC doesn't reprocess all songs
        #         with tempfile.TemporaryDirectory() as btc_tmp_dir:
        #             shutil.copy(mixed_file_path, os.path.join(btc_tmp_dir, f"mixed_{id}.wav"))
        #             command = [
        #                 "python",
        #                 "test.py",
        #                 "--audio_dir", btc_tmp_dir,
        #                 "--save_dir", btc_tmp_dir,
        #                 "--voca", "True"
        #             ]
        #             print("val_audio_dir", btc_tmp_dir)
        #             working_directory = "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite/BTC-ISMIR19"
        #             try:
        #                 result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=working_directory)
        #                 print("Command executed successfully.")
        #                 print("Output:\n", result.stdout)
        #                 if result.stderr:
        #                     print("Stderr:\n", result.stderr)
        #             except subprocess.CalledProcessError as e:
        #                 print("Error occurred while running the command:")
        #                 print(e.stderr)

        #             # Find generated chord .lab file produced by BTC
        #             gen_lab_files = [f for f in os.listdir(btc_tmp_dir) if f.endswith(f'mixed_{id}.lab')]
        #             print("gen_lab_files", gen_lab_files)
        #             if len(gen_lab_files) != 1:
        #                 print(f"Warning: Expected 1 .lab for mixed_{id}, got {gen_lab_files}. Skipping chord scoring for this song.")
        #                 continue
        #             gen_chord_path = os.path.join(btc_tmp_dir, gen_lab_files[0])

        #             def _chord_file_to_chroma(chord_path, total_samples):
        #                 _CHORDS = Chords()
        #                 chroma = np.zeros((12, total_samples))
        #                 with open(chord_path, 'r') as f:
        #                     chord_infos = f.read().splitlines()
        #                 for info in chord_infos:
        #                     parts = info.split(' ')
        #                     if len(parts) < 3:
        #                         continue
        #                     s, t, chord = parts[0], parts[1], parts[2]
        #                     s_idx = int(float(s) * 44100)
        #                     t_idx = min(int(float(t) * 44100), total_samples)
        #                     if s_idx >= total_samples:
        #                         continue
        #                     mhot = _CHORDS.chord(chord)
        #                     final_vec = np.roll(mhot[2], mhot[0])[..., None]
        #                     chroma[:, s_idx:t_idx] = final_vec
        #                 chroma = F.interpolate(
        #                     torch.from_numpy(chroma).unsqueeze(0), size=4756, mode='nearest'
        #                 )
        #                 return chroma.flatten().cpu().numpy()

        #             total_samples = sf.info(mixed_file_path).frames
        #             c_gen = _chord_file_to_chroma(gen_chord_path, total_samples)
        #             c_gt  = _chord_file_to_chroma(gt_chord_path, total_samples)

        #             c_gen_bin = (c_gen > 0).astype(int)
        #             c_gt_bin  = (c_gt  > 0).astype(int)

        #             chord_f1 = f1_score(c_gt_bin, c_gen_bin, average='binary')
        #             score_chord.append(chord_f1)
        #             print(f"Chord F1 score for id {id}: {chord_f1:.4f}")
        #         # btc_tmp_dir and its contents are automatically deleted here

        #     # # Save per-song scores to result.json
        #     # song_result = {"id": id}
        #     # if "chord" in config["condition_type"] and score_chord:
        #     #     song_result["chord_f1"] = score_chord[-1]
        #     # if "rhythm" in config["condition_type"] and score_rhythm:
        #     #     song_result["rhythm_f1"] = score_rhythm[-1]
        #     # result_path = os.path.join(output_dir, f"result_chord_{id}.json")
        #     # with open(result_path, 'w') as f:
        #     #     json.dump(song_result, f, indent=4)
        # print(float(np.mean(score_chord)) if score_chord else None)
        # print(float(np.mean(score_rhythm)) if score_rhythm else None)
        # avg_scores = {
        #     "avg_chord_f1": float(np.mean(score_chord)) if score_chord else None,
        #     "avg_rhythm_f1": float(np.mean(score_rhythm)) if score_rhythm else None,
        # }
        # avg_scores_path = os.path.join(output_dir, "avg_scores.json")
        # with open(avg_scores_path, 'w') as f:
        #     json.dump(avg_scores, f, indent=4)
        # print(f"Saved average scores to {avg_scores_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AP-adapter Inference Script")
    config = get_config()  # Pass the parsed arguments to get_config
    main(config)

import argparse
import itertools
import math
import os
import random
import shutil
import warnings
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset, random_split, DataLoader
import torchaudio
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.training_utils import free_memory

from diffusers.models.embeddings import get_1d_rotary_pos_embed
from scipy.io.wavfile import write
from scipy.signal import savgol_filter
from diffusers.loaders import AttnProcsLayers
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import matplotlib.pyplot as plt
from safetensors.torch import load_file  # Import safetensors
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.stable_audio_dataset_utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from torchaudio import transforms as T
import soundfile as sf
from pipeline.stable_audio_multi_cfg_pipe_free import StableAudioPipeline
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
    StableAudioAttnProcessor2_0_rotary_double,
    StableAudioAttnProcessor2_0_echo,
    StableAudioAttnProcessor2_0_rotary_free,
)
from utils.extract_conditions import compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm, create_activations_from_timestamps
from sklearn.metrics import f1_score
from config_training import get_config
from torch.cuda.amp import autocast
from madmom.features import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
### same way as stable audio loads audio file
import gc
torchaudio.set_audio_backend("sox_io")
import datetime
import torch.distributed as dist
import time
from BeatNet.BeatNet import BeatNet
import torch
import re
from safetensors.torch import load_file
import torch.nn as nn
import pyloudnorm as pyln
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import os, shlex, subprocess
from safetensors.torch import save_file  
import subprocess, re
from sklearn.metrics import f1_score
# Accept ASCII (#/b) and Unicode (♯/♭) accidentals
# KEY_LINE_FULL = re.compile(r"^[A-G](?:[#♯b♭])?(?:m)?$")
# KEY_TOKEN_IN_LINE = re.compile(r"\b([A-G](?:[#♯b♭])?(?:m)?)\b")
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
    print("Loaded QKV. Missing:", incompatible.missing_keys, "Unexpected:", incompatible.unexpected_keys)
def save_attn_processors(pipeline, out_dir, filename="attn_procs.safetensors"):
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for name, proc in pipeline.transformer.attn_processors.items():
        # grab all registered parameters on the processor
        for p_name, p in proc.named_parameters(recurse=True):
            state[f"{name}.{p_name}"] = p.detach().cpu()
    # nothing to save if processors are stateless
    if len(state) == 0:
        print("No trainable parameters found in attn processors (they might be stateless).")
    save_file(state, os.path.join(out_dir, filename))
    print(f"Saved {len(state)} tensors to {os.path.join(out_dir, filename)}")



def mix_and_save(
    audio_full_path,               # list/tuple with one path
    waveform_vocal,                # (C, T) tensor, any device
    audio,                         # batch tensor; we use audio[0] -> (C, T)
    pipeline,
    gen_file, trained_mixed_file, gt_mixed_file, original_file,
    gen_file_no_audio_condition,
    audio_condition_ends_s,
    eps=1e-8,
):
    sr = pipeline.vae.sampling_rate

    # context manager if autograd might already be off
    ctx = torch.no_grad if torch.is_grad_enabled() else (lambda: nullcontext())
    with ctx():

        # keep compute on the same device as vocal/out
        device = waveform_vocal.device
        min_length = min(waveform_vocal.shape[1], audio[0].shape[1])
        vocal = waveform_vocal.to(dtype=torch.float32, device=device)[:,:min_length]
        out   = audio[0].to(dtype=torch.float32, device=device)[:,:min_length]

        # load backing on CPU, then move once
        backing_cpu, _sr_file = torchaudio.load(audio_full_path[0])  # float32, (C,T)
        if _sr_file != sr:
            backing_cpu = torchaudio.functional.resample(backing_cpu, _sr_file, sr)
        if backing_cpu.shape[1] < 2097152:
            backing_cpu = torch.nn.functional.pad(backing_cpu, (0, 2097152 - backing_cpu.shape[1]))
        backing = backing_cpu.to(dtype=torch.float32, device=device, non_blocking=True)[:,:2097152]

        # precompute linear gains once
        g8 = float(10 ** (-8.0 / 20.0))
        g1 = float(10 ** (-1.0 / 20.0))

        # per-signal peak scales (one reduction each)
        v_scale = g8 / vocal.abs().amax().clamp_min(eps)
        o_scale = g8 / out.abs().amax().clamp_min(eps)
        b_scale = g8 / backing.abs().amax().clamp_min(eps)

        # normalize then mix
        trained_mix = 0.5 * (vocal * v_scale + out * o_scale)
        gt_mix      = 0.5 * (vocal * v_scale + backing * b_scale)

        # final peak to -1 dB (one reduction per mix)
        trained_mix = trained_mix * (g1 / trained_mix.abs().amax().clamp_min(eps))
        gt_mix      = gt_mix      * (g1 / gt_mix.abs().amax().clamp_min(eps))

        # move to CPU once for saving
        out_cpu      = out.detach().to("cpu", non_blocking=True)
        trained_cpu  = trained_mix.detach().to("cpu", non_blocking=True)
        gt_cpu       = gt_mix.detach().to("cpu", non_blocking=True)
        def _save(p, x): torchaudio.save(p, x, sr, encoding="PCM_S", bits_per_sample=16)
        with ThreadPoolExecutor(max_workers=3) as ex:
            ex.submit(_save, gen_file, out_cpu)
            ex.submit(_save, trained_mixed_file, trained_cpu)
            ex.submit(_save, gt_mixed_file, gt_cpu)
            # ex.submit(_save, gen_file_no_audio_condition, out_cpu[:, int(audio_condition_ends_s*44100):])

        # copy original as-is
        shutil.copy(audio_full_path[0], original_file)

def pad_time_to_4097(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != 8:
        raise ValueError(f"Expected shape (8, x), got {a.shape}")

    t = a.shape[1]  # time length
    if t == 4097:
        return a
    if t < 4097:
        return np.pad(a, ((0, 0), (0, 4097 - t)), mode="constant", constant_values=0)
    # If you want to *forbid* truncation, raise instead of slicing:
    return a[:, :4097]
_CHORD_FULL_LEN = 2097152
_CHORD_OUT_FRAMES = 1024

def load_chord_from_lab(lab_path):
    """Read a .lab chord annotation file and return a [1, 12, 1024] float32 tensor.
    Follows extract_chord_condition_multi_process.py exactly.
    Chords() is instantiated inside to avoid multiprocessing pickling issues.
    """
    from btc_chords import Chords
    CHORDS = Chords()
    chroma = np.zeros((12, _CHORD_FULL_LEN), dtype=np.float32)
    if not os.path.exists(lab_path):
        # Filenames may differ in trailing zeros (e.g. 227.70 vs 227.7, 177.00 vs 177.0)
        # Use float parsing to normalize: str(float('177.00')) == '177.0'
        normalized = os.path.join(
            os.path.dirname(lab_path),
            re.sub(r'\d+\.\d+', lambda m: str(float(m.group(0))), os.path.basename(lab_path))
        )
        if os.path.exists(normalized):
            lab_path = normalized
        else:
            ten = torch.from_numpy(chroma).unsqueeze(0)
            print("missing chord file", lab_path)
            return F.interpolate(ten, size=_CHORD_OUT_FRAMES, mode='linear', align_corners=False)
    with open(lab_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        s, t, chord = parts[0], parts[1], parts[2]
        mhot = CHORDS.chord(chord)
        vec = np.roll(mhot[2], int(mhot[0])).astype(np.float32)  # (12,)
        s_idx = max(0, int(float(s) * 44100))
        t_idx = min(_CHORD_FULL_LEN, int(float(t) * 44100))
        if t_idx > s_idx:
            chroma[:, s_idx:t_idx] = vec[:, None]
    ten = torch.from_numpy(chroma).unsqueeze(0)  # (1, 12, FULL_LEN)
    return F.interpolate(ten, size=_CHORD_OUT_FRAMES, mode='linear', align_corners=False)  # (1, 12, 1024)

class AudioInversionDataset(Dataset):
    def __init__(
        self,
        config,
        device,
        force_channels="stereo"
    ):
        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )
        self.root_paths = []
        self.force_channels = force_channels
        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        self.config = config
        self.device = device
        self.meta_path = config['meta_data_path']
        with open(self.meta_path) as f:
            self.meta = json.load(f)

        self.structure2id = {
            'intro': 0,
            'outro': 1,
            'break': 2,
            'bridge': 3,
            'inst': 4,
            'solo': 5,
            'verse': 6,
            'chorus': 7,
        }
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):   
        meta_entry = self.meta[i]
        structures = meta_entry.get('structures')
        structure_starts_seconds = meta_entry.get('structure_starts_seconds')
        captions = meta_entry.get('captions')
        beats = meta_entry.get('beats')
        downbeats = meta_entry.get('downbeats')
        vocal_melody_path = meta_entry.get('vocal_f0_melody')
        latent_path = meta_entry.get('latent_path')
        chord_path = meta_entry.get('chord_path')
        # # Load numpy arrays concurrently
        if vocal_melody_path and os.path.exists(vocal_melody_path):
            vocal_melody_curve = torch.load(vocal_melody_path)
            # print("vocal_melody_curve", vocal_melody_curve.shape)
        else:
            print("missing vocal melody file", vocal_melody_path)
            vocal_melody_curve = torch.zeros((256, 4756))
        # vocal_melody_curve = pad_time_to_4097(vocal_melody_curve)
        rhythm_curve = create_activations_from_timestamps(beats, downbeats)
        vocal_path = meta_entry.get('vocal_path')
        # key_info = meta_entry.get('key_info')
        instrumental_path = meta_entry.get('path')
        # key_ids = self.map_notes(key_info)
        structures_ids = [self.structure2id[s] for s in structures]
        audio = torch.load(latent_path, map_location=torch.device('cpu'))
        captions = [c if isinstance(c, str) else '' for c in captions]
        if chord_path.endswith('.lab'):
            chord = load_chord_from_lab(chord_path)
        else:
            chord = torch.load(chord_path, map_location=torch.device('cpu'))
        # keys_ids_expand = []
        structures_ids_expand = []
        total_duration = 2097152 / 44100
        slice_len = total_duration / 1024
        for i in range(1024):
            t = (i + 0.5) * slice_len  # midpoint of this slice
            # find which segment t falls into
            j = 0
            while j + 1 < len(structure_starts_seconds) and t >= structure_starts_seconds[j + 1]:
                j += 1
            # keys_ids_expand.append(key_ids[j])
            structures_ids_expand.append(structures_ids[j])
        assert len(structures_ids) == len(structure_starts_seconds)
        if len(structures) == 1:
            audio_mid_s = 2097152/44100
        else:
            audio_mid_s = structure_starts_seconds[1]
        for j in range(min(len(structures), len(captions))):
            if random.random() < 0.9:
                structure_str = structures[j] if isinstance(structures[j], str) else ', '.join(structures[j])
                captions[j] = structure_str + ',' + captions[j]
        example = {
            "audio_mid_s": audio_mid_s,
            "structures_ids": structures_ids_expand,
            "audio_condition_ends": structure_starts_seconds,
            "text": captions,
            "vocal_path": vocal_path,
            "instrumental_path": instrumental_path,
            "chord": chord,
            "audio": audio,
            "vocal_melody_curve": vocal_melody_curve,
            "rhythm_curve": rhythm_curve,
            "rhythm_timestamp": beats,
            "seconds_start": 0,
            "seconds_end": 2097152 / 44100,
        }
        return example
    
class CollateFunction:
    def __call__(self, examples):
        audio = [example["audio"] for example in examples]
        audio = torch.stack(audio)
        chord = [example["chord"] for example in examples]
        chord_condition = torch.stack(chord).float().squeeze(0)
        
        prompt_texts = [example["text"] for example in examples]
        instrumental_path = [example["instrumental_path"] for example in examples]
        vocal_path = [example["vocal_path"] for example in examples]

        seconds_start = [example["seconds_start"] for example in examples]
        seconds_end = [example["seconds_end"] for example in examples]
        audio_condition_ends = [example["audio_condition_ends"] for example in examples]
        
        # key_ids = [example["key_ids"] for example in examples]
        # key_ids = [torch.tensor(cond) for cond in key_ids]
        # key_ids = torch.stack(key_ids)

        structures_ids = [example["structures_ids"] for example in examples]
        structures_ids = [torch.tensor(cond) for cond in structures_ids]
        structures_ids = torch.stack(structures_ids)


        melody_condition = [example["vocal_melody_curve"] for example in examples]
        melody_condition = [torch.nn.functional.pad(t, (0, 4756 - t.shape[-1])) if t.shape[-1] < 4756 else t for t in melody_condition]
        melody_condition = torch.stack(melody_condition)

        rhythm_condition = [example["rhythm_curve"] for example in examples]
        rhythm_condition = [torch.tensor(cond) for cond in rhythm_condition]
        rhythm_condition = torch.stack(rhythm_condition)

        rhythm_timestamp = [example["rhythm_timestamp"] for example in examples]
        rhythm_timestamp = [torch.tensor(cond) for cond in rhythm_timestamp]
        
        audio_mid_s = [example["audio_mid_s"] for example in examples]
        
        batch = {
            "audio_mid_s": audio_mid_s,
            "structures_ids": structures_ids,
            "audio_condition_ends": audio_condition_ends,
            "instrumental_path": instrumental_path,
            "vocal_path": vocal_path,
            "audio": audio,
            "melody_condition": melody_condition,
            "chord_condition": chord_condition,
            "rhythm_condition": rhythm_condition,
            "rhythm_timestamp":rhythm_timestamp,
            "prompt_texts": prompt_texts,
            "seconds_start": seconds_start,
            "seconds_end": seconds_end,
        }
        return batch

class structure_extractor(nn.Module):
    def __init__(self):
        super(structure_extractor, self).__init__()
        self.emb = nn.Embedding(num_embeddings=8, embedding_dim=176, padding_idx=0)
    def forward(self, x):
        x = self.emb(x)    
        return x
class MelodyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Four Conv1d layers, each with kernel_size=3, padding=1:
        self.conv1 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 176, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(176, 176, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)# shape: (batchsize, 128, 4756)
        x = F.silu(x)
        x = self.conv2(x) # shape: (batchsize, 256, 2378)
        x = F.silu(x)
        x = self.conv3(x) # shape: (batchsize, 256, 2378)
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
def log_validation(accelerator, val_dataloader, rhythm_extractor, struct_emb_extractor, chord_extractor, melody_emb_extractor, condition_type, pipeline, config, weight_dtype, global_step):
    import wandb
    val_audio_dir = os.path.join(config["output_dir"], "val_audio_{}".format(global_step))
    os.makedirs(val_audio_dir, exist_ok=True)
    wandb_logs = {}
    score_rhythm = []
    # key_acc = []
    score_chord = []
    # self.key2idx {'': 0, 'C major': 1, 'C minor': 2, 'C# major': 3, 'C# minor': 4, 'D major': 5, 'D minor': 6, 'D# major': 7, 'D# minor': 8, 'E major': 9, 'E minor': 10, 'F major': 11, 'F minor': 12, 'F# major'
    #: 13, 'F# minor': 14, 'G major': 15, 'G minor': 16, 'G# major': 17, 'G# minor': 18, 'A major': 19, 'A minor': 20, 'A# major': 21, 'A# minor': 22, 'B major': 23, 'B minor': 24}
    for step, batch in enumerate(val_dataloader):
        if step > config["test_num"] - 1:
            break
        pipeline.transformer.eval()  # Set the transformer to evaluation mode
        # key_emb_extractor.eval()
        struct_emb_extractor.eval()
        melody_emb_extractor.eval()
        chord_extractor.eval()
        rhythm_extractor.eval()
        audio_mid_s = batch["audio_mid_s"]
        prompt_texts = batch["prompt_texts"]
        melody_condition = batch["melody_condition"].to(accelerator.device)
        rhythm_condition = batch["rhythm_condition"].to(accelerator.device)
        chord_condition = batch["chord_condition"].to(accelerator.device)
        rhythm_timestamp = batch["rhythm_timestamp"]
        extracted_audio_condition = batch['audio'].to(accelerator.device)

        audio_full_path = batch["instrumental_path"]
        vocal_path = batch["vocal_path"]
        waveform_vocal, sr_vocal = torchaudio.load(vocal_path[0])
        # print("waveform_vocal", waveform_vocal.shape)
        # print("sr_vocal", sr_vocal)
        if sr_vocal != pipeline.vae.sampling_rate:
            waveform_vocal = torchaudio.functional.resample(waveform_vocal, sr_vocal, pipeline.vae.sampling_rate)
        if waveform_vocal.shape[1] < 2097152:
            waveform_vocal = torch.nn.functional.pad(waveform_vocal, (0, 2097152 - waveform_vocal.shape[1]))
        with torch.no_grad():
            extracted_chord_condition = chord_extractor(chord_condition)
        masked_extracted_chord_condition = torch.full_like(extracted_chord_condition.to(torch.float32), fill_value=0)
        ### conditioned
        with torch.no_grad():
            extracted_melody_condition = melody_emb_extractor(melody_condition)
        masked_extracted_melody_condition = torch.full_like(extracted_melody_condition.to(torch.float32), fill_value=0)
        with torch.no_grad():
            extracted_rhythm_condition = rhythm_extractor(rhythm_condition.float())

        masked_extracted_rhythm_condition= torch.full_like(extracted_rhythm_condition.to(torch.float32), fill_value=0)

        struct_condition = batch["structures_ids"].to(accelerator.device)
        # print("struct_condition", struct_condition.shape)
        with torch.no_grad():
            extracted_struct_condition = struct_emb_extractor(struct_condition).transpose(2,1)
        # print("extracted_struct_condition", extracted_struct_condition.shape)
        masked_extracted_struct_condition= torch.full_like(extracted_struct_condition, fill_value=0)

        masked_extracted_audio_condition= torch.full_like(extracted_audio_condition, fill_value=0)

        extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
        masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
        extracted_rhythm_condition = F.interpolate(extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
        masked_extracted_rhythm_condition = F.interpolate(masked_extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
        # print("extracted_audio_condition", extracted_audio_condition.shape)
        # print("extracted_rhythm_condition", extracted_rhythm_condition.shape)
        # print("extracted_melody_condition", extracted_melody_condition.shape)
        # print("extracted_struct_condition", extracted_struct_condition.shape)
        # print("extracted_key_condition", extracted_key_condition.shape)
        total_seconds = 2097152/44100

        audio_condition_ends_in_s = batch['audio_condition_ends'][0]
        audio_condition_ends = [int(s / total_seconds * 1024) for s in audio_condition_ends_in_s]
         
        if len(audio_condition_ends) == 1:
            extracted_audio_condition[0] = torch.zeros_like(extracted_audio_condition[0])
            audio_where = f"None {audio_condition_ends[0]}"
        elif struct_condition[0][0] == 0 and random.random() > 0.5:
            extracted_audio_condition[0] = torch.zeros_like(extracted_audio_condition[0])
            audio_where = f"improvise intro {audio_condition_ends[1]}"
        else:
            extracted_audio_condition[0][:,audio_condition_ends[1]:] = 0
            audio_where = f"first segment {audio_condition_ends[1]}"

        extracted_condition = torch.concat((extracted_rhythm_condition, extracted_melody_condition, extracted_struct_condition, extracted_audio_condition, extracted_chord_condition), dim=1)
        masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_melody_condition, masked_extracted_struct_condition, masked_extracted_audio_condition, masked_extracted_chord_condition), dim=1)
        extracted_condition = torch.concat((masked_extracted_condition, masked_extracted_condition, extracted_condition), dim=0)
        extracted_condition = extracted_condition.transpose(1, 2)
        generator = torch.Generator(device=extracted_condition.device).manual_seed(42)
        # prompt_texts[0] = prompt_texts[0][1] if len(prompt_texts[0]) > 1 else prompt_texts[0][0]
        with torch.no_grad():
            audio = pipeline(
                extracted_condition = extracted_condition, 
                audio_mid_s=audio_mid_s,
                guidance_scale_con = config['guidance_scale_con'],
                guidance_scale_text=config["guidance_scale_text"],
                prompt_1 = prompt_texts[0][0],
                prompt_2 = prompt_texts[0][1] if len(prompt_texts[0]) > 1 else prompt_texts[0][0],
                negative_prompt="",
                num_inference_steps=config["denoise_step"],
                audio_end_in_s=2097152/44100,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios
        audio_condition_ends_s = audio_condition_ends_in_s[1] if len(audio_condition_ends_in_s) > 1 else audio_condition_ends_in_s[0]
        melody_condition = melody_condition[0].detach().cpu().numpy()
        rhythm_condition = rhythm_condition[0].detach().cpu().numpy()
        gen_file = os.path.join(val_audio_dir, f"validation_{step}_{audio_condition_ends_s}~47.wav")
        gen_file_no_audio_condition = os.path.join(val_audio_dir, f"validation_{step}_{audio_condition_ends_s}~47.wav")
        original_file = os.path.join(val_audio_dir, f"original_{step}.wav")
        
        trained_mixed_file = os.path.join(val_audio_dir, f"trained_mixed_{step}.wav")
        gt_mixed_file = os.path.join(val_audio_dir, f"gt_mixed_{step}.wav")
        mix_and_save(audio_full_path, waveform_vocal, audio, pipeline, gen_file, trained_mixed_file, gt_mixed_file, original_file, gen_file_no_audio_condition, audio_condition_ends_s, eps=1e-8,)
        discription_path = os.path.join(val_audio_dir, "description.txt")
        with open(discription_path, 'a') as file:
            file.write(f'{step}\n')
            file.write(f'{prompt_texts[0][0]}\n')
            file.write(f'{audio_mid_s}\n')
            file.write(f'{prompt_texts[0][1] if len(prompt_texts[0]) > 1 else prompt_texts[0][0]}\n')
        caption = f"{prompt_texts[0][0]} | {audio_mid_s} {audio_where}| {prompt_texts[0][1] if len(prompt_texts[0]) > 1 else prompt_texts[0][0]}"
        wandb_logs[f"val_audio_{step}"] = wandb.Audio(
            gen_file, sample_rate=pipeline.vae.sampling_rate, caption=caption
        )
        wandb_logs[f"trained_mixed_{step}"] = wandb.Audio(
            trained_mixed_file, sample_rate=pipeline.vae.sampling_rate, caption=caption
        )
        wandb_logs[f"gt_mixed_{step}"] = wandb.Audio(
            gt_mixed_file, sample_rate=pipeline.vae.sampling_rate, caption=caption
        )
        if "rhythm" in condition_type:
            estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
            generated_timestamps = estimator.process(gen_file)
            input_timestamps = rhythm_timestamp[0]
            # print("input_probabilities", input_probabilities.shape)
            # print("generated_probabilities", generated_probabilities.shape)
            # hmm_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4, 6, 9, 12], fps=100)
            # input_timestamps = hmm_processor(input_probabilities)
            # generated_timestamps = hmm_processor(generated_probabilities)
            precision, recall, f1 = evaluate_f1_rhythm(input_timestamps.detach().cpu().numpy(), generated_timestamps)
            # Output results
            print(f"F1 Score: {f1:.2f}")
            score_rhythm.append(f1)
        discription_path = os.path.join(val_audio_dir, "description.txt")
        # with open(discription_path, 'a') as file:
        #     file.write(f'{prompt_texts}\n')
        #     file.write(f'gen_file: {ka}\n')
        #     file.write(f'original_file: {kb}\n')
        
    # Define the command to execute
    command = [
        "python", 
        "test.py", 
        "--audio_dir", f"{val_audio_dir}",  # Replace with your audio folder
        "--save_dir", f"{val_audio_dir}",    # Replace with your save folder
        "--voca", "True"
    ]
    print("val_audio_dir", val_audio_dir)
    # Specify the working directory (where BTC-ISMIR19 is located)
    working_directory = "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite/BTC-ISMIR19"  # Replace with the path to the BTC-ISMIR19 directory

    # Run the command
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=working_directory)
        print("Command executed successfully.")
        print("Output:\n", result.stdout)
        if result.stderr:
            print("Stderr:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the command:")
        print(e.stderr)

    from btc_chords import Chords
    CHORDS = Chords()
    for i in range(config['test_num']):
        chord_files = [f for f in os.listdir(val_audio_dir) if f.endswith(f'mixed_{i}.lab')]
        print("chord_files", chord_files)
        assert len(chord_files) == 2
        chord_path_0 = os.path.join(val_audio_dir, chord_files[0])
        with open(chord_path_0, 'r') as f:
            chord_infos_0 = f.read().splitlines()

        chroma_0 = np.zeros((12, 2097152))
        for info in chord_infos_0:
            s, t, chord = info.split(' ')
            mhot = CHORDS.chord(chord)
            final_vec = np.roll(mhot[2], mhot[0])
            final_vec = final_vec[..., None]
            chroma_0[:, int(float(s)*44100): int(float(t)*44100)] = final_vec
        chord_path_1 = os.path.join(val_audio_dir, chord_files[1])
        with open(chord_path_1, 'r') as f:
            chord_infos_1 = f.read().splitlines()

        chroma_1 = np.zeros((12, 2097152))
        for info in chord_infos_1:
            s, t, chord = info.split(' ')
            mhot = CHORDS.chord(chord)
            final_vec = np.roll(mhot[2], mhot[0])
            final_vec = final_vec[..., None]
            chroma_1[:, int(float(s)*44100): int(float(t)*44100)] = final_vec
        chroma_0 = F.interpolate(torch.from_numpy(chroma_0).unsqueeze(0), size=4756, mode='nearest')
        chroma_1 = F.interpolate(torch.from_numpy(chroma_1).unsqueeze(0), size=4756, mode='nearest')

        # 攤平成一維 + 轉成 numpy
        c0 = chroma_0.flatten().cpu().numpy()
        c1 = chroma_1.flatten().cpu().numpy()

        # 把 >0 當作 1，其餘 (0, -1) 都當 0  → 變成真正的 binary label
        c0_bin = (c0 > 0).astype(int)
        c1_bin = (c1 > 0).astype(int)

        f1 = f1_score(c0_bin, c1_bin, average='binary')
        score_chord.append(f1)

    print("score_chord", score_chord)
    accelerator.log(wandb_logs, step=global_step)
    return np.mean(score_rhythm), np.mean(score_chord)
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
def check_and_print_non_float32_parameters(model):
    non_float32_params = []
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            non_float32_params.append((name, param.dtype))
    
    if non_float32_params:
        print("Not all parameters are in float32!")
        print("The following parameters are not in float32:")
        for name, dtype in non_float32_params:
            print(f"Parameter: {name}, Data Type: {dtype}")
    else:
        print("All parameters are in float32.")

def main():
    torch.manual_seed(42)
    config = get_config()
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

    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    # os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        log_with="wandb",
    )

    if not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if config["output_dir"] is not None:
            os.makedirs(config["output_dir"], exist_ok=True)
    # decide weight precision for freezed models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # initialize models
    pipeline = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype).to(accelerator.device)
    if config.get("transformer_pretrain_ckpt") and os.path.isfile(config["transformer_pretrain_ckpt"]):
        print(f"Loading transformer checkpoint from {config['transformer_pretrain_ckpt']}")
        transformer_sd = load_file(config["transformer_pretrain_ckpt"], device="cpu")
        incompatible = pipeline.transformer.load_state_dict(transformer_sd, strict=False)
        print(f"Transformer loaded. Missing: {incompatible.missing_keys}, Unexpected: {incompatible.unexpected_keys}")
    text_encoder=pipeline.text_encoder
    projection_model=pipeline.projection_model
    vae=pipeline.vae
    noise_scheduler=pipeline.scheduler
    noise_scheduler.config.sigma_max = config["sigma_max"]
    noise_scheduler.config.sigma_min = config["sigma_min"]
    transformer = pipeline.transformer
    # key_emb_extractor = key_extractor().to(accelerator.device).float()
    struct_emb_extractor = structure_extractor().to(accelerator.device).float()
    melody_emb_extractor = MelodyEncoder().to(accelerator.device).float()
    chord_extractor = Chord_extractor().to(accelerator.device).float()
    rhythm_extractor = Rhythm_extractor().to(accelerator.device).float()
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

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    projection_model.requires_grad_(False)
    struct_emb_extractor.requires_grad_(True)
    rhythm_extractor.requires_grad_(True)
    melody_emb_extractor.requires_grad_(True)
    chord_extractor.requires_grad_(True)
    # Define a dictionary to map types to corresponding processor classes, currently only "rotary" is available.
    processor_classes = {
        "rotary": StableAudioAttnProcessor2_0_rotary,
        "rotary_double": StableAudioAttnProcessor2_0_rotary_double,
        "echo": StableAudioAttnProcessor2_0_echo,
        "rotary_free": StableAudioAttnProcessor2_0_rotary_free,
    }
    print(config["attn_processor_type"])
    # Get the processor classes based on the type
    attn_processor = processor_classes.get(config["attn_processor_type"], None)
    attn_procs = {}
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
            ).to(accelerator.device, dtype=torch.float32)
    # Load checkpoint
    if config['self_attention_ckpt'] is not None:
        load_attn1_qkv_into_pipeline(pipeline, config['self_attention_ckpt'], dtype=torch.float32, strict=False)
    if config["transformer_ckpt"] is not None:
        if "bin" in config["transformer_ckpt"]:
            state_dict = torch.load(config["transformer_ckpt"])
        elif "safetensors" in config["transformer_ckpt"]:
            state_dict = load_file(config["transformer_ckpt"], device="cpu")
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
    # class _Wrapper(AttnProcsLayers):
    #     def forward(self, *args, **kwargs):
    #         return pipeline.transformer(*args, **kwargs)

    proc_container = AttnProcsLayers(pipeline.transformer.attn_processors)
    qkv_params = []
    core = pipeline.transformer                                            
    from torch.cuda.amp import autocast
    if config["train_self"]:
        for mod_name, m in pipeline.transformer.named_modules():
            if ".attn1" in mod_name and all(hasattr(m, n) for n in ("to_q", "to_k", "to_v")):
                for lin_name in ("to_q", "to_k", "to_v"):
                    lin = getattr(m, lin_name)

                    # keep params in FP32
                    lin.to(dtype=torch.float32)

                    # correct wrapper: use the unbound class method and pass self explicitly
                    orig_fwd = type(lin).forward
                    def _fp32_forward(self, x, *args, **kwargs):
                        with autocast(enabled=False):
                            return orig_fwd(self, x.float(), *args, **kwargs)

                    lin.forward = _fp32_forward.__get__(lin, type(lin))

                    # make sure trainable
                    for p in lin.parameters():
                        p.requires_grad_(True)
                    qkv_params += list(lin.parameters())
    optimizer_class = torch.optim.AdamW
    params_to_optimize = itertools.chain(
        proc_container.parameters(), qkv_params,
        rhythm_extractor.parameters(),
        struct_emb_extractor.parameters(),
        melody_emb_extractor.parameters(),
        chord_extractor.parameters(),
        # *[model.parameters() for model in condition_extractors.values()]
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay= config['weight_decay'],
        eps=1e-08,
    )

    # Dataset and DataLoaders creation:
    dataset = AudioInversionDataset(
        config,
        device=accelerator.device,
        )
    val_size =  config["validation_num"]
    train_size = len(dataset) - val_size 

    # Ensure consistent splitting
    g = torch.Generator().manual_seed(config.get("seed", 42))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
    seed = int(config.get("seed", 42))
    set_seed(seed) 

    # DataLoader
    train_collate_fn = CollateFunction()
    val_collate_fn = CollateFunction()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
        drop_last=True,
        # prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=val_collate_fn,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
    if config["max_train_steps"] is None:
        config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config['lr_scheduler'],
        optimizer=optimizer,
        step_rules = None,
        num_warmup_steps = 500,
        num_training_steps = config['max_train_steps'] * accelerator.num_processes,
        num_cycles = 1,
        power = 1.0,
        last_epoch = -1,
    )
    print("accelerator.num_processes", accelerator.num_processes)
    # Prepare everything with our `accelerator`.
    # condition_extractor_values = list(condition_extractors.values())

    rhythm_extractor, chord_extractor, melody_emb_extractor, struct_emb_extractor, core, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        rhythm_extractor, chord_extractor, melody_emb_extractor, struct_emb_extractor, core, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
    if overrode_max_train_steps:
        config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config["num_train_epochs"] = math.ceil(config["max_train_steps"] / num_update_steps_per_epoch)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="vocal2backing",      # your W&B project
            config=config,                        # whatever hyperparams you’re logging
            init_kwargs={
                "wandb": {
                    "name": config['wand_run_name'],   # <— your chosen run name
                }
            }
        )
    global_step = 0
    first_epoch = 0
    # score_key = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config["max_train_steps"]), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    print("log_validation_first", config["log_first"])
    score_chord, score_melody, score_rhythm = 0, 0, 0
    if config["log_first"]:
        # accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            score_rhythm, score_chord = log_validation(
                accelerator, val_dataloader, rhythm_extractor, struct_emb_extractor, chord_extractor, melody_emb_extractor, config["condition_type"],
                pipeline, config, weight_dtype, global_step
            )
        # accelerator.wait_for_everyone()
    for epoch in range(first_epoch, config["num_train_epochs"]):
        for step, batch in enumerate(train_dataloader):
            # if accelerator.is_main_process:
            #     print(f'main process, step:{step}')
            #     print(f'main process, train_dataloader:{len(train_dataloader)}')
            # else:
            #     print(f'second process, step:{step}')
            #     print(f'second process, train_dataloader:{len(train_dataloader)}')
            core.train()
            # for model in condition_extractors.values():
            #     model.train()
            rhythm_extractor.train()
            struct_emb_extractor.train()
            melody_emb_extractor.train()
            chord_extractor.train()
            with accelerator.accumulate(core, rhythm_extractor, struct_emb_extractor, chord_extractor, melody_emb_extractor):
                # Convert audios to latent space
                latents = batch["audio"].to(weight_dtype)
                bsz, channels, height = latents.shape
                # Sample a random timestep for each image using uniform distribution
                t = torch.sigmoid(torch.randn(bsz, device=latents.device))
                # Calculate the noise schedule parameters for those timesteps
                alphas, sigmas = get_alphas_sigmas(t)  # get_alphas_sigmas should be defined as in the wrapper
                alphas = alphas[:, None, None].to(weight_dtype)  # Shape to match latents
                sigmas = sigmas[:, None, None].to(weight_dtype)
                # Sample noise and add it to the latent
                noise = torch.randn_like(latents)
                noisy_latents = latents * alphas + noise * sigmas
                # Determine the target for v_prediction
                if noise_scheduler.config.prediction_type == "v_prediction":
                    targets = alphas * noise - sigmas * latents
                else:
                    targets = noise  # For epsilon, the target is just the noise
                prompt_texts = batch["prompt_texts"]
                # desired_repeats = 128 // 64  # Number of repeats needed
                extracted_audio_condition = latents
                extracted_melody_condition = melody_emb_extractor(batch["melody_condition"])
                extracted_rhythm_condition = rhythm_extractor(batch["rhythm_condition"].float())
                
                struct_condition = batch["structures_ids"]
                extracted_struct_condition = struct_emb_extractor(struct_condition).transpose(1,2)
                extracted_chord_condition = chord_extractor(batch["chord_condition"].squeeze(1))
                extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                extracted_rhythm_condition = F.interpolate(extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
                for i in range(len(prompt_texts)):
                    if len(prompt_texts[i]) == 1:
                        prompt_texts[i].append(prompt_texts[i][0])
                    elif len(prompt_texts[i]) > 2:
                        prompt_texts[i] = prompt_texts[i][:2]
                    rand_num = random.random()
                    total_seconds = 2097152/44100
                    audio_condition_ends_in_s = batch['audio_condition_ends'][i]
                    audio_condition_ends = [int(s / total_seconds * 1024) for s in audio_condition_ends_in_s]
                    if struct_condition[i][0] == 0 and len(audio_condition_ends) > 1 and random.random() < 0.5:
                        # mask the audio condition for intro
                        extracted_audio_condition[i] = torch.zeros_like(extracted_audio_condition[i])
                    elif len(audio_condition_ends) > 1:
                        # mask the audio condition for segments after the reference segment
                        extracted_audio_condition[i][:,audio_condition_ends[1]:] = 0
                    else:
                        # mask the audio condition if only one segment is given
                        extracted_audio_condition[i][:,audio_condition_ends[0]:] = 0
                    rand_num = random.random()
                    if rand_num < 0.15:
                        prompt_texts[i][0] = ""
                    elif rand_num < 0.30:
                        prompt_texts[i][1] = ""
                    elif rand_num < 0.45:
                        prompt_texts[i][0] = ""
                        prompt_texts[i][1] = ""

                    if random.random() < 0.2:
                        extracted_melody_condition[i] = torch.zeros_like(extracted_melody_condition[i])
                    if random.random() < 0.2:
                        extracted_rhythm_condition[i] = torch.zeros_like(extracted_rhythm_condition[i])
                    if random.random() < 0.2:
                        extracted_struct_condition[i] = torch.zeros_like(extracted_struct_condition[i])
                    if random.random() < 0.4:
                        extracted_chord_condition[i] = torch.zeros_like(extracted_chord_condition[i])
                    if random.random() < 0.4:
                        extracted_audio_condition[i] = torch.zeros_like(extracted_audio_condition[i])
                    if random.random() < 0.05:
                        extracted_audio_condition[i] = torch.zeros_like(extracted_audio_condition[i])
                        extracted_melody_condition[i] = torch.zeros_like(extracted_melody_condition[i])
                        extracted_rhythm_condition[i] = torch.zeros_like(extracted_rhythm_condition[i])
                        extracted_struct_condition[i] = torch.zeros_like(extracted_struct_condition[i])
                        extracted_chord_condition[i] = torch.zeros_like(extracted_chord_condition[i])
               
                transposed_prompt_text = [list(col) for col in zip(*prompt_texts)]
                with torch.no_grad():
                    # print(prompt_texts)
                    prompt_embeds_1 = pipeline.encode_prompt(
                        prompt=transposed_prompt_text[0],
                        device="cuda",
                        do_classifier_free_guidance=False,
                    )
                    prompt_embeds_2 = pipeline.encode_prompt(
                        prompt=transposed_prompt_text[1],
                        device="cuda",
                        do_classifier_free_guidance=False,
                    )
                    audio_start_in_s = batch["seconds_start"]
                    audio_end_in_s = batch["seconds_end"]
                    # Encode duration
                    seconds_start_hidden_states, seconds_end_hidden_states = pipeline.encode_duration(
                        audio_start_in_s,
                        audio_end_in_s,
                        device="cuda",
                        do_classifier_free_guidance=False,
                        batch_size=bsz,
                    )
                audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2).to(weight_dtype)
                text_audio_duration_embeds_1 = torch.cat(
                    [prompt_embeds_1, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
                )

                text_audio_duration_embeds_2 = torch.cat(
                    [prompt_embeds_2, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
                )
                # print("extracted_rhythm_condition", extracted_rhythm_condition.shape)
                # print("extracted_melody_condition", extracted_melody_condition.shape)
                # print("extracted_key_condition", extracted_key_condition.shape)
                # print("extracted_struct_condition", extracted_struct_condition.shape)
                # print("extracted_audio_condition", extracted_audio_condition.shape)
                # print("extracted_chord_condition", extracted_chord_condition.shape)
                extracted_condition = torch.concat((extracted_rhythm_condition, extracted_melody_condition, extracted_struct_condition, extracted_audio_condition, extracted_chord_condition), dim=1).to(weight_dtype)
                extracted_condition = extracted_condition.transpose(1, 2)
                # This rotary_embedding is for self attention layers in Stable-audio 
                rotary_embed_dim = pipeline.transformer.config.attention_head_dim // 2
                rotary_embedding = get_1d_rotary_pos_embed(
                    rotary_embed_dim,
                    latents.shape[2] + audio_duration_embeds.shape[1],
                    use_real=True,
                    repeat_interleave_real=False,
                )      
                audio_mid_s = batch["audio_mid_s"]         
                with accelerator.autocast():
                    # print("noisy_latents", noisy_latents.dtype)
                    # print("extracted_condition", extracted_condition.dtype)
                    # print("text_audio_duration_embeds_1", text_audio_duration_embeds_1.dtype)
                    # print("t", t.dtype)
                    model_pred = core(
                        noisy_latents,
                        t,  # Use continuous t for conditioning
                        encoder_hidden_states=text_audio_duration_embeds_1,
                        encoder_hidden_states_2=text_audio_duration_embeds_2,
                        audio_mid_s = audio_mid_s,
                        encoder_hidden_states_con=extracted_condition,
                        global_hidden_states=audio_duration_embeds,
                        rotary_embedding=rotary_embedding,
                        return_dict=False,
                    )[0]
                    # Compute the loss
                    # print("model_pred", model_pred.dtype)
                    # print("targets", targets.dtype)
                    loss = F.mse_loss(model_pred.float(), targets.float(), reduction="mean")
                
                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            core.parameters(),
                            # key_emb_extractor.parameters(),
                            struct_emb_extractor.parameters(),
                            melody_emb_extractor.parameters(),
                            chord_extractor.parameters(),
                            rhythm_extractor.parameters(),
                            # *[model.parameters() for model in condition_extractors.values()]
                        )
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # gc.collect()
                    # torch.cuda.empty_cache()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                audios = []
                progress_bar.update(1)
                global_step += 1
            
                if accelerator.is_main_process:
                    if global_step % config["checkpointing_steps"] == 0:
                        save_dir = os.path.join(config["output_dir"], f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)

                        core_unwrap = accelerator.unwrap_model(core)
                        
                        # key_emb_path = os.path.join(save_dir, "key_emb.safetensors")
                        struct_emb_path = os.path.join(save_dir, "struct_emb.safetensors")
                        melody_emb_path = os.path.join(save_dir, "melody_emb.safetensors")
                        chord_cnn_path = os.path.join(save_dir, "chord_cnn.safetensors")
                        rhythm_cnn_path = os.path.join(save_dir, "rhythm_cnn.safetensors")
                        qkv_path = os.path.join(save_dir, "attn1_qkv.safetensors")

                        # save as .safetensors (optionally include metadata)
                        if config["train_self"]:
                            # --- cleanup from previous bug: remove dir if it exists at the file path ---
                            if os.path.isdir(qkv_path):
                                import shutil
                                shutil.rmtree(qkv_path)
                            qkv_state = {
                            k: v.detach().cpu().contiguous()
                            for k, v in core_unwrap.state_dict().items()
                                if (".attn1.to_q." in k) or (".attn1.to_k." in k) or (".attn1.to_v." in k)
                            }
                            save_file(qkv_state, qkv_path)
                        # save_file(key_emb_extractor.state_dict(), key_emb_path)
                        save_file(struct_emb_extractor.state_dict(), struct_emb_path)
                        save_file(melody_emb_extractor.state_dict(), melody_emb_path)
                        save_file(chord_extractor.state_dict(), chord_cnn_path)
                        save_file(rhythm_extractor.state_dict(), rhythm_cnn_path)

                        # accelerator.print(f"Saved QKV to {qkv_path}")
                        save_attn_processors(pipeline, save_dir)

                        # if other ranks will read anything under save_dir afterwards:
                        # accelerator.wait_for_everyone()
                # accelerator.wait_for_everyone() 
                if global_step % config["validation_steps"] == 0:
                    # accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        score_rhythm, score_chord = log_validation(
                            accelerator, val_dataloader, rhythm_extractor, struct_emb_extractor, chord_extractor, melody_emb_extractor, config["condition_type"],
                            pipeline, config, weight_dtype, global_step
                        )
                    # accelerator.wait_for_everyone()
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "score_rhythm": score_rhythm, "score_chord": score_chord}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config["max_train_steps"]:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
    

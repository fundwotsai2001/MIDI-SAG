import torch
import soundfile as sf
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
    StableAudioAttnProcessor2_0_rotary_double,
)
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file  # Import safetensors
import os
import numpy as np
import matplotlib.pyplot as plt
from config_inference import get_config
import argparse
import json
from utils.extract_conditions import compute_melody_v2, compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm, calculate_beats_and_downbeats, create_activations_from_timestamps, compute_rhythm_beatnet
from utils.stable_audio_dataset_utils import Stereo, PhaseFlipper
import random
from torchaudio import transforms as T
import torchaudio
import re
import bisect
from btc_chords import Chords
import mido


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

    chroma = torch.from_numpy(chroma).unsqueeze(0).float().cuda()  # shape (1, 12, 2097152)
    chroma = F.interpolate(chroma, size=1024, mode='linear', align_corners=False)  # shape (1, 12, 4756)
    return chroma
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
    print("Loaded QKV. Missing:", incompatible.missing_keys, "Unexpected:", incompatible.unexpected_keys)

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
        audio = audio[:, int(segment_starts*44100):]
        return audio
    except RuntimeError:
        print(f"Failed to decode audio file: {filename}")
        return None
def format_key(entry):
    key = entry.get("key")
    mode = (entry.get("mode") or "").lower()
    if not key:
        return None
    return key + ("m" if mode == "minor" else "")
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
    rhythm_cnn_extractor = Rhythm_extractor().to("cuda").float()
    struct_emb_extractor = structure_extractor().to("cuda").float()
    melody_emb_extractor = MelodyEncoder().to("cuda").float()
    chord_extractor = Chord_extractor().to("cuda").float()
    if config["checkpoint_path"]:
        config["self_attention_ckpt"] = os.path.join(config["checkpoint_path"], "attn1_qkv.safetensors")
        config["transformer_ckpt"] = os.path.join(config["checkpoint_path"], "attn_procs.safetensors")
        config["rhythm_cnn_ckpt"] = os.path.join(config["checkpoint_path"], "rhythm_cnn.safetensors")
        config["struct_emb_ckpt"] = os.path.join(config["checkpoint_path"], "struct_emb.safetensors")
        config["melody_emb_ckpt"] = os.path.join(config["checkpoint_path"], "melody_emb.safetensors")
        config["chord_cnn_ckpt"] = os.path.join(config["checkpoint_path"], "chord_cnn.safetensors")
    else:
        config["self_attention_ckpt"] = None
        config["transformer_ckpt"] = None
        config["rhythm_cnn_ckpt"] = None
        config["struct_emb_ckpt"] = None
        config["melody_emb_ckpt"] = None
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
    if config['rhythm_cnn_ckpt'] is not None:
        state_dict = load_file(config['rhythm_cnn_ckpt'])
        # Check keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        rhythm_cnn_extractor.load_state_dict(new_state_dict)
        print("load rhythm_cnn_extractor")
    if config['struct_emb_ckpt'] is not None:
        state_dict = load_file(config['struct_emb_ckpt'])
        # Check keys
        # print(f"Loaded {len(state_dict)} tensors:")
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
        # print(f"Loaded {len(state_dict)} tensors:")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        melody_emb_extractor.load_state_dict(new_state_dict)
        print("load melody_emb_extractor")

    if config["weight_dtype"] == "fp16":
        weight_dtype = torch.float16
    elif config["weight_dtype"] == "bp16":
        weight_dtype = torch.bfloat16
    if config["apadapter"]:
        from pipeline.stable_audio_multi_cfg_pipe import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained("/volume/fundwo-test/MuseControlLite/stable-audio", torch_dtype=weight_dtype)
        if config['self_attention_ckpt'] is not None:
            load_attn1_qkv_into_pipeline(pipe, config['self_attention_ckpt'], dtype=torch.float32, strict=False)
        pipe.scheduler.config.sigma_max = config["sigma_max"]
        pipe.scheduler.config.sigma_min = config["sigma_min"]
        transformer = pipe.transformer
        attn_procs = {}
        processor_classes = {
            "rotary": StableAudioAttnProcessor2_0_rotary,
            "rotary_double": StableAudioAttnProcessor2_0_rotary_double,
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
    # print("structure2id", structure2id)
    # print("key2idx", key2idx)
    # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
    total_seconds = 2097152/44100
    if config['use_audio_mask']:
        audio_mask_start = int(config["audio_mask_start_seconds"] / total_seconds * 1024) # 1024 is the latent length for 2097152/44100 seconds
        audio_mask_end = int(config["audio_mask_end_seconds"] / total_seconds * 1024)
    elif config['use_musical_attribute_mask']:
        musical_attribute_mask_start = int(config["musical_attribute_mask_start_seconds"] / total_seconds * 1024)
        musical_attribute_mask_end = int(config["musical_attribute_mask_end_seconds"] / total_seconds * 1024)
    with torch.no_grad():
        transformer.eval()
        melody_midi = config["melody_midi"][0]
        midi = mido.MidiFile(melody_midi)
        print(f"Duration: {midi.length:.2f} seconds")
        with open(config['structure_tag'], "r", encoding="utf-8") as f:
            config['structure_tag'] = json.load(f)
        with open(config['structure_start_seconds'], "r", encoding="utf-8") as f:
            config['structure_start_seconds'] = json.load(f)
        structure_starts_seconds = config['structure_start_seconds']
        print("structure_starts_seconds", structure_starts_seconds)
        config['key_conditions'] = []
        with open(config['key_info'][0], "r", encoding="utf-8") as f:
            data = json.load(f)
            files = data.get("processed_files", [])
            for entry in files:
                out = format_key(entry)
        for i in range(len(structure_starts_seconds)):
            config.setdefault('key_conditions', []).append(str(out))
        print("config['key_conditions']", config['key_conditions'])
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
        print("structures_ids", structures_ids)
        config['structure_start_seconds'].append(config['structure_start_seconds'][-1] + 2097152/44100)
        for i, prompt_texts in enumerate(config['text']):
            backing_audio = torch.empty(2, 0)
            segments_list = list(range(len(config['structure_tag'])))
            if "intro" in config['structure_tag']:
                segments_list[0], segments_list[1] = segments_list[1], segments_list[0]
                tensors = {name: torch.empty(0) for name in config['structure_tag']}
            print("segments_list", segments_list)
            for s, segments in enumerate(segments_list):
                print(f"Generating segment {segments + 1}/{len(config['structure_tag'])} for prompt {i + 1}/{len(config['text'])}")
                print("prompt_texts", prompt_texts[segments])
                print("structure_tag", config['structure_tag'][segments])
                beat_times_all, downbeat_times_all = calculate_beats_and_downbeats(melody_midi)
                beat_step = beat_times_all[-1] - beat_times_all[-2]
                downbeat_step = downbeat_times_all[-1] - downbeat_times_all[-2]
                extra_beat = [beat_times_all[-1] + beat_step * (i + 1) for i in range(len(beat_times_all))]
                beat_times_all = beat_times_all + extra_beat
                down_beat_step = downbeat_times_all[-1] - downbeat_times_all[-2]
                extra_downbeat = [downbeat_times_all[-1] + downbeat_step * (i + 1) for i in range(len(downbeat_times_all))]
                downbeat_times_all = downbeat_times_all + extra_downbeat
                if config["apadapter"]:
                    gt_vocal_audio_file = config["vocal_audio_files"][0]
                    if config["no_text"] is True:
                        prompt_texts[segments] = ""
                    description_path = os.path.join(output_dir, "description.txt")
                    if "audio" in config["condition_type"] and s != 0:
                        if s == 1:
                            print(tensors[config['structure_tag'][segments + 1]].shape)
                            audio = tensors[config['structure_tag'][segments + 1]][:, :int(2097152 - (44100*config['structure_start_seconds'][s]))].unsqueeze(0).to(weight_dtype).cuda()
                        if s > 1:
                            audio = tensors[config['structure_tag'][segments-1]][:, int((config['structure_start_seconds'][segments] - config['structure_start_seconds'][s - 1])*44100):].unsqueeze(0).to(weight_dtype).cuda()
                            print(f"{config['structure_start_seconds'][segments]} ~ {config['structure_start_seconds'][segments - 1] + 2097152/44100} seconds will be reference audio")
                        # print("output", output.shape)
                        audio_condition = torch.zeros((1, 64, 1024), device="cuda")
                        print("audio", audio.shape)
                        audio_condition_ref = pipe.vae.encode(audio).latent_dist.sample()
                        print("audio_condition_ref", audio_condition_ref.shape)
                        if s > 1:
                            audio_condition[:,:,:audio_condition_ref.shape[2]] = audio_condition_ref
                        else:
                            audio_condition[:,:,1024 - audio_condition_ref.shape[2]:] = audio_condition_ref
                        # print("audio_condition", audio_condition.shape)
                        desired_repeats = 128 // 64
                        extracted_audio_condition = audio_condition.repeat_interleave(desired_repeats, dim=1).float()
                        masked_extracted_audio_condition = torch.zeros_like(extracted_audio_condition)
                        # print("audio_condition", audio_condition.shape)
                    else: 
                        extracted_audio_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_audio_condition = extracted_audio_condition
                    if "key" in config['condition_type']:
                        key_start = int(config['structure_start_seconds'][segments] / (2097152/44100) * 1024)
                        key_segment = key_ids[key_start: key_start + 1024]
                        extracted_key_condition = key_emb_extractor(key_segment.cuda().unsqueeze(0)).transpose(1,2)
                        # print("key_condition", extracted_key_condition.shape)
                        masked_extracted_key_condition = torch.zeros_like(extracted_key_condition)
                    else: 
                        extracted_key_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_key_condition = extracted_key_condition
                    if "strucure" in config['condition_type']:
                        structures_ids_start = int(config['structure_start_seconds'][segments] / (2097152/44100) * 1024)
                        print("structures_ids_start", structures_ids_start)
                        print("structure_start_seconds", config['structure_start_seconds'][segments])
                        structures_ids_segment = structures_ids[structures_ids_start:structures_ids_start + 1024]
                        extracted_struct_condition = struct_emb_extractor(structures_ids_segment.cuda().unsqueeze(0)).transpose(1,2)
                        # print("structure_condition", extracted_struct_condition.shape)
                        masked_extracted_struct_condition = torch.zeros_like(extracted_struct_condition)
                    else: 
                        extracted_struct_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_struct_condition = extracted_struct_condition
                    # For single conition, we can utilize the full cross-attention dimension 768, instead of 768/4 in MuseControlLite_inference_on_the_fly_all.py
                    if "melody" in config["condition_type"]:
                        melody_condition = compute_melody_v2(gt_vocal_audio_file, segment_starts = config['structure_start_seconds'][segments])
                        melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
                        extracted_melody_condition = melody_emb_extractor(melody_condition)
                        # print("melody_condition", extracted_melody_condition.shape)
                        # extracted_melody_condition = condition_extractors["melody"](melody_condition.to(torch.float32))
                        masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
                        extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                        masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                    else: 
                        extracted_melody_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_melody_condition = extracted_melody_condition
                    if "chord" in config["condition_type"]:
                        chord_condition = extract_chords_lab(config['chord_info'][0], segment_starts = config['structure_start_seconds'][segments])
                        extracted_chord_condition = chord_extractor(chord_condition)
                        # print("chord_condition", extracted_chord_condition.shape)
                        # extracted_melody_condition = condition_extractors["melody"](melody_condition.to(torch.float32))
                        masked_extracted_chord_condition = torch.zeros_like(extracted_chord_condition)
                        # extracted_chord_condition = F.interpolate(extracted_chord_condition, size=1024, mode='linear', align_corners=False)
                        # masked_extracted_chord_condition = F.interpolate(masked_extracted_chord_condition, size=1024, mode='linear', align_corners=False)
                    else: 
                        extracted_chord_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_chord_condition = extracted_chord_condition
                    if "rhythm" in config["condition_type"]:
                        beat_times = sublist_between(beat_times_all, config['structure_start_seconds'][segments], config['structure_start_seconds'][segments] + 2097152/44100)
                        downbeat_times = sublist_between(downbeat_times_all, config['structure_start_seconds'][segments], config['structure_start_seconds'][segments] + 2097152/44100)
                        # print("beat_times: ", beat_times)
                        # print("downbeat_times: ", downbeat_times)
                        beat_times = [x - config['structure_start_seconds'][segments] for x in beat_times]
                        downbeat_times = [x - config['structure_start_seconds'][segments] for x in downbeat_times]
                        downbeat_times = []
                        beat_times = []
                        print("beat_times: ", beat_times)
                        print("downbeat_times: ", downbeat_times)
                        rhythm_condition = create_activations_from_timestamps(beat_times, downbeat_times)
                        extracted_rhythm_condition = torch.from_numpy(rhythm_condition).cuda().unsqueeze(0).repeat_interleave(128//2, dim=1).float()
                        # print("rhythm_condition", extracted_rhythm_condition.shape)
                        masked_extracted_rhythm_condition = torch.zeros_like(extracted_rhythm_condition)
                        extracted_rhythm_condition = F.interpolate(extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
                        masked_extracted_rhythm_condition = F.interpolate(masked_extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
                    else: 
                        extracted_rhythm_condition = torch.zeros((1, 128, 1024), device="cuda")
                        masked_extracted_rhythm_condition = extracted_rhythm_condition
                    
                    # Use multiple cfg
                    extracted_condition = torch.concat((extracted_rhythm_condition, extracted_melody_condition, extracted_key_condition, extracted_struct_condition, extracted_audio_condition, extracted_chord_condition), dim=1)
                    masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_melody_condition, masked_extracted_key_condition, masked_extracted_struct_condition, masked_extracted_audio_condition, masked_extracted_chord_condition), dim=1)
                    extracted_condition = torch.concat((masked_extracted_condition, masked_extracted_condition, extracted_condition), dim=0)
                    extracted_condition = extracted_condition.transpose(1, 2)
                    waveform = pipe(
                        extracted_condition = extracted_condition, 
                        prompt=prompt_texts[segments],
                        negative_prompt=negative_text_prompt,
                        num_inference_steps=config["denoise_step"],
                        guidance_scale_text=config["guidance_scale_text"],
                        guidance_scale_con=config["guidance_scale_con"],
                        num_waveforms_per_prompt=1,
                        audio_end_in_s=2097152 / 44100,
                        generator=generator,
                    ).audios 
                    # print(f"{i}")      
                    
                    # output = waveform[0]
                    tensors[config['structure_tag'][segments]] = waveform[0]
                    # print("output", output.shape)
                    # print("backing_audio", backing_audio.shape)
                    print(f"Generate {config['structure_tag'][segments]} segment")
                    if segments == 0 and s == 1:
                        backing_audio = torch.cat((tensors[config['structure_tag'][segments]][:, :int(44100 * config['structure_start_seconds'][s])].cpu(), backing_audio), dim=1)
                        print(f"generate 0 ~ {config['structure_start_seconds'][s]} seconds")
                        # save_segments = os.path.join(output_dir, f"0_{config['structure_start_seconds'][s]}.wav")
                        # sf.write(save_segments, tensors[config['structure_tag'][segments]][:, :int(44100 * config['structure_start_seconds'][s])].T.float().cpu().numpy(), pipe.vae.sampling_rate)    
                    else:
                        backing_audio = torch.cat((backing_audio, tensors[config['structure_tag'][segments]][:, :int(44100 * (config['structure_start_seconds'][segments + 1] - config['structure_start_seconds'][segments]))].cpu()), dim=1)
                        print(f"generate {config['structure_start_seconds'][segments]} ~ {config['structure_start_seconds'][segments + 1]} seconds")
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
            # print("waveform_vocal", waveform_vocal.shape)
            # print("backing_audio", backing_audio.shape)
            final_length = midi.length
            waveform_vocal, backing_audio = pad_to_match(waveform_vocal, backing_audio, int(final_length*44100))
            # min_len = min(waveform_vocal.shape[1], backing_audio.shape[1])
            # waveform_vocal = waveform_vocal[:, :min_len]
            # backing_audio = backing_audio[:, :min_len]
            g8 = float(10 ** (-8.0 / 20.0))
            g1 = float(10 ** (-1.0 / 20.0))
            eps=1e-8
            v_scale = g8 / waveform_vocal.abs().amax().clamp_min(eps)
            o_scale = g8 / backing_audio.abs().amax().clamp_min(eps)
            mix = 0.5 * (waveform_vocal.cpu() * v_scale.cpu() + backing_audio.cpu() * o_scale.cpu())
            mixed_file_path = os.path.join(output_dir, f"mixed_{i}.wav")
            sf.write(mixed_file_path, mix.T.float().cpu().numpy(), pipe.vae.sampling_rate)
        data_to_save = {"config": config}

        # if "dynamics" in config["condition_type"]:
        # data_to_save["score_dynamics"] = np.mean(score_dynamics)

        # # if "rhythm" in config["condition_type"]:
        # data_to_save["score_rhythm"] = np.mean(score_rhythm)

        # # if "melody" in config["condition_type"]:
        # data_to_save["score_melody"] = np.mean(score_melody)
        # print(data_to_save)
        file_path = os.path.join(output_dir, "result.txt")
        with open(file_path, "w") as file:
            json.dump(data_to_save, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AP-adapter Inference Script")
    config = get_config()  # Pass the parsed arguments to get_config
    main(config)

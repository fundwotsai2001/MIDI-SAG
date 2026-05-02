import torch
import soundfile as sf
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
)
import torch.nn.functional as F
from safetensors.torch import load_file  # Import safetensors
import os
import numpy as np
from config_inference import get_config
import argparse
from utils.extract_conditions import calculate_beats_and_downbeats, create_activations_from_timestamps
from utils.audio_processing import mix_audio, extract_chords_lab, sublist_between, load_audio_file
import random
from pathlib import Path
import torch.nn as nn
import re
# ── RMVPE / F0 melody encoder ─────────────────────────────────────────
from rmvpe import RMVPE  # noqa: E402
import librosa
import math
RMVPE_CKPT     = "./MIDI-SAG_checkpoints/rmvpe_model.pt"
F0_MELODY_CKPT = "./MIDI-SAG_checkpoints/melody_encoder.pt"
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


def _safe_filename_component(text: str, max_len: int = 80) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (component[:max_len] or "prompt")


def _normalize_prompt_list(prompt_value):
    if isinstance(prompt_value, str):
        prompts = [prompt_value]
    else:
        prompts = list(prompt_value)
    return prompts or [""]


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    output_dir = config["output_dir"] + f"text_{config['guidance_scale_text']}_con_{config['guidance_scale_con']}"
    os.makedirs(output_dir, exist_ok=True)
    weight_dtype = torch.float32
    melody_emb_extractor = MelodyEncoder().to("cuda").float()
    chord_extractor = Chord_extractor().to("cuda").float()
    
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
    structures_ids_expand = [6]*1024
    structures_ids = torch.tensor(structures_ids_expand)
    if config["weight_dtype"] == "fp16":
        weight_dtype = torch.float16
    elif config["weight_dtype"] == "bp16":
        weight_dtype = torch.bfloat16
    if config["apadapter"]:
        from pipeline.stable_audio_multi_cfg_pipe import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
        pipe.scheduler.config.sigma_max = config["sigma_max"]
        pipe.scheduler.config.sigma_min = config["sigma_min"]
        transformer = pipe.transformer
        attn_procs = {}
        processor_classes = {
            "rotary": StableAudioAttnProcessor2_0_rotary,
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
                keys = list(state_dict.keys())

            for name, processor in attn_procs.items():
                if isinstance(processor, attn_processor):
                    if 'echo' in config["attn_processor_type"]:
                        weight_name_proj_gamma = name + ".proj_gamma.weight"
                        weight_name_proj_beta = name + ".proj_beta.weight"
                        weight_name_hidden_proj = name + ".hidden_proj.weight"
                        weight_name_con_proj = name + ".con_proj.weight"
                        processor.proj_gamma.weight = torch.nn.Parameter(state_dict[weight_name_proj_gamma].to(torch.float32))
                        processor.proj_beta.weight = torch.nn.Parameter(state_dict[weight_name_proj_beta].to(torch.float32))
                        processor.hidden_proj.weight = torch.nn.Parameter(state_dict[weight_name_hidden_proj].to(torch.float32))
                        processor.con_proj.weight = torch.nn.Parameter(state_dict[weight_name_con_proj].to(torch.float32))
                    if "rotary" in config["attn_processor_type"]:
                        weight_name_v = name + ".to_v_ip.weight"
                        weight_name_k = name + ".to_k_ip.weight"
                        conv_out_weight = name + ".conv_out.weight"
                        processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].to(torch.float32))
                        processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].to(torch.float32))
                        processor.conv_out.weight = torch.nn.Parameter(state_dict[conv_out_weight].to(torch.float32))
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
    if config['self_attention_ckpt'] is not None:
        load_attn1_qkv_into_pipeline(pipe, config['self_attention_ckpt'], dtype=torch.float32, strict=False)
    pipe = pipe.to("cuda")
    negative_text_prompt = config["negative_text_prompt"]
    # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
    with torch.no_grad():
       
        gt_vocal_audio_file = config['vocal_audio_file']
        prompt_texts = _normalize_prompt_list(config['text_prompt'])
        
        song_name = config['vocal_audio_file'].split('/')[-1].split(".wav")[0]
        
        transformer.eval()            
        waveform_vocal = load_audio_file(gt_vocal_audio_file, segment_starts= 0)
        seconds = waveform_vocal.shape[1] / 44100
        number_of_segments = int(np.floor(seconds / (2097152 / 44100)))
        seconds_list = [2097152 / 44100 * y for y in range(number_of_segments)]
        seconds_starts = 0
        _window_start_s = 0
        
        if config["no_text"] is True:
            prompt_texts = [""]
        # if "structure" in config['condition_type']:
        extracted_struct_condition = torch.zeros((1, 176, 1024), device="cuda")
        masked_extracted_struct_condition = extracted_struct_condition
        extracted_audio_condition = torch.zeros((1, 64, 1024), device="cuda")
        masked_extracted_audio_condition = extracted_audio_condition
        # For single conition, we can utilize the full cross-attention dimension 768, instead of 768/4 in MuseControlLite_inference_on_the_fly_all.py
        if "structure" in config['condition_type']:
            print("using structure condition")
            structures_ids_start = int(_window_start_s / (2097152/44100) * 1024)
            # print("structures_ids_start", structures_ids_start)
            # print("structure_start_seconds", _window_start_s)
            structures_ids_segment = structures_ids[structures_ids_start:structures_ids_start + 1024]
            extracted_struct_condition = struct_emb_extractor(structures_ids_segment.cuda().unsqueeze(0)).transpose(1,2)
            # print("structure_condition", extracted_struct_condition.shape)
            masked_extracted_struct_condition = torch.zeros_like(extracted_struct_condition)
        else: 
            extracted_struct_condition = torch.zeros((1, 176, 1024), device="cuda")
            masked_extracted_struct_condition = extracted_struct_condition
        if "melody" in config["condition_type"]:
            print("using melody condition")
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
            print("using chord condition")
            print("chord_condition", chord_condition.shape)
            print(f"load condition from {config['chord_info']}")
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
            MIDI_FILE = config['vocal_midi_file']
            BEAT_FILE = config['vocal_beat_file']
            DOWNBEAT_FILE = config['vocal_downbeat_file']
            if MIDI_FILE is not None:
                beat_times_all, downbeat_times_all = calculate_beats_and_downbeats(MIDI_FILE)
                print(f"Loaded {len(beat_times_all)} beats and {len(downbeat_times_all)} downbeats from MIDI")
            else:
                # Load beat times from text file and derive downbeats heuristically
                beat_times_all = []
                with open(BEAT_FILE, 'r') as _bf:
                    for _line in _bf:
                        _line = _line.strip()
                        if _line and not _line.startswith('#'):
                            beat_times_all.append(float(_line))
                downbeat_times_all = []
                with open(DOWNBEAT_FILE, 'r') as _bf:
                    for _line in _bf:
                        _line = _line.strip()
                        if _line and not _line.startswith('#'):
                            downbeat_times_all.append(float(_line))
            beat_times = sublist_between(beat_times_all, seconds_starts, 2097152/44100 + seconds_starts)
            beat_times = [x - seconds_starts for x in beat_times]
            downbeat_times = sublist_between(downbeat_times_all, seconds_starts, 2097152/44100 + seconds_starts)
            downbeat_times = [x - seconds_starts for x in downbeat_times]
            print(beat_times)
            print(downbeat_times)
            print("using rhythm condition")
            rhythm_condition = create_activations_from_timestamps(beat_times, downbeat_times)
            extracted_rhythm_condition = rhythm_extractor(torch.from_numpy(rhythm_condition).cuda().unsqueeze(0).float()) 
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
        waveform_vocal_slice = waveform_vocal[:, int(seconds_starts*44100): int((seconds_starts + 2097152 / 44100)*44100)]
        # ---------------- Example usage ----------------
        # Assume you start with int16 PCM and want float32 in [-1, 1]:
        # wav_a_int16, wav_b_int16: torch.int16 tensors shaped [T] or [C, T]
        waveform_vocal_slice = (waveform_vocal_slice.to(torch.float32) / 32768.0).clamp(-1, 1)
        total_prompts = len(prompt_texts)

        for prompt_index, prompt_text in enumerate(prompt_texts, start=1):
            print(f"Generating prompt {prompt_index}/{total_prompts}: {prompt_text or '[no text]'}")
            prompt_generator = torch.Generator().manual_seed(42)
            waveform = pipe(
                extracted_condition = extracted_condition,
                prompt=prompt_text,
                negative_prompt=negative_text_prompt,
                num_inference_steps=config["denoise_step"],
                guidance_scale_text=config["guidance_scale_text"],
                guidance_scale_con=config["guidance_scale_con"],
                num_waveforms_per_prompt=1,
                audio_end_in_s=2097152 / 44100,
                generator=prompt_generator,
            ).audios
            backing_audio = waveform[0].float().cpu()
            backing_audio = (backing_audio.to(torch.float32) / 32768.0).clamp(-1, 1)
            mix = mix_audio(waveform_vocal_slice, backing_audio, target_dbfs=-18.0, out_peak_dbfs=-1.0)
            prompt_label = _safe_filename_component(prompt_text)
            mixed_file_path = os.path.join(
                output_dir,
                f"mixed_{song_name}_prompt_{prompt_index:02d}_{prompt_label}.wav",
            )
            sf.write(mixed_file_path, mix.T.float().cpu().numpy(), pipe.vae.sampling_rate)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocal_audio_file", required=True, help="Path(s) to input audio file(s)")
    parser.add_argument("--text_prompt", required=True, nargs="+", help="One or more text prompts for generation")
    parser.add_argument("--vocal_beat_file", default=None, required=False, help="Path(s) to input vocal beat file(s)")
    parser.add_argument("--vocal_downbeat_file", default=None,
                        help="Optional path to detected-downbeat times txt (one time/line, '#' header ok). "
                             "If omitted, falls back to beat_times[::4].")
    parser.add_argument("--chord_file", required=True, help="Path(s) to input chord file(s)")
    parser.add_argument("--checkpoint_path", required=True, help="Path(s) to input checkpoint file(s)")
    parser.add_argument("--output_dir", required=True, help="Path(s) for output directory")
    parser.add_argument("--vocal_midi_file", default=None, help="Path to MIDI file for beat/downbeat extraction. If not provided, beats come from --vocal_beat_file")
    args = parser.parse_args()

    config = get_config()
    config["vocal_audio_file"] = args.vocal_audio_file
    config["vocal_midi_file"] = args.vocal_midi_file
    config["text_prompt"] = args.text_prompt
    config['chord_info'] = args.chord_file
    config['vocal_beat_file'] = args.vocal_beat_file
    config['vocal_downbeat_file'] = args.vocal_downbeat_file
    config['checkpoint_path'] = args.checkpoint_path
    config['output_dir'] = args.output_dir
    main(config)

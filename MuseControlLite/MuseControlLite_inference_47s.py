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
from utils.extract_conditions import compute_melody_v2, create_activations_from_timestamps
from utils.condition_extractors import MelodyEncoder, Chord_extractor
from utils.audio_processing import mix_audio, extract_chords_lab, sublist_between, load_audio_file
import random
from pathlib import Path

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    generator = torch.Generator().manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    output_dir = config["output_dir"] + f"text_{config['guidance_scale_text']}_con_{config['guidance_scale_con']}"
    os.makedirs(output_dir, exist_ok=True)
    weight_dtype = torch.float32
    melody_emb_extractor = MelodyEncoder().to("cuda").float()
    chord_extractor = Chord_extractor().to("cuda").float()
    
    if config["checkpoint_path"]:
        config["transformer_ckpt"] = os.path.join(config["checkpoint_path"], "attn_procs.safetensors")
        config["melody_emb_ckpt"] = os.path.join(config["checkpoint_path"], "melody_emb.safetensors")
        config["chord_cnn_ckpt"] = os.path.join(config["checkpoint_path"], "chord_cnn.safetensors")
    else:
        config["transformer_ckpt"] = None
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
    if config['melody_emb_ckpt'] is not None:
        state_dict = load_file(config['melody_emb_ckpt'])
        # Check keys
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

    pipe = pipe.to("cuda")
    negative_text_prompt = config["negative_text_prompt"]
    # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
    with torch.no_grad():
       
        gt_vocal_audio_file = config['vocal_audio_file']
        prompt_texts = config['text_prompt']
        
        song_name = config['vocal_audio_file'].split('/')[-1].split(".wav")[0]
        
        transformer.eval()            
        waveform_vocal = load_audio_file(gt_vocal_audio_file, segment_starts= 0)
        seconds = waveform_vocal.shape[1] / 44100
        number_of_segments = int(np.floor(seconds / (2097152 / 44100)))
        seconds_list = [2097152 / 44100 * y for y in range(number_of_segments)]
        seconds_starts = 0
            
        if config["no_text"] is True:
            prompt_texts = ""
        # For single conition, we can utilize the full cross-attention dimension 768, instead of 768/4 in MuseControlLite_inference_on_the_fly_all.py
        if "melody" in config["condition_type"]:
            print("using melody condition")
            melody_condition = compute_melody_v2(gt_vocal_audio_file, segment_starts = seconds_starts)
            melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
            extracted_melody_condition = melody_emb_extractor(melody_condition)
            masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
            extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
            masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
        else: 
            extracted_melody_condition = torch.zeros((1, 256, 1024), device="cuda")
            masked_extracted_melody_condition = extracted_melody_condition
        if "chord" in config["condition_type"]:
            print("using chord condition")
            chord_condition, end_time = extract_chords_lab(config['chord_file'], segment_starts = seconds_starts)
            extracted_chord_condition = chord_extractor(chord_condition)
            masked_extracted_chord_condition = torch.zeros_like(extracted_chord_condition)
        else: 
            chord_condition, end_time = extract_chords_lab(config['chord_file'], segment_starts = seconds_starts)
            extracted_chord_condition = torch.zeros((1, 256, 1024), device="cuda")
            masked_extracted_chord_condition = extracted_chord_condition
        if "rhythm" in config["condition_type"]:
            path = Path(config['vocal_beat_file'])
            with path.open("r", encoding="utf-8") as f:
                beat_times_all = [float(line.strip()) for i, line in enumerate(f) if i != 0 and line.strip()] 
            beat_times = sublist_between(beat_times_all, seconds_starts, 2097152/44100 + seconds_starts)
            beat_times = [x - seconds_starts for x in beat_times]
            downbeat_times = []
            print("using rhythm condition")
            rhythm_condition = create_activations_from_timestamps(beat_times, downbeat_times)
            extracted_rhythm_condition = torch.from_numpy(rhythm_condition).cuda().unsqueeze(0).repeat_interleave(256//2, dim=1).float()
            masked_extracted_rhythm_condition = torch.zeros_like(extracted_rhythm_condition)
            extracted_rhythm_condition = F.interpolate(extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
            masked_extracted_rhythm_condition = F.interpolate(masked_extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
        else: 
            extracted_rhythm_condition = torch.zeros((1, 256, 1024), device="cuda")
            masked_extracted_rhythm_condition = extracted_rhythm_condition
        
        # Use multiple cfg
        extracted_condition = torch.concat((extracted_rhythm_condition, extracted_melody_condition, extracted_chord_condition), dim=1)
        masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_melody_condition, masked_extracted_chord_condition), dim=1)
        extracted_condition = torch.concat((masked_extracted_condition, masked_extracted_condition, extracted_condition), dim=0)
        extracted_condition = extracted_condition.transpose(1, 2)
        waveform = pipe(
            extracted_condition = extracted_condition, 
            prompt=prompt_texts,
            negative_prompt=negative_text_prompt,
            num_inference_steps=config["denoise_step"],
            guidance_scale_text=config["guidance_scale_text"],
            guidance_scale_con=config["guidance_scale_con"],
            num_waveforms_per_prompt=1,
            audio_end_in_s=2097152 / 44100,
            generator=generator,
        ).audios                 
        backing_audio = waveform[0].float().cpu()
        waveform_vocal_slice = waveform_vocal[:, int(seconds_starts*44100): int((seconds_starts + 2097152 / 44100)*44100)]
        # ---------------- Example usage ----------------
        # Assume you start with int16 PCM and want float32 in [-1, 1]:
        # wav_a_int16, wav_b_int16: torch.int16 tensors shaped [T] or [C, T]
        waveform_vocal_slice = (waveform_vocal_slice.to(torch.float32) / 32768.0).clamp(-1, 1)
        backing_audio = (backing_audio.to(torch.float32) / 32768.0).clamp(-1, 1)

        mix = mix_audio(waveform_vocal_slice, backing_audio, target_dbfs=-18.0, out_peak_dbfs=-1.0)
        mixed_file_path = os.path.join(output_dir, f"mixed_{song_name}_{prompt_texts}.wav")
        sf.write(mixed_file_path, mix.T.float().cpu().numpy(), pipe.vae.sampling_rate)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocal_audio_file", required=True, help="Path(s) to input audio file(s)")
    parser.add_argument("--text_prompt", required=True, help="Text prompt(s) for generation")
    parser.add_argument("--vocal_beat_file", required=True, help="Path(s) to input vocal beat file(s)")
    parser.add_argument("--chord_file", required=True, help="Path(s) to input chord file(s)")
    parser.add_argument("--checkpoint_path", required=True, help="Path(s) to input checkpoint file(s)")
    parser.add_argument("--output_dir", required=True, help="Path(s) for output directory")
    args = parser.parse_args()

    config = get_config()
    config["vocal_audio_file"] = args.vocal_audio_file
    config["text_prompt"] = args.text_prompt
    config['chord_file'] = args.chord_file
    config['vocal_beat_file'] = args.vocal_beat_file
    config['checkpoint_path'] = args.checkpoint_path
    config['output_dir'] = args.output_dir
    main(config)

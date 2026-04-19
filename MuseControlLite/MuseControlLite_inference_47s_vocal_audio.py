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
from utils.audio_processing import mix_audio, load_audio_file
import random
import re

_MUSECTRLLITE_DIR = os.path.dirname(os.path.abspath(__file__))


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
    
    if config["checkpoint_path"]:
        config["transformer_ckpt"] = os.path.join(config["checkpoint_path"], "attn_procs.safetensors")
       
    else:
        config["transformer_ckpt"] = None
    
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
    #     if config["transformer_ckpt"] is not None:
    #         if "bin" in config["transformer_ckpt"]:
    #             state_dict = torch.load(config["transformer_ckpt"])
    #         elif "safetensors" in config["transformer_ckpt"]:
    #             state_dict = load_file(config["transformer_ckpt"], device="cuda")
    #             keys = list(state_dict.keys())

    #         for name, processor in attn_procs.items():
    #             if isinstance(processor, attn_processor):
    #                 if 'echo' in config["attn_processor_type"]:
    #                     weight_name_proj_gamma = name + ".proj_gamma.weight"
    #                     weight_name_proj_beta = name + ".proj_beta.weight"
    #                     weight_name_hidden_proj = name + ".hidden_proj.weight"
    #                     weight_name_con_proj = name + ".con_proj.weight"
    #                     processor.proj_gamma.weight = torch.nn.Parameter(state_dict[weight_name_proj_gamma].to(torch.float32))
    #                     processor.proj_beta.weight = torch.nn.Parameter(state_dict[weight_name_proj_beta].to(torch.float32))
    #                     processor.hidden_proj.weight = torch.nn.Parameter(state_dict[weight_name_hidden_proj].to(torch.float32))
    #                     processor.con_proj.weight = torch.nn.Parameter(state_dict[weight_name_con_proj].to(torch.float32))
    #                     print(f"load {name}")
    #                 if "rotary" in config["attn_processor_type"]:
    #                     weight_name_v = name + ".to_v_ip.weight"
    #                     weight_name_k = name + ".to_k_ip.weight"
    #                     conv_out_weight = name + ".conv_out.weight"
    #                     processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].to(torch.float32))
    #                     processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].to(torch.float32))
    #                     processor.conv_out.weight = torch.nn.Parameter(state_dict[conv_out_weight].to(torch.float32))
    #                     print(f"load {name}")
    #     transformer.set_attn_processor(attn_procs)
    #     class _Wrapper(AttnProcsLayers):
    #         def forward(self, *args, **kwargs):
    #             return pipe.transformer(*args, **kwargs)
    #     transformer = _Wrapper(pipe.transformer.attn_processors)
    # else:
    #     from diffusers import StableAudioPipeline
    #     pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
    #     pipe.scheduler.config.sigma_max = config["sigma_max"]
    #     pipe.scheduler.config.sigma_min = config["sigma_min"]
    pipe = pipe.to("cuda")
    
    negative_text_prompt = config["negative_text_prompt"]
    # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
    with torch.no_grad():
       
        gt_vocal_audio_file = config['vocal_audio_file']
        prompt_texts = _normalize_prompt_list(config['text_prompt'])
        
        song_name = config['vocal_audio_file'].split('/')[-1].split(".wav")[0]
        
        transformer.eval()            
        waveform_vocal = load_audio_file(gt_vocal_audio_file, segment_starts= 0)
        print("waveform_vocal", waveform_vocal.shape)
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
        
        if "vocal_audio" in config["condition_type"]:
            desired_repeats = 768 // 64  # Number of repeats needed
            audio_condition = pipe.vae.encode(waveform_vocal.unsqueeze(0).to(weight_dtype).cuda()).latent_dist.sample()
            extracted_audio_condition = audio_condition.repeat_interleave(desired_repeats, dim=1).float()
            masked_extracted_audio_condition = torch.zeros_like(extracted_audio_condition)
        else: 
            extracted_audio_condition = torch.zeros((1, 192, 1024), device="cuda")
            masked_extracted_audio_condition = extracted_audio_condition
        extracted_condition = torch.concat((masked_extracted_audio_condition, masked_extracted_audio_condition, extracted_audio_condition), dim=0)
        extracted_condition = extracted_condition.transpose(1, 2)
        target_samples = 2097152
        waveform_vocal_slice = waveform_vocal[:, :target_samples].to(torch.float32).clamp(-1, 1)
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
            backing_audio = waveform[0].float().cpu().clamp(-1, 1)

            mix = mix_audio(waveform_vocal_slice, backing_audio, target_dbfs=-18.0, out_peak_dbfs=-1.0)
            prompt_label = _safe_filename_component(prompt_text)
            mixed_file_path = os.path.join(
                output_dir,
                f"mixed_{song_name}_prompt_{prompt_index:02d}_{prompt_label}.wav",
            )
            sf.write(mixed_file_path, mix.T.float().cpu().numpy(), pipe.vae.sampling_rate)

            

if __name__ == "__main__":
    config = get_config() 
    main(config)

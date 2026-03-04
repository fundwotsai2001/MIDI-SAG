def get_config():
    return {
        "condition_type": ["rhythm", "melody", "chord"], # options: "dynamics", "rhythm", "melody", "audio"

        "output_dir": "./output/MuseControlLite_mixed",

        "checkpoint_path": "./MIDI-SAG_checkpoints/checkpoint_vocal_beat",

        "GPU_id": "0",

        "attn_processor_type": "rotary", # Currently no other available.

        "apadapter": True, # True for MuseControlLite, False for original Stable-audio

        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.5, # The separated guidance for both Musical attribute and audio conditions. Note that if guidance scale is too large, the audio quality will be bad. Values between 0.5~2.0 is recommended.
                
        "denoise_step": 100,

        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.

        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad.

        "weight_dtype": "fp32", # fp16 and fp32 sounds quiet the same.

        "negative_text_prompt": "noisy, distortion, bad quality",

        "no_text": False, # Optional, set to true if no text prompt is needed (possible for audio inpainting or outpainting)
    }
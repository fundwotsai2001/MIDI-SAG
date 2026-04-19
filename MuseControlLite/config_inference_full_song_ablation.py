def get_config():
    return {
        "condition_type": ["structure", "rhythm", "audio", "chord", "melody"], # options: "rhythm", "melody", "structure", "chord", "audio"

        "output_dir": "/data/home/fundwotsai/MIDI-SAG/demo_website/lyrics2song/",
        "midi_folder": "/data/home/fundwotsai/MIDI-SAG/generated_midi_right_key",

        # Checkpoints (adapters and extractors): You can choose any combinations you like. 
        ###############
        "checkpoint_path": "/data/home/fundwotsai/MIDI-SAG/MuseControlLite/checkpoint-65000",

        ###############

        "GPU_id": "0",

        "attn_processor_type": "rotary_free", # Currently no other available.

        "apadapter": True, # True for MuseControlLite, False for original Stable-audio

        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.5, # The separated guidance for both Musical attribute and audio conditions. Note that if guidance scale is too large, the audio quality will be bad. Values between 0.5~2.0 is recommended.
        
        "guidance_scale_audio": 1.0,
        
        "denoise_step": 50,

        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.

        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad.

        "weight_dtype": "fp32", # fp16 and fp32 sounds quiet the same.

        "negative_text_prompt": "noisy, bad quality",

        # The below two mask should complementary, which means every time slice shouldn't receive both audio and music attribute condition.
        # Don't set both use_audio_mask and use_musical_attribute_mask to True.

        ###############
        "use_audio_mask": False,

        "audio_mask_start_seconds": 24,

        "audio_mask_end_seconds": 2097152 / 44100, # Maximum duration for stable-audio is 2097152 / 44100 seconds

        "use_musical_attribute_mask": False,

        "musical_attribute_mask_start_seconds": 0,

        "musical_attribute_mask_end_seconds": 0,
        ###############

        "no_text": False, # Optional, set to true if no text prompt is needed (possible for audio inpainting or outpainting)

        "text_file": "/data/home/fundwotsai/MIDI-SAG/MuseControlLite/prompts.json",

        # Optional: list of song IDs to process. If empty or None, all songs are processed.
        # Example: [40, 131, 49, 126, 56 ,106, 78, 83]
        "id_filter": [25,26,29,31,36,45,78,124],

        "dataset_folder": "/data/home/fundwotsai/MIDI-SAG/dataset",
    }
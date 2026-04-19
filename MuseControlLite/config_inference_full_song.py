def get_config():
    return {
        "condition_type": [ "rhythm", "melody", "structure", "chord", "audio"], # options: "rhythm", "melody", "structure", "chord", "audio"

        "output_dir": "/data/home/fundwotsai/MIDI-SAG/demo_website/chord_variation/chord_v1_text_v2/", # output directory for MuseControlLite results

        "structure_prompts": [
            "nylon guitar arpeggio, soft shaker, room reverb; warm and close.",
            "fingerpicked guitar + upright bass, brushed snare; gentle sway.",
            "trummed guitars, tambourine, simple piano chords; bright, earthy.",
            "trummed guitars, tambourine, simple piano chords; bright, earthy.",
            "Instruments peel back to single guitar; natural ring-out.",
        ],

        "structure_tags": ["intro", "verse", "chorus", "chorus", "outro"],

        "structure_starts": [0, 9.558, 28.673, 54.691, 80.708], 

        "vocal_audio_file": "/data/home/fundwotsai/MIDI-SAG/Chord_progression_editability_demo/sample02.wav",

        # "vocal_midi_file": "/data/home/fundwotsai/MIDI-SAG/Chord_progression_editability_demo/sample01.mid",

        "vocal_beat_file": None,

        "chord_info": "/data/home/fundwotsai/MIDI-SAG/Chord_progression_editability_demo/chord_from_musician_refine/1_113.txt",
        

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

        "evaluate_chord_rhythm": False,

        "text_file": "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite_song_generation/prompts_no_vocal.json",

        # Optional: list of song IDs to process. If empty or None, all songs are processed.
        # Example: [40, 131, 49, 126, 56 ,106, 78, 83]
        "id_filter": [],

        "dataset_folder": "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite_song_generation/dataset",
    }
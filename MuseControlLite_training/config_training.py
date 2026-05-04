def get_config():
    return {
        # Load files and checkpoints

        "condition_type": ["chord", "rhythm", "melody", "audio", "structure"], #"melody", "rhythm", "dynamics", "audio"

        "meta_data_path": "/volume/ai-music-data-storage/discog_shs_cpop_filtered_15k_no_silence_all_f0.json",

        "output_dir": "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite/checkpoints/MuseControlLite_SAG_fp16_text_structure",

        "checkpoint_path": None, #"/volume/nas-fundwo-storage/fundwo-test/MuseControlLite/checkpoints/MuseControlLite_SAG_fp16/checkpoint-16000",

        "transformer_pretrain_ckpt": None, #"/volume/nas-fundwo-storage/fundwo-test/MuseControlLite/checkpoints/stable-audio-multi-text-fintune-167000/checkpoint-228000/transformer.safetensors",

        # Set to a checkpoint dir to resume training, e.g. ".../checkpoint-5000". Leave None to start fresh.
        # "resume_from_checkpoint": "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite/checkpoints/stable-audio-multi-text-fintune-70000/checkpoint-167000",

        "wand_run_name": "MuseControlLite_SAG_multi_text",

        # training hyperparameters
        "GPU_id" : "0,1,2,3,4,5",

        "train_batch_size": 4,

        "learning_rate": 5e-5,

        "attn_processor_type": "rotary_free", # "rotary", "rotary_conv_in", "absolute", "echo"

        "gradient_accumulation_steps": 4,

        "max_train_steps": 10000000,

        "num_train_epochs": 20,

        "dataloader_num_workers": 8,

        "mixed_precision": "fp16", #["no", "fp16", "bf16"]

        "apadapter": True,

        "lr_scheduler": "linear", # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'

        "weight_decay": 1e-2,

        #config for validation
        "validation_num": 2000,

        "test_num": 5,

        "ap_scale": 1.0,

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.5, # The separated guidance for both Musical attribute and audio conditions. Note that if guidance scale is too large, the audio quality will be bad. Values between 0.5~2.0 is recommended.

        "checkpointing_steps": 1000,

        "validation_steps": 1000,

        "denoise_step": 50,

        "log_first": True,

        "sigma_min": 0.3,

        "sigma_max": 500,

        "train_self": True,
    }

# {"path": "/volume/ai-music-data-storage/discog-vi_allin1_result_slices_47s/-/-zxM1DeAQUs/-zxM1DeAQUs/verse_1.33_21.93.opus", "bpm": 94, "captions": ["ambient, calming, dreamy, ethereal, synthesizer pad, soft electronic percussion", "ambient, calming, relaxing, soft, synth pad, electric guitar"], "beats": [0.0, 0.65, 1.29, 1.94, 2.58, 3.23, 3.87, 4.51, 5.16, 5.8, 6.44, 7.09, 7.73, 8.38, 9.02, 9.67, 10.31, 10.95, 11.59, 12.24, 12.88, 13.53, 14.17, 14.82, 15.46, 16.11, 16.74, 17.39, 18.04, 18.68, 19.32, 19.97, 20.61, 21.26, 21.9, 22.55, 23.19, 23.83, 24.47, 25.12, 25.76, 26.41, 27.05, 27.7, 28.34, 28.99, 29.63, 30.28, 30.92, 31.56, 32.2, 32.85, 33.49, 34.14, 34.78, 35.43, 36.07, 36.72, 37.36, 38.01, 38.65, 39.29, 39.93, 40.58, 41.22, 41.87, 42.51, 43.16, 43.8, 44.44, 45.08, 45.73, 46.37, 47.02], "downbeats": [0.0, 2.58, 5.16, 7.73, 10.31, 12.88, 15.46, 18.04, 20.61, 23.19, 25.76, 28.34, 30.92, 33.49, 36.07, 38.65, 41.22, 43.8, 46.37], "structure_starts_seconds": [0.0, 20.6], "structures": ["verse", "verse"]},

import torch
import soundfile as sf
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
)
import torch.nn.functional as F
from safetensors.torch import load_file
import os
import re
import json
import subprocess
import numpy as np
from config_inference import get_config
import argparse
from utils.extract_conditions import compute_melody_v2, create_activations_from_timestamps, calculate_beats_and_downbeats, evaluate_f1_rhythm
from utils.condition_extractors import MelodyEncoder, Chord_extractor
from utils.audio_processing import mix_audio, extract_chords_lab, sublist_between, load_audio_file
import mido
import random
from pathlib import Path
from BeatNet.BeatNet import BeatNet

# Ground-truth data root (for GT MIDI beats and key JSON only)
DATA_ROOT    = Path(
    "/volume/nas-fundwo-storage/fundwo-test/lyrics2melody"
    "/generated_midi_ICML_rebuttal_pitch_shift_v3"
)
PROMPTS_JSON = Path(
    "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite_song_generation/prompts.json"
)
SEGMENT_LEN  = 2097152   # samples (~47.6 s at 44100 Hz)
SR           = 44100
TOTAL_SONGS  = 200
KEY_CNN_ENV  = "/volume/nas-fundwo-storage/fundwo-test/miniconda3/envs/key-cnn"

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------
_KEY_LINE_FULL      = re.compile(r"^[A-G](?:[#♯b♭])?(?:m)?$")
_KEY_TOKEN_IN_LINE  = re.compile(r"\b([A-G](?:[#♯b♭])?(?:m)?)\b")
_ENHARMONIC = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
    "C#m": "Dbm", "D#m": "Ebm", "F#m": "Gbm", "G#m": "Abm", "A#m": "Bbm",
}

def _normalise_key(k):
    k = k.replace("♯", "#").replace("♭", "b")
    return _ENHARMONIC.get(k, k)

# Krumhansl-Kessler tonal hierarchy profiles
_KK_MAJOR   = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR   = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def get_key_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    entry = data["processed_files"][0]
    key  = entry["key"]   # e.g. "G#", "C#", "C"  — sharp notation
    mode = entry["mode"]  # "major" or "minor"
    return key if mode == "major" else key + "m"

def compute_key_from_midi(midi_path):
    midi = mido.MidiFile(midi_path)
    pc_duration = np.zeros(12)
    for track in midi.tracks:
        active = {}
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_tick
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    pc_duration[msg.note % 12] += abs_tick - active.pop(msg.note)
    if pc_duration.sum() == 0:
        return "C"
    pcd = pc_duration / pc_duration.sum()
    best_r, best_key = -np.inf, "C"
    for root in range(12):
        r_maj = np.corrcoef(pcd, np.roll(_KK_MAJOR, root))[0, 1]
        r_min = np.corrcoef(pcd, np.roll(_KK_MINOR, root))[0, 1]
        if r_maj > best_r:
            best_r, best_key = r_maj, _NOTE_NAMES[root]
        if r_min > best_r:
            best_r, best_key = r_min, _NOTE_NAMES[root] + "m"
    return best_key

def get_key(audio_path, conda_env=None):
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if conda_env:
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        target = ["-p", conda_env] if os.path.isabs(conda_env) else ["-n", conda_env]
        cmd = [conda_exe, "run", *target, "key", "-i", audio_path]
    else:
        cmd = ["key", "-i", audio_path]
    res  = subprocess.run(cmd, capture_output=True, text=True, env=env)
    text = (res.stdout or "") + (("\n" + res.stderr) if res.stderr else "")
    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s or s.lower() == "done":
            continue
        if _KEY_LINE_FULL.fullmatch(s):
            return s
        m = _KEY_TOKEN_IN_LINE.search(s)
        if m and _KEY_LINE_FULL.fullmatch(m.group(1)):
            return m.group(1)
    raise RuntimeError(f"Could not find a key in the output for {audio_path}:\n{text}")


def main(config, pipeline_output_dir):
    pipeline_output_dir = Path(pipeline_output_dir)

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
        config["melody_emb_ckpt"]  = os.path.join(config["checkpoint_path"], "melody_emb.safetensors")
        config["chord_cnn_ckpt"]   = os.path.join(config["checkpoint_path"], "chord_cnn.safetensors")
    else:
        config["transformer_ckpt"] = None
        config["melody_emb_ckpt"]  = None
        config["chord_cnn_ckpt"]   = None

    if config['chord_cnn_ckpt'] is not None:
        state_dict = load_file(config['chord_cnn_ckpt'])
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[len("module."):] if k.startswith("module.") else k] = v
        chord_extractor.load_state_dict(new_state_dict)
        print("load chord_extractor")
    if config['melody_emb_ckpt'] is not None:
        state_dict = load_file(config['melody_emb_ckpt'])
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[len("module."):] if k.startswith("module.") else k] = v
        melody_emb_extractor.load_state_dict(new_state_dict)
        print("load melody_emb_extractor")

    if config["weight_dtype"] == "fp16":
        weight_dtype = torch.float16
    elif config["weight_dtype"] == "bp16":
        weight_dtype = torch.bfloat16

    if config["apadapter"]:
        from pipeline.stable_audio_multi_cfg_pipe import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained("/volume/nas-fundwo-storage/fundwo-test/MuseControlLite/stable-audio", torch_dtype=weight_dtype)
        pipe.scheduler.config.sigma_max = config["sigma_max"]
        pipe.scheduler.config.sigma_min = config["sigma_min"]
        transformer = pipe.transformer
        attn_procs = {}
        processor_classes = {
            "rotary": StableAudioAttnProcessor2_0_rotary,
        }
        attn_processor = processor_classes.get(config["attn_processor_type"], None)
        for name in transformer.attn_processors.keys():
            if name.endswith("attn1.processor"):
                attn_procs[name] = StableAudioAttnProcessor2_0()
            else:
                attn_procs[name] = attn_processor(
                    layer_id=name.split(".")[1],
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
                    if "rotary" in config["attn_processor_type"]:
                        processor.to_v_ip.weight  = torch.nn.Parameter(state_dict[name + ".to_v_ip.weight"].to(torch.float32))
                        processor.to_k_ip.weight  = torch.nn.Parameter(state_dict[name + ".to_k_ip.weight"].to(torch.float32))
                        processor.conv_out.weight = torch.nn.Parameter(state_dict[name + ".conv_out.weight"].to(torch.float32))
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

    # ---- Load per-song prompts ----
    with open(PROMPTS_JSON) as f:
        prompts_list = json.load(f)
    id_to_prompt = {
        int(Path(entry["json_file"]).stem): entry["global_prompt"]
        for entry in prompts_list
    }

    # ---- BeatNet (instantiate once) ----
    beatnet = BeatNet(1, mode="offline", inference_model="DBN",
                      plot=[], thread=False, device="cuda")

    rhythm_f1_scores = []
    key_matches      = []
    results_per_song = []

    with torch.no_grad():
        transformer.eval()

        for song_id in range(TOTAL_SONGS):
            song_name = f"lyrics_{song_id}"

            # ---- Detected inputs (from pipeline output of evaluate_200_songs.sh) ----
            vocal_wav  = pipeline_output_dir / "vad_audio" / f"{song_name}.wav"
            midi_file  = str(pipeline_output_dir / "vocal_MIDI" / f"{song_name}.mid")
            chord_file = str(pipeline_output_dir / "Harmonization_results" / "btc_txt"
                             / f"{song_name}_chord_gen.txt")

            # ---- GT data (evaluation only — not fed to the model) ----
            gt_song_dir    = DATA_ROOT / song_name
            gt_midi_file   = str(gt_song_dir / "sample01_shifted.mid")
            proc_json_file = gt_song_dir / "chord" / "processing_results.json"

            # Fall back to original vocal if VAD output is missing
            if not vocal_wav.exists():
                vocal_wav = gt_song_dir / "sample01_shifted.wav"
                if not vocal_wav.exists():
                    print(f"[skip] {song_name}: vocal not found")
                    continue

            if not Path(midi_file).exists():
                print(f"[skip] {song_name}: detected MIDI not found ({midi_file})")
                continue
            if not Path(chord_file).exists():
                print(f"[skip] {song_name}: detected chord file not found ({chord_file})")
                continue
            if not Path(gt_midi_file).exists():
                print(f"[skip] {song_name}: GT MIDI not found")
                continue
            if not proc_json_file.exists():
                print(f"[skip] {song_name}: processing_results.json not found")
                continue

            prompt_texts = id_to_prompt.get(song_id, "")
            if config["no_text"] is True:
                prompt_texts = ""
            print(f"\n[{song_id+1}/{TOTAL_SONGS}] {song_name} | prompt: {prompt_texts[:80]}")

            try:
                seconds_starts = 0

                # ---- Melody condition (detected VAD vocal) ----
                if "melody" in config["condition_type"]:
                    print("using melody condition")
                    melody_condition = compute_melody_v2(str(vocal_wav), segment_starts=seconds_starts)
                    melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
                    extracted_melody_condition = melody_emb_extractor(melody_condition)
                    masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
                    extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                    masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                else:
                    extracted_melody_condition = torch.zeros((1, 256, 1024), device="cuda")
                    masked_extracted_melody_condition = extracted_melody_condition

                # ---- Chord condition (detected chord from AccoMontage2) ----
                if "chord" in config["condition_type"]:
                    print("using chord condition")
                    chord_condition, end_time = extract_chords_lab(chord_file, segment_starts=seconds_starts)
                    extracted_chord_condition = chord_extractor(chord_condition)
                    masked_extracted_chord_condition = torch.zeros_like(extracted_chord_condition)
                else:
                    chord_condition, end_time = extract_chords_lab(chord_file, segment_starts=seconds_starts)
                    extracted_chord_condition = torch.zeros((1, 256, 1024), device="cuda")
                    masked_extracted_chord_condition = extracted_chord_condition

                # ---- Rhythm condition (pre-computed beat times from vocal_beat) ----
                if "rhythm" in config["condition_type"]:
                    print("using rhythm condition")
                    beat_txt = pipeline_output_dir / "vocal_beat" / song_name / "sample01_shifted_beat_times.txt"
                    with open(beat_txt) as _bf:
                        beat_times_all = [float(line.strip()) for line in _bf if line.strip() and not line.startswith("#")]
                    beat_times = sublist_between(beat_times_all, seconds_starts, SEGMENT_LEN / SR + seconds_starts)
                    beat_times = [x - seconds_starts for x in beat_times]
                    downbeat_times = []
                    rhythm_condition = create_activations_from_timestamps(beat_times, downbeat_times)
                    extracted_rhythm_condition = torch.from_numpy(rhythm_condition).cuda().unsqueeze(0).repeat_interleave(256 // 2, dim=1).float()
                    masked_extracted_rhythm_condition = torch.zeros_like(extracted_rhythm_condition)
                    extracted_rhythm_condition = F.interpolate(extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
                    masked_extracted_rhythm_condition = F.interpolate(masked_extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
                else:
                    extracted_rhythm_condition = torch.zeros((1, 256, 1024), device="cuda")
                    masked_extracted_rhythm_condition = extracted_rhythm_condition

                # ---- Stack for triple-CFG ----
                extracted_condition = torch.concat((extracted_rhythm_condition, extracted_melody_condition, extracted_chord_condition), dim=1)
                masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_melody_condition, masked_extracted_chord_condition), dim=1)
                extracted_condition = torch.concat((masked_extracted_condition, masked_extracted_condition, extracted_condition), dim=0)
                extracted_condition = extracted_condition.transpose(1, 2)

                # ---- Generate backing ----
                waveform = pipe(
                    extracted_condition=extracted_condition,
                    prompt=prompt_texts,
                    negative_prompt=negative_text_prompt,
                    num_inference_steps=config["denoise_step"],
                    guidance_scale_text=config["guidance_scale_text"],
                    guidance_scale_con=config["guidance_scale_con"],
                    num_waveforms_per_prompt=1,
                    audio_end_in_s=SEGMENT_LEN / SR,
                    generator=generator,
                ).audios

                backing_audio = waveform[0].float().cpu()

                # Vocal from load_audio_file is int16-range PCM → normalise to [-1, 1]
                waveform_vocal = load_audio_file(str(vocal_wav), segment_starts=0)
                vocal_slice    = waveform_vocal[:, int(seconds_starts * SR): int((seconds_starts + SEGMENT_LEN / SR) * SR)]
                vocal_slice    = (vocal_slice.to(torch.float32) / 32768.0).clamp(-1, 1)
                backing_norm   = backing_audio.clamp(-1, 1)

                # Save generated backing (unmixed)
                backing_path = os.path.join(output_dir, f"backing_{song_id}.wav")
                sf.write(backing_path, backing_norm.T.numpy(), pipe.vae.sampling_rate)

                # Save mixed
                mix = mix_audio(vocal_slice, backing_norm, target_dbfs=-18.0, out_peak_dbfs=-1.0)
                mixed_path = os.path.join(output_dir, f"mixed_{song_id}.wav")
                sf.write(mixed_path, mix.T.float().cpu().numpy(), pipe.vae.sampling_rate)
                print(f"  Saved: {backing_path}")

                # ---- Rhythm F1 (GT from ground-truth MIDI) ----
                seg_dur = SEGMENT_LEN / SR  # ~47.55 s
                beat_times_gt_eval, _ = calculate_beats_and_downbeats(gt_midi_file)
                beat_times_gt_eval = [t for t in beat_times_gt_eval if 0.0 <= t <= seg_dur]
                gt_timestamps = np.array(beat_times_gt_eval)

                gen_timestamps_raw = beatnet.process(backing_path)
                gen_timestamps = gen_timestamps_raw[
                    (gen_timestamps_raw[:, 0] >= 0.0) & (gen_timestamps_raw[:, 0] <= seg_dur)
                ]
                print("gt_timestamps", gt_timestamps)
                print("gen_timestamps", gen_timestamps)
                _, _, f1 = evaluate_f1_rhythm(gt_timestamps, gen_timestamps)
                rhythm_f1_scores.append(f1)
                print(f"  Rhythm F1: {f1:.4f}  (GT beats: {len(gt_timestamps)}, Gen beats: {len(gen_timestamps)})")

                # ---- Key accuracy (GT from processing_results.json) ----
                gt_key  = get_key_from_json(str(proc_json_file))
                gen_key = get_key(backing_path, conda_env=KEY_CNN_ENV)
                match   = int(_normalise_key(gt_key) == _normalise_key(gen_key))
                key_matches.append(match)
                print(f"  Key  GT={gt_key}  Gen={gen_key}  match={bool(match)}")

                results_per_song.append({
                    "song_id":        song_id,
                    "rhythm_f1":      f1,
                    "gt_key":         gt_key,
                    "gen_key":        gen_key,
                    "key_match":      bool(match),
                    "detected_midi":  midi_file,
                    "detected_chord": chord_file,
                    "vocal_wav":      str(vocal_wav),
                })

            except Exception as exc:
                print(f"  ERROR {song_name}: {exc}")
                results_per_song.append({"song_id": song_id, "error": str(exc)})


    # ---- Aggregate results ----
    mean_f1 = float(np.mean(rhythm_f1_scores)) if rhythm_f1_scores else float("nan")
    key_acc = float(np.mean(key_matches))       if key_matches      else float("nan")

    print("\n" + "=" * 60)
    print(f"Songs evaluated : {len(rhythm_f1_scores)}")
    print(f"Mean Rhythm F1  : {mean_f1:.4f}")
    print(f"Key Accuracy    : {key_acc:.4f}")
    print("=" * 60)

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "mean_rhythm_f1":      mean_f1,
                "key_accuracy":        key_acc,
                "n_evaluated":         len(rhythm_f1_scores),
                "pipeline_output_dir": str(pipeline_output_dir),
                "per_song":            results_per_song,
            },
            f, indent=2,
        )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and evaluate backing tracks using detected MIDI/chord from the pipeline."
    )
    parser.add_argument("--gpu_id",              default="0",
                        help="CUDA device id (default: 0)")
    parser.add_argument("--no_text",             action="store_true",
                        help="Disable text conditioning")
    parser.add_argument("--pipeline_output_dir", default="../output_evaluate_200",
                        help="Output dir of evaluate_200_songs.sh (default: ./output_evaluate_200)")
    args = parser.parse_args()

    config = get_config()
    config["GPU_id"]      = args.gpu_id
    config["no_text"]     = args.no_text
    config["text_prompt"] = ""
    main(config, args.pipeline_output_dir)

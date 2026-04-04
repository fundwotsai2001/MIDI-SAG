#!/usr/bin/env python3
"""
End-to-end evaluation script for MuseControlLite SAG model on the lyrics2melody dataset.

For each of 200 songs:
  1. Generates a backing track from the first 47s of the vocal audio.
  2. Computes rhythm F1  – GT beats from MIDI vs BeatNet on generated backing.
  3. Computes key accuracy – GT key from vocal audio vs key of generated backing.
"""

import os
import re
import sys
import json
import subprocess
import argparse
from pathlib import Path

import mido
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

from diffusers.loaders import AttnProcsLayers
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
)
from config_inference import get_config
from utils.extract_conditions import (
    compute_melody_v2,
    create_activations_from_timestamps,
    calculate_beats_and_downbeats,
    evaluate_f1_rhythm,
)
from utils.audio_processing import mix_audio, extract_chords_lab, load_audio_file, sublist_between
from utils.condition_extractors import MelodyEncoder, Chord_extractor
from BeatNet.BeatNet import BeatNet

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------
KEY_LINE_FULL   = re.compile(r"^[A-G](?:[#♯b♭])?(?:m)?$")
KEY_TOKEN_IN_LINE = re.compile(r"\b([A-G](?:[#♯b♭])?(?:m)?)\b")

# Map sharps → flats so enharmonic keys compare equal
_ENHARMONIC = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
    "C#m": "Dbm", "D#m": "Ebm", "F#m": "Gbm", "G#m": "Abm", "A#m": "Bbm",
}

def _normalise_key(k: str) -> str:
    k = k.replace("♯", "#").replace("♭", "b")
    return _ENHARMONIC.get(k, k)


def get_key_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    entry = data["processed_files"][0]
    key  = entry["key"]   # e.g. "G#", "C#", "C"  — sharp notation
    mode = entry["mode"]  # "major" or "minor"
    return key if mode == "major" else key + "m"


# Krumhansl-Kessler tonal hierarchy profiles
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

def compute_key_from_midi(midi_path: str) -> str:
    """
    Derive the musical key from MIDI note content using the
    Krumhansl-Schmuckler algorithm (pitch-class duration histogram
    correlated against major/minor tonal profiles).

    Returns a string like 'C', 'F#', 'Am', 'C#m' that matches the
    format returned by get_key().
    """
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat

    # Build pitch-class duration histogram (in ticks)
    pc_duration = np.zeros(12)
    for track in midi.tracks:
        active: dict[int, int] = {}   # pitch → abs_tick of note_on
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_tick
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    duration = abs_tick - active.pop(msg.note)
                    pc_duration[msg.note % 12] += duration

    if pc_duration.sum() == 0:
        return "C"   # no notes found

    # Normalise histogram
    pcd = pc_duration / pc_duration.sum()

    # Correlate against all 24 keys (12 major + 12 minor)
    best_r, best_key = -np.inf, "C"
    for root in range(12):
        profile_maj = np.roll(_KK_MAJOR, root)
        profile_min = np.roll(_KK_MINOR, root)
        r_maj = np.corrcoef(pcd, profile_maj)[0, 1]
        r_min = np.corrcoef(pcd, profile_min)[0, 1]
        if r_maj > best_r:
            best_r, best_key = r_maj, _NOTE_NAMES[root]
        if r_min > best_r:
            best_r, best_key = r_min, _NOTE_NAMES[root] + "m"

    return best_key


def get_key(audio_path: str, conda_env: str | None = None) -> str:
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    if conda_env:
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        target = ["-p", conda_env] if os.path.isabs(conda_env) else ["-n", conda_env]
        cmd = [conda_exe, "run", *target, "key", "-i", audio_path]
    else:
        cmd = ["key", "-i", audio_path]

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    text = (res.stdout or "") + (("\n" + res.stderr) if res.stderr else "")

    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s or s.lower() == "done":
            continue
        if KEY_LINE_FULL.fullmatch(s):
            return s
        m = KEY_TOKEN_IN_LINE.search(s)
        if m and KEY_LINE_FULL.fullmatch(m.group(1)):
            return m.group(1)

    raise RuntimeError(f"Could not find a key in the output for {audio_path}:\n{text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DATA_ROOT   = Path(
    "/volume/nas-fundwo-storage/fundwo-test/lyrics2melody"
    "/generated_midi_ICML_rebuttal_pitch_shift_v3"
)
PROMPTS_JSON = Path(
    "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite_song_generation/prompts.json"
)
KEY_CNN_ENV = "/volume/nas-fundwo-storage/fundwo-test/miniconda3/envs/key-cnn"
SEGMENT_LEN = 2097152          # samples  (~47.6 s at 44100 Hz)
SR          = 44100
TOTAL_SONGS = 200


def build_model(config, weight_dtype):
    from pipeline.stable_audio_pipeline import StableAudioPipeline

    pipe = StableAudioPipeline.from_pretrained(
        "/volume/nas-fundwo-storage/fundwo-test/MuseControlLite/stable-audio", torch_dtype=weight_dtype
    )
    pipe.scheduler.config.sigma_max = config["sigma_max"]
    pipe.scheduler.config.sigma_min = config["sigma_min"]

    transformer = pipe.transformer
    
    return pipe, transformer




def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU_id"]
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = {"fp16": torch.float16, "bp16": torch.bfloat16}.get(
        config["weight_dtype"], torch.float32
    )

    # ---- Load auxiliary encoders ----
    melody_emb_extractor = MelodyEncoder().to("cuda").float()
    chord_extractor      = Chord_extractor().to("cuda").float()

    if config["checkpoint_path"]:
        config["transformer_ckpt"]  = os.path.join(config["checkpoint_path"], "attn_procs.safetensors")
    else:
        config["transformer_ckpt"] = config["melody_emb_ckpt"] = config["chord_cnn_ckpt"] = None


    # ---- Load per-song prompts ----
    with open(PROMPTS_JSON) as f:
        prompts_list = json.load(f)
    id_to_prompt = {
        int(Path(entry["json_file"]).stem): entry["global_prompt"]
        for entry in prompts_list
    }

    # ---- Build diffusion pipeline ----
    pipe, transformer = build_model(config, weight_dtype)
    pipe = pipe.to("cuda")
    negative_text = config["negative_text_prompt"]

    # ---- BeatNet (instantiate once) ----
    beatnet = BeatNet(1, mode="offline", inference_model="DBN",
                      plot=[], thread=False, device="cuda")

    # ---- Per-song metrics ----
    rhythm_f1_scores  = []
    key_matches       = []
    results_per_song  = []

    generator = torch.Generator().manual_seed(42)

    with torch.no_grad():
        transformer.eval()

        for song_id in range(TOTAL_SONGS):
            song_dir       = DATA_ROOT / f"lyrics_{song_id}"
            vocal_wav      = song_dir / "sample01_shifted.wav"
            midi_file      = str(song_dir / "sample01_shifted.mid")
            chord_file     = str(song_dir / "chord" / "chord_btc_txt"
                             / "sample01_shifted_chord_gen_filled_empty_bars.txt")
            proc_json_file = song_dir / "chord" / "processing_results.json"

            if not vocal_wav.exists():
                print(f"[skip] lyrics_{song_id}: vocal not found")
                continue
            if not Path(midi_file).exists():
                print(f"[skip] lyrics_{song_id}: MIDI not found")
                continue
            if not Path(chord_file).exists():
                print(f"[skip] lyrics_{song_id}: chord file not found")
                continue
            if not proc_json_file.exists():
                print(f"[skip] lyrics_{song_id}: processing_results.json not found")
                continue

            prompt_texts = id_to_prompt.get(song_id, "")
            if config["no_text"]:
                prompt_texts = ""
            print(f"\n[{song_id+1}/{TOTAL_SONGS}] lyrics_{song_id} | prompt: {prompt_texts[:80]}")

            try:
                # ---- Generate backing ----
                waveform = pipe(
                    prompt=prompt_texts,
                    negative_prompt=negative_text,
                    num_inference_steps=config["denoise_step"],
                    guidance_scale_text=config["guidance_scale_text"],
                    guidance_scale_con=config["guidance_scale_con"],
                    num_waveforms_per_prompt=1,
                    audio_end_in_s=SEGMENT_LEN / SR,
                    generator=generator,
                ).audios

                backing_audio = waveform[0].float().cpu()  # pipeline output: float32 in [-1, 1]

                # Vocal from load_audio_file is int16-range PCM → normalise to [-1, 1]
                waveform_vocal = load_audio_file(str(vocal_wav), segment_starts=0)
                vocal_slice    = waveform_vocal[:, :SEGMENT_LEN]
                vocal_slice    = (vocal_slice.to(torch.float32) / 32768.0).clamp(-1, 1)
                backing_norm   = backing_audio.clamp(-1, 1)

                # Save generated backing (unmixed)
                backing_path = str(output_dir / f"backing_{song_id}.wav")
                sf.write(backing_path, backing_norm.T.numpy(), pipe.vae.sampling_rate)

                # Save mixed
                mix = mix_audio(vocal_slice, backing_norm, target_dbfs=-18.0, out_peak_dbfs=-1.0)
                mixed_path = str(output_dir / f"mixed_{song_id}.wav")
                sf.write(mixed_path, mix.T.float().numpy(), pipe.vae.sampling_rate)

                # ---- Rhythm F1 ----
                beat_times_gt, _ = calculate_beats_and_downbeats(midi_file)
                seg_dur = SEGMENT_LEN / SR  # ~47.55 s
                # Restrict GT beats to [0, seg_dur]
                beat_times_gt = [t for t in beat_times_gt if 0.0 <= t <= seg_dur]
                gt_timestamps  = np.array(beat_times_gt)   # 1-D for evaluate_f1_rhythm

                # BeatNet on generated backing; restrict to [0, seg_dur]
                gen_timestamps_raw = beatnet.process(backing_path)  # 2-D: col-0=time, col-1=beat type
                gen_timestamps = gen_timestamps_raw[
                    (gen_timestamps_raw[:, 0] >= 0.0) & (gen_timestamps_raw[:, 0] <= seg_dur)
                ]

                _, _, f1 = evaluate_f1_rhythm(gt_timestamps, gen_timestamps)
                rhythm_f1_scores.append(f1)
                print(f"  Rhythm F1: {f1:.4f}  (GT beats: {len(gt_timestamps)}, Gen beats: {len(gen_timestamps)})")

                # ---- Key accuracy (GT derived from MIDI note content) ----
                gt_key  = get_key_from_json(str(proc_json_file))
                gen_key = get_key(backing_path, conda_env=KEY_CNN_ENV)
                match   = int(_normalise_key(gt_key) == _normalise_key(gen_key))
                key_matches.append(match)
                print(f"  Key  GT={gt_key}  Gen={gen_key}  match={bool(match)}")

                results_per_song.append({
                    "song_id":   song_id,
                    "rhythm_f1": f1,
                    "gt_key":    gt_key,
                    "gen_key":   gen_key,
                    "key_match": bool(match),
                })

            except Exception as exc:
                print(f"  ERROR: {exc}")
                results_per_song.append({"song_id": song_id, "error": str(exc)})

    # ---- Aggregate results ----
    mean_f1  = float(np.mean(rhythm_f1_scores)) if rhythm_f1_scores else float("nan")
    key_acc  = float(np.mean(key_matches))       if key_matches      else float("nan")

    print("\n" + "="*60)
    print(f"Songs evaluated : {len(rhythm_f1_scores)}")
    print(f"Mean Rhythm F1  : {mean_f1:.4f}")
    print(f"Key Accuracy    : {key_acc:.4f}")
    print("="*60)

    # Save full per-song results
    results_path = str(output_dir / "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "mean_rhythm_f1": mean_f1,
                "key_accuracy":   key_acc,
                "n_evaluated":    len(rhythm_f1_scores),
                "per_song":       results_per_song,
            },
            f, indent=2,
        )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate backing + evaluate rhythm F1 and key accuracy for 200 songs."
    )
    parser.add_argument("--gpu_id",           default="0",    help="CUDA device id (default: 0)")
    parser.add_argument("--no_text",          action="store_true", help="Disable text conditioning")
    args = parser.parse_args()

    config = get_config()
    config["GPU_id"]          = args.gpu_id
    config["no_text"]         = args.no_text
    config["text_prompt"]     = ""   # unused – prompts loaded from prompts.json per song
    main(config)

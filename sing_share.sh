#!/usr/bin/env bash
# Sing the real-region MIDIs (midi_real) with FastSinger WITHOUT any pitch shift.
#
# The melody in midi_real is already in the real song's key/register, so it is
# sung exactly as written. The ONLY pitch-based decision is which singer fits
# the melody best (male -> id 5, female -> id 6); no transposition is applied.
#
#   1. prepare_real_region_jobs.py: per song, score the melody at shift 0 against
#      the male/female profiles, pick the singer, reconstruct the lyric .txt, and
#      emit meta_real_region.json (midi_path -> midi_real, pitch_shifts:[0]).
#   2. Run FastSinger once (single model load) over the whole batch.
#
# Output: $WAV_DIR (singing_voice_real_region), plus lyrics_real_region/,
# meta_pp_real_region/, meta_real_region.json under $OUT_ROOT.
#
# Usage:
#   bash sing_share.sh                                   # all songs, auto singer
#   SINGER=female bash sing_share.sh                     # force a singer
#   LIMIT=3 bash sing_share.sh                           # smoke test on first 3
#   RESET=1 bash sing_share.sh                           # clear previous outputs first
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-/data/home/fundwotsai/MIDI-SAG/share_singing}
MIDI_DIR=${MIDI_DIR:-$OUT_ROOT/midi_real}
WAV_DIR=${WAV_DIR:-$OUT_ROOT/singing_voice_real_region}
FASTSINGER_DIR=${FASTSINGER_DIR:-/data/home/fundwotsai/MIDI-SAG/fastsinger}
PITCH_PICKING_DIR=${PITCH_PICKING_DIR:-/data/home/fundwotsai/MIDI-SAG/lyrics2melody_new}
CONDA_ENV=${CONDA_ENV:-fastsinger}
SINGER=${SINGER:-auto}
LIMIT=${LIMIT:-0}
GPU=${GPU:-0}
OVERWRITE=${OVERWRITE:-0}
RESET=${RESET:-0}

META_PATH="$OUT_ROOT/meta_real_region.json"

cd "$FASTSINGER_DIR"

if [[ "$RESET" != "0" ]]; then
  echo "=== RESET: clearing previous real-region outputs ==="
  rm -rf "$WAV_DIR" "$OUT_ROOT/lyrics_real_region" "$OUT_ROOT/meta_pp_real_region"
  rm -f "$META_PATH"
fi

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" != "0" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

echo "=== Preparing lyric files + singer selection (no pitch shift) ==="
conda run --no-capture-output -n "$CONDA_ENV" python prepare_real_region_jobs.py \
  --midi_dir "$MIDI_DIR" \
  --out_root "$OUT_ROOT" \
  --wav_dir "$WAV_DIR" \
  --pitch_picking_dir "$PITCH_PICKING_DIR" \
  --singer "$SINGER" \
  --limit "$LIMIT" \
  "${OVERWRITE_FLAG[@]}"

echo "=== Singing with FastSinger (untransposed) ==="
CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$CONDA_ENV" python inference.py \
  --model_id suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag \
  --model_epoch 400 \
  --shift_consonant_forward_alignment \
  --meta_json_path "$META_PATH"

echo "=== Done. Singing voices under $WAV_DIR ==="

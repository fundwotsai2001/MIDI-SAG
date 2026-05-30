#!/usr/bin/env bash
# Sing MIDIs from the `share` folder with FastSinger, with optional structural filtering.
#
#   1. (optional) Select songs whose structure is intro -> verse -> chorus and
#      whose verse starts within VERSE_MAX seconds (the MAX_SONGS earliest).
#   2. Pitch-shift each MIDI into a comfortable range and pick a male/female
#      singer via lyrics2melody_new/pitch_picking.py (as in lyrics2singing_ablation.sh).
#   3. Reconstruct a FastSinger lyric .txt from each MIDI's embedded lyrics.
#   4. Run FastSinger once (single model load) over the whole batch.
#   5. Write structure start seconds for the rendered songs.
#
# Outputs under OUT_ROOT: midi_shifted/ (shifted MIDIs), wav/ (singing voice),
# structure_times.json (structure start seconds), plus lyrics/, meta_pp/, meta.json.
#
# Usage:
#   bash sing_share.sh                                  # all songs, auto singer
#   VERSE_MAX=16.4 MAX_SONGS=200 RESET=1 bash sing_share.sh   # filtered 200-song run
#   LIMIT=3 bash sing_share.sh                          # smoke test on first 3 songs
set -euo pipefail

SHARE_DIR=${SHARE_DIR:-/data/home/fundwotsai/MIDI-SAG_not_using/share}
OUT_ROOT=${OUT_ROOT:-/data/home/fundwotsai/MIDI-SAG/share_singing}
FASTSINGER_DIR=${FASTSINGER_DIR:-/data/home/fundwotsai/MIDI-SAG/fastsinger}
PITCH_PICKING_DIR=${PITCH_PICKING_DIR:-/data/home/fundwotsai/MIDI-SAG/lyrics2melody_new}
STRUCT_DIR=${STRUCT_DIR:-/data/home/fundwotsai/MIDI-SAG_not_using/sentence_struct}
CONDA_ENV=${CONDA_ENV:-fastsinger}
SINGER=${SINGER:-auto}
LIMIT=${LIMIT:-0}
SAMPLE=${SAMPLE:-0}
SEED=${SEED:-0}
VERSE_MAX=${VERSE_MAX:-0}
MAX_SONGS=${MAX_SONGS:-0}
GPU=${GPU:-0}
OVERWRITE=${OVERWRITE:-0}
RESET=${RESET:-0}

META_PATH="$OUT_ROOT/meta.json"

cd "$FASTSINGER_DIR"

if [[ "$RESET" != "0" ]]; then
  echo "=== RESET: clearing previous outputs under $OUT_ROOT ==="
  rm -rf "$OUT_ROOT/wav" "$OUT_ROOT/midi_shifted" "$OUT_ROOT/lyrics" "$OUT_ROOT/meta_pp"
  rm -f "$OUT_ROOT/meta.json" "$OUT_ROOT/structure_times.json"
fi

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" != "0" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

echo "=== Selecting + pitch shifting + preparing lyric files and meta.json ==="
conda run --no-capture-output -n "$CONDA_ENV" python prepare_share_jobs.py \
  --share_dir "$SHARE_DIR" \
  --out_root "$OUT_ROOT" \
  --pitch_picking_dir "$PITCH_PICKING_DIR" \
  --struct_dir "$STRUCT_DIR" \
  --singer "$SINGER" \
  --limit "$LIMIT" \
  --sample "$SAMPLE" \
  --seed "$SEED" \
  --verse_max "$VERSE_MAX" \
  --max_songs "$MAX_SONGS" \
  "${OVERWRITE_FLAG[@]}"

echo "=== Singing with FastSinger ==="
CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$CONDA_ENV" python inference.py \
  --model_id suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag \
  --model_epoch 400 \
  --shift_consonant_forward_alignment \
  --meta_json_path "$META_PATH"

echo "=== Writing structure start seconds ==="
conda run --no-capture-output -n "$CONDA_ENV" python write_structure_times.py \
  --midi_dir "$OUT_ROOT/midi_shifted" \
  --struct_dir "$STRUCT_DIR" \
  --out_path "$OUT_ROOT/structure_times.json"

echo "=== Done. Outputs under $OUT_ROOT (wav/, midi_shifted/, structure_times.json) ==="

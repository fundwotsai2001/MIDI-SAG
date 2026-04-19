#!/usr/bin/env bash
set -euo pipefail
logdir=/data/home/fundwotsai/MIDI-SAG/logs
mkdir -p "$logdir"

# ---- logging wrapper ---------------------------------------------------------
run() {
  local name=$1; shift
  echo "=== [$name] start $(date) ==="
  (
    stdbuf -oL -eL "$@" 2>&1 | sed -u "s/^/[$name] /"
  ) | tee -a "$logdir/$name.log"
  echo "=== [$name] end   $(date) ==="
  echo
}

# ---- per-song job ------------------------------------------------------------
process_one() {
  local ORIGINAL_LYRIC_PATH=$1
  local STEM
  STEM="$(basename "$ORIGINAL_LYRIC_PATH" .txt)"

  # Base dirs
  local L2M_ROOT=/data/home/fundwotsai/MIDI-SAG
  local OUTDIR="$L2M_ROOT/generated_midi_right_key/$STEM"
  mkdir -p "$OUTDIR" "$OUTDIR/chord" "$OUTDIR/chord/chord_txt" "$OUTDIR/chord/chord_btc_txt"

  # Paths within this song dir
  local MIDI_FOLDER="$OUTDIR"
  local MIDI_PATH="$OUTDIR/sample01.mid"
  local SHIFTED_MIDI_PATH="$OUTDIR/sample01_shifted.mid"
  local BPM_PATH="$OUTDIR/sample01.json"
  local LYRIC_PATH="$OUTDIR/sample01.txt"
  local TIME_LYRIC_PATH="$OUTDIR/sample01_shifted_times.txt"
  local STRUCT_label_LIST="$OUTDIR/sample01_struct_label.json"
  local STRUCT_time_LIST="$OUTDIR/sample01_struct_time.json"
  local VOCAL_PATH="$OUTDIR/sample01_shifted.wav"

  # # ===== Pipeline =====
  # cd "$L2M_ROOT"

  # run "generate_midi:$STEM" \
  #   conda run -p /volume/nas-fundwo-storage/fundwo-test/miniconda3/envs/CSL-L2M \
  #     python -u generate.py \
  #       config/CSLL2M.yaml pretrained_CSLL2M.pt \
  #       "$OUTDIR" 1 "$ORIGINAL_LYRIC_PATH"

  # run "read_midi:$STEM" \
  #   conda run -p /volume/nas-fundwo-storage/fundwo-test/miniconda3/envs/CSL-L2M \
  #     python -u read_midi.py "$MIDI_PATH" "$LYRIC_PATH"

  # run "pitch_picking:$STEM" \
  #   conda run -p /volume/nas-fundwo-storage/fundwo-test/miniconda3/envs/CSL-L2M \
  #     python -u pitch_picking.py "$MIDI_PATH" --bpm "$BPM_PATH"
  # run "read_midi_lyrics_timestamp:$STEM" \
  #   conda run -p /volume/nas-fundwo-storage/fundwo-test/miniconda3/envs/CSL-L2M \
  #     python -u read_midi_lyrics_timestamp.py "$MIDI_FOLDER"

  # run "match_segment_times:$STEM" \
  #   conda run -p /volume/nas-fundwo-storage/fundwo-test/miniconda3/envs/CSL-L2M \
  #     python -u match_segment_times.py "$ORIGINAL_LYRIC_PATH" "$TIME_LYRIC_PATH" "$STRUCT_time_LIST" "$STRUCT_label_LIST"

  # local SINGER_ID
  # SINGER_ID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['singer'])" "$BPM_PATH")

  # cd /volume/nas-fundwo-storage/fundwo-test/fastsinger
  # run "fast_singer:$STEM" \
  #   conda run -p /volume/nas-fundwo-storage/fundwo-test/miniconda3/envs/fast-singer \
  #     python inference.py \
  #     --model_id suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag \
  #     --model_epoch 400 \
  #     --midi_path "$SHIFTED_MIDI_PATH" \
  #     --lyric_path "$LYRIC_PATH" \
  #     --spkr_ref "$SINGER_ID" \
  #     --pitch_shifts 0 \
  #     --shift_consonant_forward_alignment \
  #     --output_path "$VOCAL_PATH"

  cd /data/home/fundwotsai/MIDI-SAG/AccoMontage2
  run "melody2chord:$STEM" \
    python demo_SOME.py \
      --midi_path "$SHIFTED_MIDI_PATH" \
      --output_dir "$OUTDIR/chord" \
      --chord_style POP_COMPLEX
}

# ---- batch driver ------------------------------------------------------------
# Usage: bash lyrics2singing_ablation.sh [id1 id2 ...]
#   Pass IDs as positional args to process only those songs, e.g.: bash lyrics2singing_ablation.sh 28 40 50 203
#   If no IDs are given, all songs are processed.
SONG_DIR=/data/home/fundwotsai/MIDI-SAG/generated_midi_right_key
ID_ARGS=("$@")

shopt -s nullglob
if (( ${#ID_ARGS[@]} > 0 )); then
  echo "ID filter: ${ID_ARGS[*]}"
  DIRS=()
  for id in "${ID_ARGS[@]}"; do
    d="$SONG_DIR/lyrics_${id}"
    if [[ -d "$d" ]]; then
      DIRS+=("$d")
    else
      echo "Warning: $d not found, skipping ID $id"
    fi
  done
else
  DIRS=("$SONG_DIR"/lyrics_*)
fi

if (( ${#DIRS[@]} == 0 )); then
  echo "No songs to process."; exit 1
fi

echo "Processing ${#DIRS[@]} song(s)..."
for d in "${DIRS[@]}"; do
  stem="$(basename "$d")"
  # Pass a synthetic lyric path so process_one derives the correct STEM
  process_one "$d/${stem}.txt"
done

echo "All jobs done."

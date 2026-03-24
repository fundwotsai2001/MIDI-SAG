# ── User config ──────────────────────────────────────────────────────────────
VOCAL_AUDIO_PATH="/data/home/fundwotsai/MIDI-SAG/vocal_audio_test_data/366.wav"
OUTPUT_DIR="./output_game_3_23"
BACKING_TEXT_PROMPT="piano and drums, in the style of pop music"
MUSECONTROLLITE_CHECKPOINT="./MIDI-SAG_checkpoints/checkpoint_gt_beat"
# Chord style for harmonization: POP_STANDARD | POP_COMPLEX | DARK | RANDB | NOCONSTRAINT
CHORD_STYLE="POP_COMPLEX"

# ── Derived paths ─────────────────────────────────────────────────────────────
SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"
VOCAL_MIDI_PATH="$OUTPUT_DIR/vocal_MIDI/$SONG_NAME.mid"
CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
BEAT_PATH="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times.txt"

# ── VAD: trim leading silence ─────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/vad_audio"
python vad_trim.py "$VOCAL_AUDIO_PATH" "$OUTPUT_DIR/vad_audio/$SONG_NAME.wav"
VOCAL_AUDIO_PATH="$OUTPUT_DIR/vad_audio/$SONG_NAME.wav"

# ── Pipeline ──────────────────────────────────────────────────────────────────
# 1. Vocal beat tracking
python Singing-Vocal-Beat-Tracking/inference_vad.py \
    --audio_path "$VOCAL_AUDIO_PATH" \
    --model_path MIDI-SAG_checkpoints/model-16.pt \
    --use_vad --fill_silence \
    --output_dir "$OUTPUT_DIR/vocal_beat"

# 2. Vocal MIDI transcription
python GAME/infer.py extract "$VOCAL_AUDIO_PATH" \
    -m GAME/GAME-1.0-medium/model.pt \
    --output-dir "$OUTPUT_DIR/vocal_MIDI"

# 3. Melody harmonization
python AccoMontage2/demo_SOME.py \
    --midi_path "$VOCAL_MIDI_PATH" \
    --beat_file "$BEAT_PATH" \
    --output_dir "$OUTPUT_DIR/Harmonization_results" \
    --beat_subdivision 1 \
    --chord_style "$CHORD_STYLE"

# 4. Backing track generation
python MuseControlLite/MuseControlLite_inference_47s.py \
    --vocal_audio_file "$VOCAL_AUDIO_PATH" \
    --text_prompt "$BACKING_TEXT_PROMPT" \
    --chord_file "$CHORD_PATH" \
    --vocal_beat_file "$BEAT_PATH" \
    --checkpoint_path "$MUSECONTROLLITE_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR/Backing_track"

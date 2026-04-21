# ── User config ──────────────────────────────────────────────────────────────
VOCAL_AUDIO_PATH="/data/home/fundwotsai/MIDI-SAG/demo_website/experiment_1/gt_vocal_MIDI/简单爱-周杰伦-97-C.wav"
SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"
OUTPUT_DIR="/data/home/fundwotsai/MIDI-SAG/test/${SONG_NAME}"

# Mode selector: "47s" | "full_song"
#   47s       → MuseControlLite_inference_47s_scale_up.py, one text prompt per run
#   full_song → MuseControlLite_inference_continuation.py, uses STRUCTURE_* arrays
MODE="full_song"
MUSECONTROLLITE_CHECKPOINT="./MIDI-SAG_checkpoints/MuseControlLite_checkpoint"
# Chord style for harmonization: POP_STANDARD | POP_COMPLEX | DARK | RANDB | NOCONSTRAINT
CHORD_STYLE="POP_COMPLEX"
# Chords per bar: 1 (default, bar-level) or 2 (half-bar; 2nd half = next bar's chord).
CHORDS_PER_BAR=1
# Musical key of the vocal melody passed to AccoMontage2/demo_SOME.py.
#   Major: C  C#  Db  D  D#  Eb  E  F  F#  Gb  G  G#  Ab  A  A#  Bb  B
#   Minor: Cm C#m Dbm Dm D#m Ebm Em Fm F#m Gbm Gm G#m Abm Am A#m Bbm Bm
#   auto : read MIDI key_signature, fall back to Bellman-Budge heuristic
KEY="auto"
# Index into the detected-only beat file (…_beat_times_detected_only.txt)
# selecting which detected beat is bar 1 beat 1. Downbeats then occur at
# that anchor ± 4·beat_interval·x for integer x. Set to None to let
# AccoMontage2/demo_SOME.py detect the downbeat phase automatically (sometimes it is not accurate).
DOWNBEAT_PHASE=3
# DBN beat tracker BPM bounds (passed to Singing-Vocal-Beat-Tracking/inference_vad.py).
# Leave empty to use madmom defaults.
MIN_BPM=60
MAX_BPM=160

# ── 47s mode config ──────────────────────────────────────────────────────────
# One prompt per run; the inference script is called once per entry.
BACKING_TEXT_PROMPTS=(
  "reflective instrumental pop with piano, synth pad, bass, and steady drums"
  "groovy funk with bass guitar, electric guitar, drums, and electric piano"
  "intense instrumental rock with electric guitar riffs, bass guitar, drums, and powerful energy"
  "melancholic electronic backing with piano, bass, reflective synth textures, and instrumental pop style"
)

# ── Full-song (continuation) mode config ─────────────────────────────────────
# One entry per segment; STRUCTURE_STARTS, STRUCTURE_TAGS, STRUCTURE_PROMPTS
# must all have the same length. Tags should be in this set (intro, verse, chorus, bridge, outro, break, inst, solo)
STRUCTURE_PROMPTS=(
    "Tranquil and dreamy, featuring soft piano, synth pads, and a slow, ethereal, meditative atmosphere."
    "Gentle and sentimental instrumental pop with electric piano (Rhodes), cello, strings, and a poignant, nostalgic, reflective mood."
    "Upbeat, inspiring, and cinematic with drums, piano, strings, bass guitar, and a soaring, uplifting, emotional, hopeful energy."
    "Upbeat, inspiring, and cinematic with drums, piano, strings, bass guitar, and a soaring, uplifting, emotional, hopeful energy."
)
# The gap between each structure starts should be shorter than 47 seconds (the last segment could not be longer then 47 second either, i.e, vocal_audio_length - final_structure_start < 47 seconds)
STRUCTURE_STARTS=( 0.00  27.53  66.19 96 )
STRUCTURE_TAGS=(   intro   verse   chorus chorus)


# ── Derived paths ─────────────────────────────────────────────────────────────
VOCAL_MIDI_PATH="$OUTPUT_DIR/vocal_MIDI/$SONG_NAME.mid"
CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
BEAT_PATH="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times.txt"
BEAT_PATH_DETECTED_ONLY="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times_detected_only.txt"
DOWNBEAT_PATH="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_downbeat_times.txt"

# ── VAD: trim leading silence (skipped in full_song mode) ────────────────────
if [ "$MODE" != "full_song" ]; then
    mkdir -p "$OUTPUT_DIR/vad_audio"
    python vad_trim.py "$VOCAL_AUDIO_PATH" "$OUTPUT_DIR/vad_audio/$SONG_NAME.wav"
    VOCAL_AUDIO_PATH="$OUTPUT_DIR/vad_audio/$SONG_NAME.wav"
else
    echo "Mode: full_song — skipping VAD trim"
fi

# ── Pipeline ──────────────────────────────────────────────────────────────────
# 1. Vocal beat tracking
BPM_ARGS=()
[ -n "$MIN_BPM" ] && BPM_ARGS+=(--min_bpm "$MIN_BPM")
[ -n "$MAX_BPM" ] && BPM_ARGS+=(--max_bpm "$MAX_BPM")

python Singing-Vocal-Beat-Tracking/inference_vad.py \
    --audio_path "$VOCAL_AUDIO_PATH" \
    --model_path MIDI-SAG_checkpoints/model-16.pt \
    --use_vad --fill_silence --vad_merge_gap 3.0 --first_bpm_margin 10 \
    "${BPM_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR/vocal_beat"

# 2. Vocal MIDI transcription
python GAME/infer.py extract "$VOCAL_AUDIO_PATH" \
    -m GAME/GAME-1.0-medium/model.pt \
    --output-dir "$OUTPUT_DIR/vocal_MIDI"

# 3. Melody harmonization
#    Use the detected-only beat file to warp/quantize the melody MIDI onto its
#    own clean grid (so the template matcher isn't fed gap-filled fake beats),
#    then use the interpolated beat file to place final chord-text timestamps
#    on the full-song timeline.
python AccoMontage2/demo_SOME.py \
    --midi_path "$VOCAL_MIDI_PATH" \
    --beat_file "$BEAT_PATH" \
    --beat_file_detected "$BEAT_PATH_DETECTED_ONLY" \
    --output_dir "$OUTPUT_DIR/Harmonization_results" \
    --beat_subdivision 1 \
    --downbeat_phase "$DOWNBEAT_PHASE" \
    --chord_style "$CHORD_STYLE" \
    --chords_per_bar "$CHORDS_PER_BAR" \
    --key "$KEY"

# 4. Backing track generation
mkdir -p "$OUTPUT_DIR/Backing_track"
PROMPT_LOG_PATH="$OUTPUT_DIR/Backing_track/prompts_used.txt"

case "$MODE" in
  47s)
    INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_47s_scale_up.py"
    echo "Mode: 47s — loading model once, generating ${#BACKING_TEXT_PROMPTS[@]} prompt(s)"

    if python "$INFERENCE_SCRIPT" \
        --vocal_audio_file "$VOCAL_AUDIO_PATH" \
        --text_prompt "${BACKING_TEXT_PROMPTS[@]}" \
        --chord_file "$CHORD_PATH" \
        --vocal_beat_file "$BEAT_PATH" \
        --vocal_downbeat_file "$DOWNBEAT_PATH" \
        --checkpoint_path "$MUSECONTROLLITE_CHECKPOINT" \
        --output_dir "$OUTPUT_DIR/Backing_track/"; then
        {
            printf 'inference_script=%s\n\n[text_prompts]\n' "$INFERENCE_SCRIPT"
            for i in "${!BACKING_TEXT_PROMPTS[@]}"; do
                printf '%d\t%s\n' "$((i + 1))" "${BACKING_TEXT_PROMPTS[$i]}"
            done
        } > "$PROMPT_LOG_PATH"
        echo "Saved prompts to $PROMPT_LOG_PATH"
    fi
    ;;

  full_song)
    INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_continuation.py"
    echo "Mode: full_song — continuation with ${#STRUCTURE_STARTS[@]} segment(s)"

    if [ ${#STRUCTURE_STARTS[@]} -ne ${#STRUCTURE_TAGS[@]} ] || \
       [ ${#STRUCTURE_STARTS[@]} -ne ${#STRUCTURE_PROMPTS[@]} ]; then
        echo "Error: STRUCTURE_STARTS, STRUCTURE_TAGS, STRUCTURE_PROMPTS must have the same length." >&2
        exit 1
    fi

    STRUCT_ARGS=(
        --structure_starts  "${STRUCTURE_STARTS[@]}"
        --structure_tags    "${STRUCTURE_TAGS[@]}"
        --structure_prompts "${STRUCTURE_PROMPTS[@]}"
    )

    if python "$INFERENCE_SCRIPT" \
        --vocal_audio_file "$VOCAL_AUDIO_PATH" \
        --chord_file "$CHORD_PATH" \
        --vocal_beat_file "$BEAT_PATH" \
        --vocal_downbeat_file "$DOWNBEAT_PATH" \
        --checkpoint_path "$MUSECONTROLLITE_CHECKPOINT" \
        --output_dir "$OUTPUT_DIR/Backing_track/" \
        "${STRUCT_ARGS[@]}"; then
        {
            printf 'inference_script=%s\n\n[structure_segments]\n' "$INFERENCE_SCRIPT"
            for i in "${!STRUCTURE_STARTS[@]}"; do
                printf '%d\t%s\t%s\t%s\n' \
                    "$((i + 1))" \
                    "${STRUCTURE_STARTS[$i]}" \
                    "${STRUCTURE_TAGS[$i]}" \
                    "${STRUCTURE_PROMPTS[$i]}"
            done
        } > "$PROMPT_LOG_PATH"
        echo "Saved prompts to $PROMPT_LOG_PATH"
    fi
    ;;

  *)
    echo "Error: MODE must be '47s' or 'full_song' (got '$MODE')." >&2
    exit 1
    ;;
esac

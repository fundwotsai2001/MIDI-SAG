# ── User config ──────────────────────────────────────────────────────────────
VOCAL_MIDI_PATH="/data/home/fundwotsai/MIDI-SAG/demo_website/experiment_1/gt_vocal_MIDI/小手拉大手-梁靜茹-130-C大调.mid"
VOCAL_AUDIO_PATH="/data/home/fundwotsai/MIDI-SAG/demo_website/experiment_1/gt_vocal_MIDI/小手拉大手-梁靜茹-130-C大调.wav"

# Mode selector: "47s" | "full_song"
#   47s       → MuseControlLite_inference_47s_scale_up.py, all prompts in one model load
#   full_song → MuseControlLite_inference_continuation.py, uses STRUCTURE_* arrays
MODE="full_song"

SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"
OUTPUT_DIR="/data/home/fundwotsai/MIDI-SAG/test/${SONG_NAME}"
CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
MUSECONTROLLITE_CHECKPOINT="./MIDI-SAG_checkpoints/MuseControlLite_checkpoint"
PROMPT_LOG_PATH="$OUTPUT_DIR/Backing_track/prompts_used.txt"

# Chord style for harmonization: POP_STANDARD | POP_COMPLEX | DARK | RANDB | NOCONSTRAINT
CHORD_STYLE="POP_STANDARD"

# ── 47s mode config ──────────────────────────────────────────────────────────
# Loaded once; the inference script iterates through these prompts internally.
BACKING_TEXT_PROMPTS=(
  "reflective instrumental pop with piano, synth pad, bass, and steady drums"
  "groovy funk with bass guitar, electric guitar, drums, and electric piano"
  "intense instrumental rock with electric guitar riffs, bass guitar, drums, and powerful energy"
  "melancholic electronic backing with piano, bass, reflective synth textures, and instrumental pop style"
)

# ── Full-song (continuation) mode config ─────────────────────────────────────
# One entry per segment; STRUCTURE_STARTS, STRUCTURE_TAGS, STRUCTURE_PROMPTS
# must all have the same length. Tags: intro | verse | chorus | bridge | outro | break | inst | solo
STRUCTURE_PROMPTS=(
  "Soft synthesizer pads open slowly, ethereal and tranquil, meditative and dreamy, a gentle wash of sound with no rhythm yet."
  "A subtle drum machine beat enters beneath the synth pads, bass synthesizer pulses gently, calm and chillout, soothing and relaxed with a peaceful, floating atmosphere."
  "Synthesizer leads rise over the pads and beat, drum machine intensifies slightly, bass drives with more energy, uplifting and danceable, electronic and dreamy with an inspiring, euphoric surge."
)
STRUCTURE_STARTS=( 0.00  30.23  73.85 )
STRUCTURE_TAGS=(   intro   verse   chorus )


# Optional harmonization step (uncomment to regenerate chords from the MIDI)
python AccoMontage2/demo_SOME.py \
    --midi_path "$VOCAL_MIDI_PATH" \
    --output_dir "$OUTPUT_DIR/Harmonization_results" \
    --beat_subdivision 1 \
    --chord_style "$CHORD_STYLE" \
    --chords_per_bar 1

# ── Backing track generation ─────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/Backing_track"

case "$MODE" in
  47s)
    INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_47s_scale_up.py"
    echo "Mode: 47s — loading model once, generating ${#BACKING_TEXT_PROMPTS[@]} prompt(s)"

    if python "$INFERENCE_SCRIPT" \
        --vocal_audio_file "$VOCAL_AUDIO_PATH" \
        --text_prompt "${BACKING_TEXT_PROMPTS[@]}" \
        --chord_file "$CHORD_PATH" \
        --vocal_midi_file "$VOCAL_MIDI_PATH" \
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
        --vocal_midi_file "$VOCAL_MIDI_PATH" \
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

VOCAL_MIDI_PATH="/data/home/fundwotsai/MIDI-SAG/demo_website/experiment_1/gt_vocal_MIDI/小手拉大手-梁靜茹-130-C大调.mid"
VOCAL_AUDIO_PATH="/data/home/fundwotsai/MIDI-SAG/demo_website/experiment_1/gt_vocal_MIDI/小手拉大手-梁靜茹-130-C大调.wav"


SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"
OUTPUT_DIR="/data/home/fundwotsai/MIDI-SAG/demo_website/experiment_1/${SONG_NAME}"
CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
MUSECONTROLLITE_CHECKPOINT="/data/home/fundwotsai/MIDI-SAG/MuseControlLite/checkpoint-65000"
PROMPT_LOG_PATH="$OUTPUT_DIR/Backing_track/prompts_used.txt"

INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_continuation.py"
# INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_47s_scale_up.py"

# Chord style for harmonization: POP_STANDARD | POP_COMPLEX | DARK | RANDB | NOCONSTRAINT
CHORD_STYLE="POP_STANDARD"
BACKING_TEXT_PROMPTS=(
  "reflective instrumental pop with piano, synth pad, bass, and steady drums"
  "groovy funk with bass guitar, electric guitar, drums, and electric piano"
  "intense instrumental rock with electric guitar riffs, bass guitar, drums, and powerful energy"
  "melancholic electronic backing with piano, bass, reflective synth textures, and instrumental pop style"
)
# python AccoMontage2/demo_SOME.py \
#     --midi_path "$VOCAL_MIDI_PATH" \
#     --output_dir "$OUTPUT_DIR/Harmonization_results" \
#     --beat_subdivision 1 \
#     --chord_style "$CHORD_STYLE" \
#     --chords_per_bar 1
TEXT_PROMPT_ARGS=(--text_prompt "${BACKING_TEXT_PROMPTS[@]}")
STRUCTURE_PROMPTS=(
  "Soft synthesizer pads open slowly, ethereal and tranquil, meditative and dreamy, a gentle wash of sound with no rhythm yet."
  "A subtle drum machine beat enters beneath the synth pads, bass synthesizer pulses gently, calm and chillout, soothing and relaxed with a peaceful, floating atmosphere."
  "Synthesizer leads rise over the pads and beat, drum machine intensifies slightly, bass drives with more energy, uplifting and danceable, electronic and dreamy with an inspiring, euphoric surge."
)

STRUCTURE_STARTS=( 0.00  30.23  73.85 )
STRUCTURE_TAGS=(   intro   verse   chorus )






# Structure args are only used by continuation mode, not 47s_scale_up.
STRUCT_ARGS=()
if [[ "$INFERENCE_SCRIPT" == *"continuation"* ]] && [ ${#STRUCTURE_STARTS[@]} -gt 0 ]; then
    STRUCT_ARGS+=(--structure_starts "${STRUCTURE_STARTS[@]}")
    [ -n "$STRUCTURE_DURATION" ] && STRUCT_ARGS+=(--structure_duration "$STRUCTURE_DURATION")
    [ ${#STRUCTURE_TAGS[@]}    -gt 0 ] && STRUCT_ARGS+=(--structure_tags    "${STRUCTURE_TAGS[@]}")
    [ ${#STRUCTURE_PROMPTS[@]} -gt 0 ] && STRUCT_ARGS+=(--structure_prompts "${STRUCTURE_PROMPTS[@]}")
fi

PROMPTS_TO_RUN=()
if [ ${#BACKING_TEXT_PROMPTS[@]} -gt 0 ]; then
    PROMPTS_TO_RUN=("${BACKING_TEXT_PROMPTS[@]}")
else
    PROMPTS_TO_RUN=("$BACKING_TEXT_PROMPT")
fi

TEXT_PROMPT_ARGS=()
if [[ "$INFERENCE_SCRIPT" == *"47s_scale_up"* ]]; then
    TEXT_PROMPT_ARGS=(--text_prompt "${PROMPTS_TO_RUN[@]}")
else
    TEXT_PROMPT_ARGS=(--text_prompt "$BACKING_TEXT_PROMPT")
fi

if python "$INFERENCE_SCRIPT" \
    --vocal_audio_file "$VOCAL_AUDIO_PATH" \
    "${TEXT_PROMPT_ARGS[@]}" \
    --chord_file "$CHORD_PATH" \
    --vocal_midi_file "$VOCAL_MIDI_PATH" \
    --checkpoint_path "$MUSECONTROLLITE_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR/Backing_track/" \
    "${STRUCT_ARGS[@]}"; then
    {
        printf 'inference_script=%s\n' "$INFERENCE_SCRIPT"
        printf 'fallback_text_prompt=%s\n' "$BACKING_TEXT_PROMPT"
        printf '\n'

        if [[ "$INFERENCE_SCRIPT" == *"47s_scale_up"* ]]; then
            printf '[text_prompts]\n'
            for i in "${!PROMPTS_TO_RUN[@]}"; do
                printf '%d\t%s\n' "$((i + 1))" "${PROMPTS_TO_RUN[$i]}"
            done
        else
            printf '[structure_segments]\n'
            for i in "${!STRUCTURE_STARTS[@]}"; do
                segment_tag="${STRUCTURE_TAGS[$i]:-verse}"
                segment_prompt="${STRUCTURE_PROMPTS[$i]:-$BACKING_TEXT_PROMPT}"
                printf '%d\t%s\t%s\t%s\n' \
                    "$((i + 1))" \
                    "${STRUCTURE_STARTS[$i]}" \
                    "$segment_tag" \
                    "$segment_prompt"
            done
        fi
    } > "$PROMPT_LOG_PATH"
    echo "Saved prompts to $PROMPT_LOG_PATH"
fi

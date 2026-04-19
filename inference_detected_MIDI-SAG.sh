# ── User config ──────────────────────────────────────────────────────────────
VOCAL_AUDIO_PATH="/data/home/fundwotsai/MIDI-SAG/demo_website/experiment_1/gt_vocal_MIDI/简单爱-周杰伦-97-C.wav"
OUTPUT_DIR="/data/home/fundwotsai/MIDI-SAG/demo_website/experiment_1/简单爱-周杰伦-97-C-detected"
# Optional batch prompt list for step 4. If non-empty, the inference script is
# called once and will iterate through these prompts after loading the model and
# conditions. Leave empty to use BACKING_TEXT_PROMPT only.
BACKING_TEXT_PROMPTS=(
  "reflective instrumental pop with piano, synth pad, bass, and steady drums"
  "groovy funk with bass guitar, electric guitar, drums, and electric piano"
  "intense instrumental rock with electric guitar riffs, bass guitar, drums, and powerful energy"
  "melancholic electronic backing with piano, bass, reflective synth textures, and instrumental pop style"
)
MUSECONTROLLITE_CHECKPOINT="/data/home/fundwotsai/MIDI-SAG/MuseControlLite/checkpoint-65000"
# Chord style for harmonization: POP_STANDARD | POP_COMPLEX | DARK | RANDB | NOCONSTRAINT
CHORD_STYLE="POP_COMPLEX"
# Chords per bar: 1 (default, bar-level) or 2 (half-bar; 2nd half = next bar's chord).
CHORDS_PER_BAR=1
# Index into the detected-only beat file (…_beat_times_detected_only.txt)
# selecting which detected beat is bar 1 beat 1. Downbeats then occur at
# that anchor ± 4·beat_interval·x for integer x. Set to None to let
# AccoMontage2/demo_SOME.py detect the downbeat phase automatically.
DOWNBEAT_PHASE=3
# Inference script: continuation generates segment-by-segment with audio carry-over;
#                   47s_scale_up uses a sliding-window approach.
INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_continuation.py"
# INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_47s_scale_up.py"
# Optional structure for continuation mode (ignored by 47s_scale_up).
# One entry per segment; all three arrays must have the same length, or all be empty.
# Tags: intro | verse | chorus | bridge | outro | break | inst | solo
# Leave all empty to use BACKING_TEXT_PROMPT for the whole song as one segment.
# The Nth start/tag/prompt entries belong to the same segment.
#                start(s)  tag       prompt

# Schoolboy Fascination
# STRUCTURE_STARTS=(  0.0     45.4   64.51    108.56    143.46   178.04)
# STRUCTURE_TAGS=(   verse   chorus  verse  chorus    verse   chorus)
STRUCTURE_PROMPTS=(
    "Tranquil and dreamy, featuring soft piano, synth pads, and a slow, ethereal, meditative atmosphere."
    "Gentle and sentimental instrumental pop with electric piano (Rhodes), cello, strings, and a poignant, nostalgic, reflective mood."
    "Upbeat, inspiring, and cinematic with drums, piano, strings, bass guitar, and a soaring, uplifting, emotional, hopeful energy."
    "Upbeat, inspiring, and cinematic with drums, piano, strings, bass guitar, and a soaring, uplifting, emotional, hopeful energy."
)

STRUCTURE_STARTS=( 0.00  27.53  66.19 96 )
STRUCTURE_TAGS=(   intro   verse   chorus chorus)




# ── Derived paths ─────────────────────────────────────────────────────────────
SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"
VOCAL_MIDI_PATH="$OUTPUT_DIR/vocal_MIDI/$SONG_NAME.mid"
CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
BEAT_PATH="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times.txt"
BEAT_PATH_DETECTED_ONLY="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times_detected_only.txt"
DOWNBEAT_PATH="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_downbeat_times.txt"

# # ── VAD: trim leading silence ─────────────────────────────────────────────────
# mkdir -p "$OUTPUT_DIR/vad_audio"
# python vad_trim.py "$VOCAL_AUDIO_PATH" "$OUTPUT_DIR/vad_audio/$SONG_NAME.wav"
# VOCAL_AUDIO_PATH="$OUTPUT_DIR/vad_audio/$SONG_NAME.wav"

# # ── Pipeline ──────────────────────────────────────────────────────────────────
# # 1. Vocal beat tracking
# python Singing-Vocal-Beat-Tracking/inference_vad.py \
#     --audio_path "$VOCAL_AUDIO_PATH" \
#     --model_path MIDI-SAG_checkpoints/model-16.pt \
#     --use_vad --fill_silence --vad_merge_gap 3.0 --first_bpm_margin 10\
#     --output_dir "$OUTPUT_DIR/vocal_beat"

# # 2. Vocal MIDI transcription
# python GAME/infer.py extract "$VOCAL_AUDIO_PATH" \
#     -m GAME/GAME-1.0-medium/model.pt \
#     --output-dir "$OUTPUT_DIR/vocal_MIDI"

# # 3. Melody harmonization
# #    Use the detected-only beat file to warp/quantize the melody MIDI onto its
# #    own clean grid (so the template matcher isn't fed gap-filled fake beats),
# #    then use the interpolated beat file to place final chord-text timestamps
# #    on the full-song timeline.
# python AccoMontage2/demo_SOME.py \
#     --midi_path "$VOCAL_MIDI_PATH" \
#     --beat_file "$BEAT_PATH" \
#     --beat_file_detected "$BEAT_PATH_DETECTED_ONLY" \
#     --output_dir "$OUTPUT_DIR/Harmonization_results" \
#     --beat_subdivision 1 \
#     --downbeat_phase "$DOWNBEAT_PHASE" \
#     --chord_style "$CHORD_STYLE" \
#     --chords_per_bar 1
# 4. Backing track generation
#    Structure args are only used by continuation mode, not 47s_scale_up.
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

mkdir -p "$OUTPUT_DIR/Backing_track"
PROMPT_LOG_PATH="$OUTPUT_DIR/Backing_track/prompts_used.txt"
echo "Running backing-track inference for ${#PROMPTS_TO_RUN[@]} prompt(s) in one model load"
if python "$INFERENCE_SCRIPT" \
    --vocal_audio_file "$VOCAL_AUDIO_PATH" \
    "${TEXT_PROMPT_ARGS[@]}" \
    --chord_file "$CHORD_PATH" \
    --vocal_beat_file "$BEAT_PATH" \
    --vocal_downbeat_file "$DOWNBEAT_PATH" \
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

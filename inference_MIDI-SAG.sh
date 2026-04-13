# ── User config ──────────────────────────────────────────────────────────────
VOCAL_AUDIO_PATH="/data/home/fundwotsai/MIDI-SAG/vocal_audio_test_data/366.wav"
OUTPUT_DIR="/data/home/fundwotsai/MIDI-SAG/demo_website/ilikeyou"
BACKING_TEXT_PROMPT="jazz ensemble with piano and bass"
MUSECONTROLLITE_CHECKPOINT="/data/home/fundwotsai/MIDI-SAG/MuseControlLite/checkpoint-65000"
# Chord style for harmonization: POP_STANDARD | POP_COMPLEX | DARK | RANDB | NOCONSTRAINT
CHORD_STYLE="POP_COMPLEX"
# Inference script: continuation generates segment-by-segment with audio carry-over;
#                   47s_scale_up uses a sliding-window approach.
# INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_47s_scale_up.py"
INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_47s_scale_up.py"
# Optional structure for continuation mode (ignored by 47s_scale_up).
# One entry per segment; all three arrays must have the same length, or all be empty.
# Tags: intro | verse | chorus | bridge | outro | break | inst | solo
# Leave all empty to use BACKING_TEXT_PROMPT for the whole song as one segment.
#                start(s)  tag       prompt
STRUCTURE_STARTS=(  0.0    20.42   45.24   74.54    86.42    116.1   132.66  148.68  181.76  198.31 235.48)
STRUCTURE_TAGS=(    intro   verse   chorus  verse  inst    verse   verse   chorus  chorus  outro  outro)
STRUCTURE_PROMPTS=(
    "indie-rock and experimental pop intro, energetic yet restrained, shimmering electric guitar, soft piano motifs, atmospheric synthesizer textures, gradual rhythmic anticipation"
    "driving indie-rock verse, steady drums, warm bass guitar groove, rhythmic electric guitar, subtle piano accents, intimate but forward-moving energy"
    "big energetic chorus, rock and experimental pop fusion, soaring electric guitar, punchy bass guitar, powerful drums, bright piano chords, wide synthesizer layers"
    "indie-rock verse with more momentum, tighter drum groove, melodic bass line, electric guitar riffs, light piano support, synth textures adding tension"
    "uplifting chorus, energetic and driving, full electric guitar and bass guitar, dynamic drums, piano reinforcement, expansive synthesizer atmosphere"
    "instrumental bridge with experimental pop texture, pulsing synthesizer, expressive piano, layered electric guitar lines, bass-driven movement, drums creating tension and release"
    "final verse, fuller indie-rock arrangement, stronger drums, deeper bass guitar, emotional electric guitar phrasing, piano and synthesizer supporting a rising buildup"
    "final explosive chorus, energetic rock peak, anthemic electric guitar, heavy bass guitar, forceful drums, bright piano, massive synthesizer layers, emotional climax"
    "fading outro, indie-rock and experimental pop atmosphere, ringing electric guitar, soft piano, reduced drums, warm bass sustain, airy synthesizer resolution"
    "fading outro, indie-rock and experimental pop atmosphere, ringing electric guitar, soft piano, reduced drums, warm bass sustain, airy synthesizer resolution"
    "fading outro, indie-rock and experimental pop atmosphere, ringing electric guitar, soft piano, reduced drums, warm bass sustain, airy synthesizer resolution"
)

# ── Derived paths ─────────────────────────────────────────────────────────────
SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"
VOCAL_MIDI_PATH="$OUTPUT_DIR/vocal_MIDI/$SONG_NAME.mid"
CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
BEAT_PATH="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times.txt"
BEAT_PATH_DETECTED_ONLY="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times_detected_only.txt"

# ── VAD: trim leading silence ─────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/vad_audio"
python vad_trim.py "$VOCAL_AUDIO_PATH" "$OUTPUT_DIR/vad_audio/$SONG_NAME.wav"
VOCAL_AUDIO_PATH="$OUTPUT_DIR/vad_audio/$SONG_NAME.wav"

# ── Pipeline ──────────────────────────────────────────────────────────────────
# 1. Vocal beat tracking
python Singing-Vocal-Beat-Tracking/inference_vad.py \
    --audio_path "$VOCAL_AUDIO_PATH" \
    --model_path MIDI-SAG_checkpoints/model-16.pt \
    --use_vad --fill_silence --vad_merge_gap 3.0 --first_bpm_margin 10\
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
    --output_dir "$OUTPUT_DIR/Harmonization_results" \
    --beat_subdivision 1 \
    --chord_style "$CHORD_STYLE"

# 4. Backing track generation
#    Structure args are only used by continuation mode, not 47s_scale_up.
STRUCT_ARGS=()
if [[ "$INFERENCE_SCRIPT" == *"continuation"* ]] && [ ${#STRUCTURE_STARTS[@]} -gt 0 ]; then
    STRUCT_ARGS+=(--structure_starts "${STRUCTURE_STARTS[@]}")
    [ -n "$STRUCTURE_DURATION" ] && STRUCT_ARGS+=(--structure_duration "$STRUCTURE_DURATION")
    [ ${#STRUCTURE_TAGS[@]}    -gt 0 ] && STRUCT_ARGS+=(--structure_tags    "${STRUCTURE_TAGS[@]}")
    [ ${#STRUCTURE_PROMPTS[@]} -gt 0 ] && STRUCT_ARGS+=(--structure_prompts "${STRUCTURE_PROMPTS[@]}")
fi
python "$INFERENCE_SCRIPT" \
    --vocal_audio_file "$VOCAL_AUDIO_PATH" \
    --text_prompt "$BACKING_TEXT_PROMPT" \
    --chord_file "$CHORD_PATH" \
    --vocal_beat_file "$BEAT_PATH" \
    --checkpoint_path "$MUSECONTROLLITE_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR/Backing_track" \
    "${STRUCT_ARGS[@]}"

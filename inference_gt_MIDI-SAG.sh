###############################################################################
# ── User config ──────────────────────────────────────────────────────────────
VOCAL_MIDI_PATH="./example_input/小手拉大手-梁靜茹-130-C大调.mid"
VOCAL_AUDIO_PATH="./example_input/小手拉大手-梁靜茹-130-C大调.wav"
SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"
OUTPUT_DIR="./output_midi_sag/${SONG_NAME}"

# Mode selector: "47s" | "full_song"
#   47s       → MuseControlLite_inference_47s_scale_up.py, all prompts in one model load
#   full_song → MuseControlLite_inference_continuation.py, uses STRUCTURE_* arrays
MODE="full_song"
MUSECONTROLLITE_CHECKPOINT="./MIDI-SAG_checkpoints/MuseControlLite_checkpoint"
# Chord style for harmonization: POP_STANDARD | POP_COMPLEX | DARK | RANDB | NOCONSTRAINT
CHORD_STYLE="POP_STANDARD"
# Chords per bar: 1 (default, bar-level) or 2 (half-bar; 2nd half = next bar's chord).
CHORDS_PER_BAR=1
# Musical key of the vocal melody passed to AccoMontage2/demo_SOME.py.
#   Major: C  C#  Db  D  D#  Eb  E  F  F#  Gb  G  G#  Ab  A  A#  Bb  B
#   Minor: Cm C#m Dbm Dm D#m Ebm Em Fm F#m Gbm Gm G#m Abm Am A#m Bbm Bm
#   auto : read MIDI key_signature, fall back to Bellman-Budge heuristic
KEY="auto"

# Default text prompt for backing track generation (used as fallback in full_song
# mode when a structure tag has no entry in STRUCTURE_TAG_PROMPTS).
BACKING_TEXT_PROMPT="reflective instrumental pop with piano, synth pad, bass, and steady drums"

# ── 47s mode config ──────────────────────────────────────────────────────────
# Loaded once; the inference script iterates through these prompts internally.
BACKING_TEXT_PROMPTS=(
  "$BACKING_TEXT_PROMPT"
  "groovy funk with bass guitar, electric guitar, drums, and electric piano"
  "intense instrumental rock with electric guitar riffs, bass guitar, drums, and powerful energy"
  "melancholic electronic backing with piano, bass, reflective synth textures, and instrumental pop style"
)

# ── Full-song (continuation) mode config ─────────────────────────────────────
# Default text prompt per structure tag (edit to taste).
# Tags not listed here fall back to BACKING_TEXT_PROMPT above.
# Valid tags: intro, verse, chorus, bridge, outro, break, inst, solo
declare -A STRUCTURE_TAG_PROMPTS=(
    [intro]="Soft synthesizer pads open slowly, ethereal and tranquil, meditative and dreamy, a gentle wash of sound"
    [verse]="Melancholic yet hopeful acoustic feel with fingerpicked acoustic guitar, electric piano, bass, and steady drums"
    [chorus]="Energetic rock with driving drums, electric guitar riffs, bass guitar, and an uplifting, emotional release"
    [bridge]="Reflective instrumental passage with piano, synth pad, bass, and gentle percussion"
    [outro]="Soft synthesizer pads fading slowly, ethereal and tranquil, peaceful resolution"
    [break]="Minimal instrumental break with sparse percussion and ambient textures"
    [inst]="Instrumental section with expressive lead melody and full band accompaniment"
    [solo]="Virtuosic instrumental solo with dynamic expression and energy"
)
# The gap between each structure start should be shorter than 47 seconds
# (the last segment could not be longer than 47 seconds either,
#  i.e., vocal_audio_length - final_structure_start < 47 seconds)
STRUCTURE_STARTS=( 0.00  30.23  73.85 )
STRUCTURE_TAGS=(   intro   verse   chorus )

# ── Derived paths ─────────────────────────────────────────────────────────────
CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
PROMPT_LOG_PATH="$OUTPUT_DIR/Mixed_track/prompts_used.txt"

###############################################################################
# you do not need to modify the following 

# ── 1. Melody harmonization ─────────────────────────────────────────────────
python AccoMontage2/demo_SOME.py \
    --midi_path "$VOCAL_MIDI_PATH" \
    --output_dir "$OUTPUT_DIR/Harmonization_results" \
    --beat_subdivision 1 \
    --chord_style "$CHORD_STYLE" \
    --chords_per_bar "$CHORDS_PER_BAR" \
    --key "$KEY"

# ── 2. Backing track generation ─────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/Mixed_track"

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
        --output_dir "$OUTPUT_DIR/Mixed_track/"; then
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

    if [ ${#STRUCTURE_STARTS[@]} -ne ${#STRUCTURE_TAGS[@]} ]; then
        echo "Error: STRUCTURE_STARTS and STRUCTURE_TAGS must have the same length." >&2
        exit 1
    fi

    # Map each tag to its text prompt (fall back to BACKING_TEXT_PROMPT)
    STRUCTURE_PROMPTS=()
    for _tag in "${STRUCTURE_TAGS[@]}"; do
        if [ -n "${STRUCTURE_TAG_PROMPTS[$_tag]+x}" ]; then
            STRUCTURE_PROMPTS+=( "${STRUCTURE_TAG_PROMPTS[$_tag]}" )
        else
            echo "[WARN] No prompt for tag '$_tag'; using BACKING_TEXT_PROMPT" >&2
            STRUCTURE_PROMPTS+=( "$BACKING_TEXT_PROMPT" )
        fi
    done

    echo "Mode: full_song — continuation with ${#STRUCTURE_STARTS[@]} segment(s)"

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
        --output_dir "$OUTPUT_DIR/Mixed_track/" \
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

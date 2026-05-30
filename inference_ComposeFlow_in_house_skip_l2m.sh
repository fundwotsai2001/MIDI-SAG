#!/bin/bash
###############################################################################
# ── User config ──────────────────────────────────────────────────────────────
# This variant SKIPS the CSL-L2M (lyrics→melody) step and re-uses an existing
# lyrics+melody output folder. Point INPUT_FOLDER at a directory that already
# contains the artifacts produced by inference_ComposeFlow_in_house.sh step 1,
# i.e. it must have this layout:
#
#   $INPUT_FOLDER/
#     sample01.txt                              # FastSinger-formatted lyrics
#     lyrics2melody/
#       sample01.mid                            # melody MIDI (with lyric metas)
#       sample01.json                           # {bpm, singer, pitch_shift}
#       sample01_times.txt                      # per-line timestamps
#       sample01_struct_label.json              # ["verse", "chorus", ...]
#       sample01_struct_time.json               # [0.0, 12.34, ...]
#
INPUT_FOLDER="./output_composerflow_demo/25"
OUTPUT_DIR="./output_composerflow_skip_l2m/25/"

# FastSinger voice config
SINGER_GENDER="female" # use male or female (kept for reference; not used here)
FASTSINGER_MODEL_ID="suming_MBJCUganFM_rmvpe_bs32_autoalign_slur_flag"
FASTSINGER_MODEL_EPOCH=400
FASTSINGER_SPKR_REF=6
FASTSINGER_PITCH_SHIFTS=0

# Mode selector: "47s" | "full_song"
#   47s       → MuseControlLite_inference_47s_scale_up.py, one text prompt per run
#   full_song → MuseControlLite_inference_continuation.py, uses structure tag prompts
MODE="full_song"
MUSECONTROLLITE_CHECKPOINT="./MIDI-SAG_checkpoints/MuseControlLite_checkpoint"
# Chord style for harmonization: POP_STANDARD | POP_COMPLEX | DARK | RANDB | NOCONSTRAINT
CHORD_STYLE="POP_STANDARD"
CHORDS_PER_BAR=1
# Musical key of the vocal melody passed to AccoMontage2/demo_SOME.py.
#   Major: C  C#  Db  D  D#  Eb  E  F  F#  Gb  G  G#  Ab  A  A#  Bb  B
#   Minor: Cm C#m Dbm Dm D#m Ebm Em Fm F#m Gbm Gm G#m Abm Am A#m Bbm Bm
#   auto : read MIDI key_signature, fall back to Bellman-Budge heuristic
KEY="auto"
# Default text prompt for backing track generation (used as fallback in full_song
# mode when a structure tag has no entry in STRUCTURE_TAG_PROMPTS).
BACKING_TEXT_PROMPT="piano and drums, in the style of pop music"

# ── 47s mode config ──────────────────────────────────────────────────────────
# Wraps the single BACKING_TEXT_PROMPT into an array; add more entries if desired.
BACKING_TEXT_PROMPTS=( "$BACKING_TEXT_PROMPT" )

# ── Full-song (continuation) mode config ─────────────────────────────────────
# Default text prompt per structure tag (edit to taste).
# Tags not listed here fall back to BACKING_TEXT_PROMPT above.
# Valid tags: intro, verse, chorus, bridge, outro, break, inst, solo

declare -A STRUCTURE_TAG_PROMPTS=(
    [intro]="Clean electric guitar arpeggios with reverb."
    [verse]="Female vocal with guitars and light drums."
    [chorus]="Full band with soaring guitar harmonies and stronger rhythm."
    [bridge]="Reflective instrumental passage with piano, synth pad, bass, and gentle percussion"
    [outro]="Guitars sustain while drums slowly fade."
    [break]="Minimal instrumental break with sparse percussion and ambient textures"
    [inst]="Instrumental section with expressive lead melody and full band accompaniment"
    [solo]="Virtuosic instrumental solo with dynamic expression and energy"
)
# Note: structure times and tags are read from the preset lyrics2melody JSONs
# (sample01_struct_time.json / sample01_struct_label.json), no need to set
# them manually.
###############################################################################
# you do not need to modify the following
# ── Resolve relative user paths to absolute (before any cd) ──────────────────
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: INPUT_FOLDER '$INPUT_FOLDER' does not exist." >&2
    exit 1
fi
INPUT_FOLDER="$(cd "$INPUT_FOLDER" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# ── Derived paths ─────────────────────────────────────────────────────────────
LYRIC_PATH="$OUTPUT_DIR/sample01.txt"
VOCAL_SAVE_DIR="$OUTPUT_DIR/vocal_audio"
MIDI_DIR="$OUTPUT_DIR/lyrics2melody"
BPM_PATH="$MIDI_DIR/sample01.json"
TIME_LYRIC_PATH="$MIDI_DIR/sample01_times.txt"
STRUCT_label_LIST="$MIDI_DIR/sample01_struct_label.json"
STRUCT_time_LIST="$MIDI_DIR/sample01_struct_time.json"
GENERATED_MIDI="$MIDI_DIR/sample01.mid"
CHORD_DIR="$MIDI_DIR/chord/$CHORD_STYLE"
SONG_NAME="sample01"
HARMONIZE_DIR="$OUTPUT_DIR/Harmonization_results"
CHORD_PATH="$HARMONIZE_DIR/btc_txt/${SONG_NAME}_chord_gen.txt"
PROMPT_LOG_PATH="$OUTPUT_DIR/Mixed_track/prompts_used.txt"

# ── 1. Load preset lyrics + melody (CSL-L2M skipped) ────────────────────────
mkdir -p "$MIDI_DIR"

SRC_LYRIC="$INPUT_FOLDER/sample01.txt"
SRC_MIDI_DIR="$INPUT_FOLDER/lyrics2melody"
REQUIRED_FILES=(
    "$SRC_LYRIC"
    "$SRC_MIDI_DIR/sample01.mid"
    "$SRC_MIDI_DIR/sample01.json"
    "$SRC_MIDI_DIR/sample01_times.txt"
    "$SRC_MIDI_DIR/sample01_struct_label.json"
    "$SRC_MIDI_DIR/sample01_struct_time.json"
)
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Error: required preset file not found: $f" >&2
        exit 1
    fi
done

echo "Copying preset lyrics + melody from $INPUT_FOLDER"
cp -f "$SRC_LYRIC" "$LYRIC_PATH"
cp -f "$SRC_MIDI_DIR/sample01.mid"                    "$GENERATED_MIDI"
cp -f "$SRC_MIDI_DIR/sample01.json"                   "$BPM_PATH"
cp -f "$SRC_MIDI_DIR/sample01_times.txt"              "$TIME_LYRIC_PATH"
cp -f "$SRC_MIDI_DIR/sample01_struct_label.json"      "$STRUCT_label_LIST"
cp -f "$SRC_MIDI_DIR/sample01_struct_time.json"       "$STRUCT_time_LIST"

# ── 2. FastSinger: MIDI → Vocal audio ───────────────────────────────────────
mkdir -p "$VOCAL_SAVE_DIR"
cd fastsinger
conda run --no-capture-output -n fastsinger \
    python inference.py \
    --model_id "$FASTSINGER_MODEL_ID" \
    --model_epoch "$FASTSINGER_MODEL_EPOCH" \
    --spkr_ref "$FASTSINGER_SPKR_REF" \
    --pitch_shifts "$FASTSINGER_PITCH_SHIFTS" \
    --shift_consonant_forward_alignment \
    --midi_path "$GENERATED_MIDI" \
    --lyric_path "$LYRIC_PATH" \
    --output_path "$VOCAL_SAVE_DIR/sample01.wav"

# ── 3. AccoMontage2: MIDI → Chord harmonization ─────────────────────────────
cd ../

python AccoMontage2/demo_SOME.py \
    --midi_path "$GENERATED_MIDI" \
    --output_dir "$HARMONIZE_DIR" \
    --beat_subdivision 1 \
    --chord_style "$CHORD_STYLE" \
    --chords_per_bar "$CHORDS_PER_BAR" \
    --key "$KEY"

# ── 4. MuseControlLite: Backing track generation ────────────────────────────
# Locate the vocal audio generated by FastSinger
VOCAL_AUDIO_PATH="$(find "$VOCAL_SAVE_DIR" -maxdepth 1 -name '*.wav' | head -1)"
if [ -z "$VOCAL_AUDIO_PATH" ]; then
    echo "Error: no .wav found in $VOCAL_SAVE_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/Mixed_track"

case "$MODE" in
  47s)
    INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_47s_scale_up.py"
    echo "Mode: 47s — loading model once, generating ${#BACKING_TEXT_PROMPTS[@]} prompt(s)"

    if python "$INFERENCE_SCRIPT" \
        --vocal_audio_file "$VOCAL_AUDIO_PATH" \
        --text_prompt "${BACKING_TEXT_PROMPTS[@]}" \
        --chord_file "$CHORD_PATH" \
        --vocal_midi_file "$GENERATED_MIDI" \
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

    # Read structure from preset JSON outputs (copied in step 1)
    readarray -t STRUCTURE_STARTS < <(python3 -c "import json; [print(x) for x in json.load(open('$STRUCT_time_LIST'))]")
    readarray -t STRUCTURE_TAGS   < <(python3 -c "import json; [print(x) for x in json.load(open('$STRUCT_label_LIST'))]")

    if [ ${#STRUCTURE_STARTS[@]} -eq 0 ]; then
        echo "Error: $STRUCT_time_LIST is empty or missing." >&2
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
        --vocal_midi_file "$GENERATED_MIDI" \
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

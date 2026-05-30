#!/bin/bash
###############################################################################
# Batch inference script for ComposeFlow (in-house / FastSinger).
#
# Reads lyrics and structure prompts from dataset JSON files and runs the full
# pipeline for each song ID.
#
# Usage:
#   ./inference_ComposeFlow_in_house_batch.sh 25 26 29
#   ./inference_ComposeFlow_in_house_batch.sh 25,26,29
###############################################################################

# ── Shared config ────────────────────────────────────────────────────────────
DATASET_DIR="/data/home/fundwotsai/MIDI-SAG_not_using/dataset"

# FastSinger voice config (shared across all songs)
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
# Musical key: C C# Db D ... B | Cm C#m ... Bm | auto
KEY="auto"
###############################################################################

# ── Parse IDs from arguments (supports "25 26 29" and "25,26,29") ────────────
IDS=()
for arg in "$@"; do
    IFS=',' read -ra parts <<< "$arg"
    IDS+=("${parts[@]}")
done

if [ ${#IDS[@]} -eq 0 ]; then
    echo "Usage: $0 <id1> <id2> ... or $0 <id1>,<id2>,..." >&2
    exit 1
fi

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Will process ${#IDS[@]} song(s): ${IDS[*]}"

for SONG_ID in "${IDS[@]}"; do
    echo ""
    echo "================================================================="
    echo "  Processing song ID: $SONG_ID"
    echo "================================================================="

    JSON_PATH="$DATASET_DIR/${SONG_ID}.json"
    if [ ! -f "$JSON_PATH" ]; then
        echo "Error: $JSON_PATH not found, skipping." >&2
        continue
    fi

    OUTPUT_DIR="$BASE_DIR/output_composerflow_demo/${SONG_ID}"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

    # ── Extract per-song config from JSON ────────────────────────────────
    # 1. Generate lyrics file from lyrics_zh
    ORIGINAL_LYRIC_PATH="$OUTPUT_DIR/lyrics_${SONG_ID}.txt"
    python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
for section, lines in data['lyrics_zh'].items():
    print(f'<{section}>')
    for item in lines:
        print(item['line'])
" "$JSON_PATH" > "$ORIGINAL_LYRIC_PATH"

    # 2. Extract vocal gender from metadata
    SINGER_GENDER=$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
print(data['metadata']['vocal_gender'].lower())
" "$JSON_PATH")

    # 3. Extract global_prompt as backing text prompt
    BACKING_TEXT_PROMPT=$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
print(data['global_prompt'])
" "$JSON_PATH")
    BACKING_TEXT_PROMPTS=( "$BACKING_TEXT_PROMPT" )

    # 4. Extract structure_en → STRUCTURE_TAG_PROMPTS (lowercase keys)
    unset STRUCTURE_TAG_PROMPTS
    declare -A STRUCTURE_TAG_PROMPTS
    while IFS=$'\t' read -r tag prompt; do
        STRUCTURE_TAG_PROMPTS["$tag"]="$prompt"
    done < <(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
for tag, prompt in data['structure_en'].items():
    print(f'{tag.lower()}\t{prompt}')
" "$JSON_PATH")

    echo "  Singer: $SINGER_GENDER"
    echo "  Backing prompt: $BACKING_TEXT_PROMPT"
    echo "  Structure tags: ${!STRUCTURE_TAG_PROMPTS[*]}"

    # ── Derived paths ────────────────────────────────────────────────────
    LYRIC_PATH="$OUTPUT_DIR/sample01.txt"
    VOCAL_SAVE_DIR="$OUTPUT_DIR/vocal_audio"
    MIDI_DIR="$OUTPUT_DIR/lyrics2melody"
    BPM_PATH="$MIDI_DIR/sample01.json"
    TIME_LYRIC_PATH="$MIDI_DIR/sample01_times.txt"
    STRUCT_label_LIST="$MIDI_DIR/sample01_struct_label.json"
    STRUCT_time_LIST="$MIDI_DIR/sample01_struct_time.json"
    GENERATED_MIDI="$MIDI_DIR/sample01.mid"
    SONG_NAME="sample01"
    HARMONIZE_DIR="$OUTPUT_DIR/Harmonization_results"
    CHORD_PATH="$HARMONIZE_DIR/btc_txt/${SONG_NAME}_chord_gen.txt"
    PROMPT_LOG_PATH="$OUTPUT_DIR/Mixed_track/prompts_used.txt"

    # ── 1. CSL-L2M: Lyrics → MIDI ───────────────────────────────────────
    mkdir -p "$MIDI_DIR"
    cd "$BASE_DIR/lyrics2melody_new"
    python -u generate.py \
        config/CSLL2M.yaml ../MIDI-SAG_checkpoints/pretrained_CSLL2M.pt \
        "$MIDI_DIR" 1 "$ORIGINAL_LYRIC_PATH"
    python -u read_midi.py "$GENERATED_MIDI" "$LYRIC_PATH"
    python -u pitch_picking.py "$GENERATED_MIDI" --bpm "$BPM_PATH" --singer "$SINGER_GENDER"
    python -u read_midi_lyrics_timestamp.py "$MIDI_DIR"
    python -u match_segment_times.py "$ORIGINAL_LYRIC_PATH" "$TIME_LYRIC_PATH" "$STRUCT_time_LIST" "$STRUCT_label_LIST"

    # ── 2. FastSinger: MIDI → Vocal audio ────────────────────────────────
    mkdir -p "$VOCAL_SAVE_DIR"
    cd "$BASE_DIR/fastsinger"
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

    # ── 3. AccoMontage2: MIDI → Chord harmonization ─────────────────────
    cd "$BASE_DIR"

    python AccoMontage2/demo_SOME.py \
        --midi_path "$GENERATED_MIDI" \
        --output_dir "$HARMONIZE_DIR" \
        --beat_subdivision 1 \
        --chord_style "$CHORD_STYLE" \
        --chords_per_bar "$CHORDS_PER_BAR" \
        --key "$KEY"

    # ── 4. MuseControlLite: Backing track generation ─────────────────────
    # Locate the vocal audio generated by FastSinger
    VOCAL_AUDIO_PATH="$(find "$VOCAL_SAVE_DIR" -maxdepth 1 -name '*.wav' | head -1)"
    if [ -z "$VOCAL_AUDIO_PATH" ]; then
        echo "Error: no .wav found in $VOCAL_SAVE_DIR for ID $SONG_ID, skipping." >&2
        continue
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

        # Read structure from pipeline JSON outputs (step 1)
        readarray -t STRUCTURE_STARTS < <(python3 -c "import json; [print(x) for x in json.load(open('$STRUCT_time_LIST'))]")
        readarray -t STRUCTURE_TAGS   < <(python3 -c "import json; [print(x) for x in json.load(open('$STRUCT_label_LIST'))]")

        if [ ${#STRUCTURE_STARTS[@]} -eq 0 ]; then
            echo "Error: $STRUCT_time_LIST is empty or missing for ID $SONG_ID, skipping." >&2
            continue
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

    echo "===== Done with song ID: $SONG_ID ====="
done

echo ""
echo "All done. Processed ${#IDS[@]} song(s): ${IDS[*]}"

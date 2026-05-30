#!/bin/bash
###############################################################################
# ── User config ──────────────────────────────────────────────────────────────
# Batch version of inference_gt_MIDI-SAG.sh.
#   Input MIDI:    share_singing/midi_shifted/<id>.mid
#   Input vocal:   share_singing/singing_voice/<id>.wav   (already generated)
#   Structure:     share_singing/structure_times.json     (3 starts per <id>)
#   Section text:  share_singing/aliged_real_music_sectioned_captions/<id>.json
# No singing-voice generation, VAD, beat-tracking, or MIDI transcription:
# the gt pipeline takes the vocal audio + MIDI directly.
MIDI_DIR="/data/home/fundwotsai/MIDI-SAG/share_singing/midi_real"
VOCAL_DIR="/data/home/fundwotsai/MIDI-SAG/share_singing/singing_voice_real_region"
STRUCTURE_TIMES_JSON="/data/home/fundwotsai/MIDI-SAG/share_singing/structure_times.json"
CAPTIONS_DIR="/data/home/fundwotsai/MIDI-SAG/share_singing/aliged_real_music_sectioned_captions"
OUTPUT_ROOT="./output_midi_sag_realsong_score"

# Structure tags are fixed: structure_times.json always provides 3 starts and
# the section captions always provide intro/verse/chorus prompts.
STRUCTURE_TAGS=( intro verse chorus )

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

# Fallback text prompt used when a structure tag has no per-song caption entry.
BACKING_TEXT_PROMPT="reflective instrumental pop with piano, synth pad, bass, and steady drums"

###############################################################################
# you do not need to modify the following

INFERENCE_SCRIPT="MuseControlLite/MuseControlLite_inference_continuation.py"

processed=0
skipped=0

shopt -s nullglob
for VOCAL_MIDI_PATH in "$MIDI_DIR"/*.mid; do
    SONG_NAME="$(basename "${VOCAL_MIDI_PATH%.*}")"
    VOCAL_AUDIO_PATH="$VOCAL_DIR/$SONG_NAME.wav"
    CAPTION_PATH="$CAPTIONS_DIR/$SONG_NAME.json"
    OUTPUT_DIR="$OUTPUT_ROOT/$SONG_NAME"
    CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
    PROMPT_LOG_PATH="$OUTPUT_DIR/Mixed_track/prompts_used.txt"

    echo "======================================================================"
    echo "Song: $SONG_NAME"
    echo "======================================================================"

    # ── Skip-on-missing guards ───────────────────────────────────────────────
    if [ ! -f "$VOCAL_AUDIO_PATH" ]; then
        echo "[WARN] No vocal audio for '$SONG_NAME' at $VOCAL_AUDIO_PATH; skipping." >&2
        skipped=$((skipped + 1))
        continue
    fi
    if [ ! -f "$CAPTION_PATH" ]; then
        echo "[WARN] No section caption for '$SONG_NAME' at $CAPTION_PATH; skipping." >&2
        skipped=$((skipped + 1))
        continue
    fi
    if ! python3 -c "import json,sys; d=json.load(open('$STRUCTURE_TIMES_JSON')); sys.exit(0 if '$SONG_NAME' in d else 1)"; then
        echo "[WARN] No structure_times entry for '$SONG_NAME'; skipping." >&2
        skipped=$((skipped + 1))
        continue
    fi

    # ── Read per-song structure starts and section prompts ───────────────────
    readarray -t STRUCTURE_STARTS < <(python3 -c "import json;print('\n'.join(str(x) for x in json.load(open('$STRUCTURE_TIMES_JSON'))['$SONG_NAME']))")
    readarray -t STRUCTURE_PROMPTS < <(python3 -c "import json;s=json.load(open('$CAPTION_PATH'))['sections'];print('\n'.join([s['intro'],s['verse'],s['chorus']]))")

    if [ ${#STRUCTURE_STARTS[@]} -ne ${#STRUCTURE_TAGS[@]} ]; then
        echo "[WARN] '$SONG_NAME': got ${#STRUCTURE_STARTS[@]} structure starts, expected ${#STRUCTURE_TAGS[@]}; skipping." >&2
        skipped=$((skipped + 1))
        continue
    fi

    # Fall back to BACKING_TEXT_PROMPT for any empty caption section.
    for i in "${!STRUCTURE_PROMPTS[@]}"; do
        if [ -z "${STRUCTURE_PROMPTS[$i]}" ]; then
            echo "[WARN] '$SONG_NAME': empty prompt for tag '${STRUCTURE_TAGS[$i]}'; using BACKING_TEXT_PROMPT" >&2
            STRUCTURE_PROMPTS[$i]="$BACKING_TEXT_PROMPT"
        fi
    done

    # ── 1. Melody harmonization ──────────────────────────────────────────────
    python AccoMontage2/demo_SOME.py \
        --midi_path "$VOCAL_MIDI_PATH" \
        --output_dir "$OUTPUT_DIR/Harmonization_results" \
        --beat_subdivision 1 \
        --chord_style "$CHORD_STYLE" \
        --chords_per_bar "$CHORDS_PER_BAR" \
        --key "$KEY"

    # ── 2. Backing track generation (full_song continuation) ─────────────────
    mkdir -p "$OUTPUT_DIR/Mixed_track"

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
        processed=$((processed + 1))
    else
        echo "[WARN] '$SONG_NAME': inference failed; continuing." >&2
        skipped=$((skipped + 1))
    fi
done
shopt -u nullglob

echo "======================================================================"
echo "Batch complete: processed=$processed  skipped=$skipped"
echo "======================================================================"

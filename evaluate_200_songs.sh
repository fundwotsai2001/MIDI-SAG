#!/bin/bash

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT="/volume/nas-fundwo-storage/fundwo-test/lyrics2melody/generated_midi_ICML_rebuttal_pitch_shift_v3"
OUTPUT_DIR="./output_evaluate_200"
MUSECONTROLLITE_CHECKPOINT="./MIDI-SAG_checkpoints/checkpoint_gt_beat"
CHORD_STYLE="POP_COMPLEX"
TOTAL_SONGS=200

CONDA_ROOT="/volume/nas-fundwo-storage/fundwo-test/miniconda3"
MIDI_SAG_PYTHON="$CONDA_ROOT/envs/midi-sag/bin/python"

mkdir -p "$OUTPUT_DIR"

# ── Steps 1–3: Pre-processing loop over 200 songs ────────────────────────────
for song_id in $(seq 0 $((TOTAL_SONGS - 1))); do
    VOCAL_AUDIO_PATH="$DATA_ROOT/lyrics_${song_id}/sample01_shifted.wav"

    if [ ! -f "$VOCAL_AUDIO_PATH" ]; then
        echo "[skip] lyrics_${song_id}: vocal not found"
        continue
    fi

    SONG_NAME="lyrics_${song_id}"
    VOCAL_MIDI_PATH="$OUTPUT_DIR/vocal_MIDI/${SONG_NAME}.mid"
    CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
    BEAT_PATH="$OUTPUT_DIR/vocal_beat/${SONG_NAME}/sample01_shifted_beat_times.txt"

    echo ""
    echo "── [${song_id}/${TOTAL_SONGS}] Processing ${SONG_NAME} ──────────────"

    # # 1. VAD: trim leading silence (fall back to original vocal on failure)
    # mkdir -p "$OUTPUT_DIR/vad_audio"
    # if "$MIDI_SAG_PYTHON" vad_trim.py "$VOCAL_AUDIO_PATH" "$OUTPUT_DIR/vad_audio/${SONG_NAME}.wav"; then
    #     VAD_AUDIO="$OUTPUT_DIR/vad_audio/${SONG_NAME}.wav"
    # else
    #     echo "[warn] lyrics_${song_id}: VAD failed, using original vocal"
    #     VAD_AUDIO="$VOCAL_AUDIO_PATH"
    # fi

    AUDIO_BASE="$(basename "${VOCAL_AUDIO_PATH%.*}")"   # → sample01_shifted

    # 2. Vocal beat tracking
    "$MIDI_SAG_PYTHON" Singing-Vocal-Beat-Tracking/inference_vad.py \
        --audio_path "$VOCAL_AUDIO_PATH" \
        --model_path MIDI-SAG_checkpoints/model-16.pt \
        --use_vad --fill_silence \
        --output_dir "$OUTPUT_DIR/vocal_beat" \
        || echo "[warn] lyrics_${song_id}: beat tracking failed"
    # Rename beat dir from sample01_shifted → lyrics_{id}
    if [ -d "$OUTPUT_DIR/vocal_beat/${AUDIO_BASE}" ]; then
        mv "$OUTPUT_DIR/vocal_beat/${AUDIO_BASE}" "$OUTPUT_DIR/vocal_beat/${SONG_NAME}"
    fi

    # 3. Vocal MIDI transcription (skip if input file missing)
    if [ -f "$VOCAL_AUDIO_PATH" ]; then
        "$MIDI_SAG_PYTHON" GAME/infer.py extract "$VOCAL_AUDIO_PATH" \
            -m GAME/GAME-1.0-medium/model.pt \
            --output-dir "$OUTPUT_DIR/vocal_MIDI" \
            || echo "[warn] lyrics_${song_id}: GAME transcription failed"
        # Rename MIDI from sample01_shifted.mid → lyrics_{id}.mid
        if [ -f "$OUTPUT_DIR/vocal_MIDI/${AUDIO_BASE}.mid" ]; then
            mv "$OUTPUT_DIR/vocal_MIDI/${AUDIO_BASE}.mid" "$VOCAL_MIDI_PATH"
        fi
    else
        echo "[warn] lyrics_${song_id}: skipping GAME, audio not found"
    fi

    # 4. Melody harmonization (skip if MIDI or beat file missing)
    if [ -f "$VOCAL_MIDI_PATH" ] && [ -f "$BEAT_PATH" ]; then
        "$MIDI_SAG_PYTHON" AccoMontage2/demo_SOME.py \
            --midi_path "$VOCAL_MIDI_PATH" \
            --beat_file "$BEAT_PATH" \
            --output_dir "$OUTPUT_DIR/Harmonization_results" \
            --beat_subdivision 1 \
            --chord_style "$CHORD_STYLE" \
            || echo "[warn] lyrics_${song_id}: harmonization failed"
        # Rename chord from lyrics_{id}_chord_gen.txt (MIDI stem) to the expected name
        SRC_CHORD="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
        if [ -f "$SRC_CHORD" ] && [ "$SRC_CHORD" != "$CHORD_PATH" ]; then
            mv "$SRC_CHORD" "$CHORD_PATH"
        fi
    else
        echo "[warn] lyrics_${song_id}: skipping harmonization (MIDI or beat file missing)"
    fi

done

# ── Step 5: Backing track generation + evaluation (all 200 songs) ─────────────
"$MIDI_SAG_PYTHON" MuseControlLite/evaluate_midi_sag_detected.py \
    --gpu_id 0 \
    --pipeline_output_dir "$OUTPUT_DIR"

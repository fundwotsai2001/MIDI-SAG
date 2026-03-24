# ── User config ──────────────────────────────────────────────────────────────
LYRIC_PATH="/data/home/fundwotsai/MIDI-SAG/lyrics_0.txt"
OUTPUT_DIR="/data/home/fundwotsai/MIDI-SAG/output_compose"
SOULX_PROMPT_WAV="example/audio/zh_prompt.mp3"
SOULX_PROMPT_META="example/audio/zh_prompt.json"
SOULX_SAVE_DIR="/data/home/fundwotsai/MIDI-SAG/output_compose/vocal_audio"
LANGUAGE="Mandarin"
CHORD_TYPE="pop_complex" # choices: "pop_standard", "pop_complex", "r&b", "dark", "None"
BACKING_TEXT_PROMPT="piano and drums, in the style of pop music"



# ── Derived paths ─────────────────────────────────────────────────────────────
MIDI_DIR="$OUTPUT_DIR/lyrics2melody"
BPM_PATH="$MIDI_DIR/sample01.json"
TIME_LYRIC_PATH="$MIDI_DIR/sample01_times.txt"
STRUCT_label_LIST="$MIDI_DIR/sample01_struct_label.json"
STRUCT_time_LIST="$MIDI_DIR/sample01_struct_time.json"
GENERATED_MIDI="$MIDI_DIR/sample01.mid"
TARGET_META="$MIDI_DIR/sample01_soulx.json"
CHORD_DIR="$MIDI_DIR/chord/$CHORD_TYPE"

# ── 1. CSL-L2M: Lyrics → MIDI ─────────────────────────────────────────────────
mkdir -p "$MIDI_DIR"
cd /data/home/fundwotsai/MIDI-SAG/lyrics2melody
python -u generate.py \
    config/CSLL2M.yaml pretrained_CSLL2M.pt \
    "$MIDI_DIR" 1 "$LYRIC_PATH"
python -u read_midi.py $MIDI_PATH $LYRIC_PATH
python -u  pitch_picking.py $MIDI_PATH --bpm $BPM_PATH
python -u read_midi_lyrics_timestamp.py $MIDI_DIR
python -u match_segment_times.py $LYRIC_PATH $TIME_LYRIC_PATH $STRUCT_time_LIST $STRUCT_label_LIST

# ── 2. Convert generated MIDI → SoulX-Singer JSON ─────────────────────────────
cd /data/home/fundwotsai/MIDI-SAG/SoulX-Singer
python midi2json.py "$GENERATED_MIDI" "$TARGET_META" --language "$LANGUAGE"

# ── 3. SoulX-Singer: MIDI + JSON → Vocal audio ────────────────────────────────
python -m cli.inference \
    --device cuda \
    --model_path pretrained_models/SoulX-Singer/model.pt \
    --config soulxsinger/config/soulxsinger.yaml \
    --prompt_wav_path "$SOULX_PROMPT_WAV" \
    --prompt_metadata_path "$SOULX_PROMPT_META" \
    --target_metadata_path "$TARGET_META" \
    --phoneset_path soulxsinger/utils/phoneme/phone_set.json \
    --save_dir "$SOULX_SAVE_DIR" \
    --control score \
    --auto_shift \
    --pitch_shift 0
# ── 4. Accomontage2 ────────────────────────────────
cd AccoMontage2
python demo.py "$MIDI_DIR" "$CHORD_DIR" "$CHORD_TYPE" 
python acc2btc.py "$CHORD_DIR/chord_txt" "$CHORD_DIR/chord_btc_txt"
# ── 5. MuseControlLite: Vocal audio + Chord progression + beat → Backing track ────────────────────────────────
python MuseControlLite/MuseControlLite_inference_47s.py \
    --vocal_audio_file "$VOCAL_AUDIO_PATH" \
    --text_prompt "$BACKING_TEXT_PROMPT" \
    --chord_file "$CHORD_PATH" \
    --vocal_beat_file "$BEAT_PATH" \
    --checkpoint_path "$MUSECONTROLLITE_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR/Backing_track"
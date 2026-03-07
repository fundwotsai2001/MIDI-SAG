# Paste your vocal audio path here (absolute or relative)
VOCAL_AUDIO_PATH="./vocal_audio_test_data/361.wav"
# Output folder name (change this to your desired output directory)
OUTPUT_DIR="./output_game"
# text command for backing track generation (can be modified as needed)
BACKING_TEXT_PROMPT="Jazzy backing music, high quality, with bass guitar, drums, piano"
# For internal use, test checkpoints using ground truth beat timings or vocal beats.
MUSECONTROLLITE_CHECKPOINT="./MIDI-SAG_checkpoints/checkpoint_gt_beat"
# Automatically extract song name from the audio path (strips directory and extension)
SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"

VOCAL_MIDI_PATH="$OUTPUT_DIR/vocal_MIDI/$SONG_NAME.mid"
CHORD_PATH="$OUTPUT_DIR/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
BEAT_PATH="$OUTPUT_DIR/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times.txt"

MIDI_FOLDER="/data/home/fundwotsai/MIDI-SAG/lyrics_200"
ORIGINAL_LYRIC_PATH="/data/home/fundwotsai/MIDI-SAG/lyrics_200/lyrics.txt"

# # vocal beat tracking with VAD
python Singing-Vocal-Beat-Tracking/inference_vad.py --audio_path $VOCAL_AUDIO_PATH --model_path MIDI-SAG_checkpoints/model-16.pt --use_vad --output_dir $OUTPUT_DIR/vocal_beat --fill_silence
# GAME: vocal MIDI transcription
python GAME/infer.py extract $VOCAL_AUDIO_PATH -m GAME/GAME-1.0-medium/model.pt --output-dir $OUTPUT_DIR/vocal_MIDI
# Accomotage2: melody harmonization
python AccoMontage2/demo_SOME.py --midi_path $VOCAL_MIDI_PATH --beat_file $BEAT_PATH --output_dir $OUTPUT_DIR/Harmonization_results --beat_subdivision 1
# MuseControlLite: Backing track generation 
python MuseControlLite/MuseControlLite_inference_47s.py --vocal_audio_file $VOCAL_AUDIO_PATH --text_prompt "$BACKING_TEXT_PROMPT" --chord_file $CHORD_PATH --vocal_beat_file $BEAT_PATH --checkpoint_path $MUSECONTROLLITE_CHECKPOINT --output_dir $OUTPUT_DIR/Backing_track

# cd lyrics2melody

# python -u generate.py \
#     config/CSLL2M.yaml pretrained_CSLL2M.pt \
#     $MIDI_FOLDER 1 $ORIGINAL_LYRIC_PATH
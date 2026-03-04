# Paste your vocal audio path here (absolute or relative)
VOCAL_AUDIO_PATH="vocal_audio_test_data/chengron.wav"
# text command for backing track generation (can be modified as needed)
BACKING_TEXT_PROMPT="rock backing music, high quality, with bass guitar, drums, synthesizer"

# Automatically extract song name from the audio path (strips directory and extension)
SONG_NAME="$(basename "${VOCAL_AUDIO_PATH%.*}")"

VOCAL_MIDI_PATH="output/vocal_MIDI/$SONG_NAME.mid"
CHORD_PATH="output/Harmonization_results/btc_txt/${SONG_NAME}_chord_gen.txt"
BEAT_PATH="output/vocal_beat/$SONG_NAME/${SONG_NAME}_beat_times.txt"



# vocal beat tracking with VAD
python Singing-Vocal-Beat-Tracking/inference_vad.py --audio_path $VOCAL_AUDIO_PATH --model_path MIDI-SAG_checkpoints/model-16.pt --use_vad --output_dir ./output/vocal_beat
# SOME: vocal MIDI transcription
python SOME/infer.py --model SOME/pretrained/0119_continuous256_5spk/model_ckpt_steps_100000_simplified.ckpt --wav $VOCAL_AUDIO_PATH --midi $VOCAL_MIDI_PATH
# Accomotage2: melody harmonization
python AccoMontage2/demo_SOME.py --midi_path $VOCAL_MIDI_PATH --beat_file $BEAT_PATH --output_dir ./output/Harmonization_results --beat_subdivision 1
# MuseControlLite: Backing track generation 
python MuseControlLite/MuseControlLite_inference.py --vocal_audio_file $VOCAL_AUDIO_PATH --text_prompt "$BACKING_TEXT_PROMPT" --chord_file $CHORD_PATH --vocal_beat_file $BEAT_PATH 
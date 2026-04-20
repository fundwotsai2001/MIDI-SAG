# MIDI-SAG

This is the official implementation of MIDI-SAG.
[paper](https://arxiv.org/abs/2602.22029) | [demo](https://composerflow.github.io/web/)


## Installation
We provide a step by step series of examples that tell you how to get a development environment running.
```
git clone https://github.com/fundwotsai2001/MIDI-SAG.git
cd MIDI-SAG
sudo apt-get install portaudio19-dev


## Install environment
conda create -n midi-sag python=3.9
conda activate midi-sag
# This will install all dependecies and checkpoints
./install.sh

```
## huggingface-cli login
You will need a token generated from [huggingface](https://huggingface.co/settings/tokens).
```
huggingface-cli login
```
## Inference
There are two files for inference `inference_detected_MIDI-SAG.sh.sh` and `inference_gt_MIDI-SAG.sh.sh`. 

```
# when ground truth vocal MIDI is present
./inference_detected_MIDI-SAG.sh.sh 
# You should specify VOCAL_AUDIO_PATH, MODE, CHORD_STYLE, BACKING_TEXT_PROMPTS, explainations are provided in the shell script

# when only vocal audio is present
./inference_gt_MIDI-SAG.sh.sh 
# You should specify VOCAL_MIDI_PATH, VOCAL_AUDIO_PATH, MODE, CHORD_STYLE, BACKING_TEXT_PROMPTS # explainations are provided in the shell script
```

The output directory will be save in the following format:

```
${OUTPUT_DIR}/
├── Backing_track/
│   ├── prompts_used.txt
│   └── text_7.0_con_1.5_rhythm_melody_structure_chord_audio_0.3_500/
│       ├── mixed_${SONG_NAME}_full.wav
│       └── mixed_${SONG_NAME}_structure_prompts.json
├── Harmonization_results/
│   ├── btc_txt/
│   │   └── ${SONG_NAME}_chord_gen.txt
│   └── chord_gen_filled_empty/
│       └── ${SONG_NAME}_chord_gen_filled_empty_bars.mid
├── vocal_beat/
│   └── ${SONG_NAME}/
│       ├── ${SONG_NAME}_beat_times.txt
│       ├── ${SONG_NAME}_beat_times_detected_only.txt
│       ├── ${SONG_NAME}_downbeat_times.txt
│       └── ${SONG_NAME}_with_metronome.wav
└── vocal_MIDI/
    └── ${SONG_NAME}.mid
```



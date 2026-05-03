# <img src="lyrics.png" alt="" width="40" align="left" />&nbsp;MIDI-SAG: MIDI-Informed Singing Accompaniment Generation

Generate full-song backing tracks conditioned on vocal melody, with structure-aware continuation and controllable chord styles. Additionally, we proposed a lyrics-to-song pipeline.

[![arXiv](https://img.shields.io/badge/arXiv-2602.22029-b31b1b.svg)](https://arxiv.org/abs/2602.22029)
[![Demo](https://img.shields.io/badge/Demo-Page-blue)](https://composerflow.github.io/web/)

## Overview

MIDI-SAG is a compositional pipeline that generates backing tracks (piano, bass, drums, synth, etc.) conditioned on vocal melody MIDI. It supports structure-aware full-song generation by chaining beat tracking, MIDI transcription, chord harmonization, and a MuseControlLite adapter on Stable Audio Open 1.0.

Three usage modes are available:

- **Detected** -- input is vocal audio only. The pipeline auto-detects beats, transcribes vocal MIDI via GAME, harmonizes chords with AccoMontage2, and generates the backing track.
- **Ground-truth** -- input is vocal audio + a pre-existing vocal MIDI. Skips beat tracking and transcription; uses the provided MIDI directly for harmonization.
- **ComposeFlow** -- input is a lyrics text file. End-to-end pipeline from text to full song: lyrics-to-melody (CSL-L2M), melody-to-singing-voice (SoulX-Singer), then harmonization and backing track generation. You can use the voice prompt from the ./example_input folder or use your own audio files (shorter than 10 seconds). **Note that the open source version isn't perfect, since the "Inference support for MIDI-based input" of SoulX-Singer is our implementation.**

## Pipeline

```
Detected mode:

  Vocal audio ─► Beat Tracking ─► GAME (MIDI) ─► AccoMontage2 (chords) ─► MuseControlLite (backing track)

Ground-truth mode:

  Vocal audio ─┐
               ├─► AccoMontage2 (chords) ─► MuseControlLite (backing track)
  Vocal MIDI  ─┘

ComposeFlow mode:

  Lyrics ─► CSL-L2M (melody MIDI) ─► SoulX-Singer (vocal audio) ─► AccoMontage2 (chords) ─► MuseControlLite (backing track)
```

## Installation

```bash
git clone https://github.com/fundwotsai2001/MIDI-SAG.git
cd MIDI-SAG
sudo apt-get install portaudio19-dev

# Create environment
conda create -n midi-sag python=3.9
conda activate midi-sag

# Install all dependencies and download checkpoints
./install.sh
```

## Hugging Face Login

You will need a token generated from [Hugging Face](https://huggingface.co/settings/tokens).

```bash
huggingface-cli login
```

## Inference

Three scripts correspond to the three modes:

| Script | Mode | Required inputs |
|--------|------|-----------------|
| `inference_detected_MIDI-SAG.sh` | Detected | Vocal audio |
| `inference_gt_MIDI-SAG.sh` | Ground-truth | Vocal audio + vocal MIDI |
| `inference_ComposeFlow.sh` | ComposeFlow | Lyrics text file |

```bash
# Detected: only vocal audio is available
./inference_detected_MIDI-SAG.sh

# Ground-truth: vocal audio and ground-truth MIDI are both available
./inference_gt_MIDI-SAG.sh

# ComposeFlow: generate everything from lyrics
./inference_ComposeFlow.sh
```

Edit the config section at the top of each script before running.

### Common parameters (all three scripts)

| Parameter | Values | Description |
|-----------|--------|-------------|
| `MODE` | `47s`, `full_song` | Single-segment (≤47 s) or structure-aware full-song generation |
| `MUSECONTROLLITE_CHECKPOINT` | path | MuseControlLite checkpoint directory |
| `CHORD_STYLE` | `POP_STANDARD`, `POP_COMPLEX`, `DARK`, `RANDB`, `NOCONSTRAINT` | Chord vocabulary constraint for harmonization |
| `CHORDS_PER_BAR` | `1`, `2` | Bar-level or half-bar chord resolution (half-bar uses the next bar's chord for the 2nd half) |
| `KEY` | `C`, `Cm`, `auto`, ... | Musical key. `auto` reads `key_signature` from the MIDI and falls back to a Bellman-Budge heuristic |
| `BACKING_TEXT_PROMPT` | string | Default text prompt; also used as a fallback in `full_song` mode for tags missing from `STRUCTURE_TAG_PROMPTS` |
| `BACKING_TEXT_PROMPTS` | array | (47s mode) One generation per entry, all under a single model load |
| `STRUCTURE_TAG_PROMPTS` | assoc. array | (full_song mode) Per-tag prompts. Valid tags: `intro`, `verse`, `chorus`, `bridge`, `outro`, `break`, `inst`, `solo` |
| `STRUCTURE_STARTS` / `STRUCTURE_TAGS` | arrays | (full_song mode) Section boundary timestamps and labels. Each gap and the trailing segment must be < 47 s |

### Detected-mode only (`inference_detected_MIDI-SAG.sh`)

| Parameter | Values | Description |
|-----------|--------|-------------|
| `VOCAL_AUDIO_PATH` | path | Vocal audio input |
| `DOWNBEAT_PHASE` | int or `None` | Index into the detected-only beat file selecting bar 1, beat 1. `None` lets AccoMontage2 detect the downbeat phase automatically (sometimes inaccurate) |
| `MIN_BPM` / `MAX_BPM` | int (empty for default) | DBN beat-tracker BPM bounds passed to `Singing-Vocal-Beat-Tracking/inference_vad.py` |

### Ground-truth-mode only (`inference_gt_MIDI-SAG.sh`)

| Parameter | Values | Description |
|-----------|--------|-------------|
| `VOCAL_AUDIO_PATH` | path | Vocal audio input |
| `VOCAL_MIDI_PATH` | path | Pre-existing vocal MIDI (skips beat tracking + transcription) |

### ComposeFlow-only (`inference_ComposeFlow.sh`)

| Parameter | Values | Description |
|-----------|--------|-------------|
| `ORIGINAL_LYRIC_PATH` | path | Lyrics file (one phrase per line; blank lines separate sections) |
| `SINGER_GENDER` | `male`, `female` | Used by pitch picking to select the vocal range |
| `SOULX_PROMPT_WAV` | path | Reference voice clip for SoulX-Singer (must be < 10 s). Examples are in `./example_input/` |
| `PROMPT_LANGUAGE` | `English`, `Chinese`, `Cantonese` | Language of the voice prompt |
| `VOCAL_SEP` | `True`, `False` | Set `True` if the prompt WAV contains backing music that needs to be separated |

> **Note (ComposeFlow)**: `STRUCTURE_STARTS` and `STRUCTURE_TAGS` are **auto-derived** from the lyrics→melody pipeline — do not set them by hand. Only `STRUCTURE_TAG_PROMPTS` need editing for `full_song` mode.

## Output Structure

### Detected / Ground-truth modes

```
${OUTPUT_DIR}/
├── Mixed_track/
│   ├── prompts_used.txt
│   └── text_<gs_text>_con_<gs_con>.../
│       ├── mixed_${SONG_NAME}_full.wav
│       └── mixed_${SONG_NAME}_structure_prompts.json   # full_song mode
├── Harmonization_results/
│   ├── btc_txt/
│   │   └── ${SONG_NAME}_chord_gen.txt
│   └── chord_gen_filled_empty/
│       └── ${SONG_NAME}_chord_gen_filled_empty_bars.mid
├── vad_audio/                                           # detected mode, 47s only
│   └── ${SONG_NAME}.wav
├── vocal_beat/                                          # detected mode only
│   └── ${SONG_NAME}/
│       ├── ${SONG_NAME}_beat_times.txt
│       ├── ${SONG_NAME}_beat_times_detected_only.txt
│       ├── ${SONG_NAME}_downbeat_times.txt
│       └── ${SONG_NAME}_with_metronome.wav
└── vocal_MIDI/                                          # detected mode only
    └── ${SONG_NAME}.mid
```

### ComposeFlow mode

```
${OUTPUT_DIR}/
├── Mixed_track/                       # backing track + prompts_used.txt
├── Harmonization_results/             # AccoMontage2 chord output
├── lyrics2melody/                     # CSL-L2M outputs
│   ├── sample01.mid
│   ├── sample01.json                  # BPM
│   ├── sample01_times.txt             # lyric timestamps
│   ├── sample01_struct_time.json      # auto-derived STRUCTURE_STARTS
│   ├── sample01_struct_label.json     # auto-derived STRUCTURE_TAGS
│   └── sample01_soulx.json            # SoulX-Singer target metadata
├── prompt_transcription/              # preprocessed voice-prompt metadata
│   └── metadata.json
└── vocal_audio/                       # SoulX-Singer synthesized vocal
    └── *.wav
```

## Citation

```bibtex
@article{tsai2026midisag,
  title={{MIDI}-Informed Singing Accompaniment Generation in a Compositional Song Pipeline},
  author={Tsai, Fang-Duo and Lai, Yi-An and Chen, Fei-Yueh and Fu, Hsueh-Wei and Chai, Li and Lee, Wei-Jaw and Cheng, Hao-Chung and Yang, Yi-Hsuan},
  journal={arXiv preprint arXiv:2602.22029},
  year={2026}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). See the [`LICENSE`](LICENSE) file for the full text.

Note that bundled third-party components (AccoMontage2, GAME, SoulX-Singer, CSL-L2M, Stable Audio Open, Singing-Vocal-Beat-Tracking) retain their original licenses; please consult each upstream repository before redistribution or commercial use.

## Acknowledgements

This project builds on the following open-source projects:

- [AccoMontage2](https://github.com/zhaojw1998/AccoMontage) -- melody harmonization (Li Yi et al., Music-X-Lab, NYU Shanghai)
- [GAME](https://github.com/openvpi/GAME) -- vocal-to-MIDI transcription (openvpi)
- [SoulX-Singer](https://github.com/nicollassistant/SoulX-Singer) -- zero-shot singing voice synthesis (Soul AILab)
- [CSL-L2M](https://lichaiustc.github.io/CSL-L2M/) -- lyrics-to-melody generation
- [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) -- base audio generation model (Stability AI)
- [Singing-Vocal-Beat-Tracking](https://github.com/SingingVocalBeatTracking) -- singing beat detection

# MIDI-SAG: MIDI-Informed Singing Accompaniment Generation

Generate full-song backing tracks conditioned on vocal melody, with structure-aware continuation and controllable chord styles.

[![arXiv](https://img.shields.io/badge/arXiv-2602.22029-b31b1b.svg)](https://arxiv.org/abs/2602.22029)
[![Demo](https://img.shields.io/badge/Demo-Page-blue)](https://composerflow.github.io/web/)

## Overview

MIDI-SAG is a compositional pipeline that generates backing tracks (piano, bass, drums, synth, etc.) conditioned on vocal melody MIDI. It supports structure-aware full-song generation by chaining beat tracking, MIDI transcription, chord harmonization, and a MuseControlLite adapter on Stable Audio Open 1.0.

Three usage modes are available:

- **Detected** -- input is vocal audio only. The pipeline auto-detects beats, transcribes vocal MIDI via GAME, harmonizes chords with AccoMontage2, and generates the backing track.
- **Ground-truth** -- input is vocal audio + a pre-existing vocal MIDI. Skips beat tracking and transcription; uses the provided MIDI directly for harmonization.
- **ComposeFlow** -- input is a lyrics text file. End-to-end pipeline from text to full song: lyrics-to-melody (CSL-L2M), melody-to-singing-voice (SoulX-Singer), then harmonization and backing track generation.

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

Edit the config section at the top of each script before running. Key parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `MODE` | `47s`, `full_song` | Single-segment or structure-aware full-song generation |
| `CHORD_STYLE` | `POP_STANDARD`, `POP_COMPLEX`, `DARK`, `RANDB`, `NOCONSTRAINT` | Chord vocabulary constraint for harmonization |
| `CHORDS_PER_BAR` | `1`, `2` | Bar-level or half-bar chord resolution |
| `KEY` | `C`, `Cm`, `auto`, ... | Musical key (`auto` for automatic detection) |
| `STRUCTURE_TAG_PROMPTS` | associative array | Per-section text prompts (intro, verse, chorus, ...) for `full_song` mode |
| `STRUCTURE_STARTS` / `STRUCTURE_TAGS` | arrays | Section boundary timestamps and labels |

## Output Structure

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

## Citation

```bibtex
@article{tsai2026midisag,
  title={{MIDI}-Informed Singing Accompaniment Generation in a Compositional Song Pipeline},
  author={Tsai, Fang-Duo and Lai, Yi-An and Chen, Fei-Yueh and Fu, Hsueh-Wei and Chai, Li and Lee, Wei-Jaw and Cheng, Hao-Chung and Yang, Yi-Hsuan},
  journal={arXiv preprint arXiv:2602.22029},
  year={2026}
}
```

## Acknowledgements

This project builds on the following open-source projects:

- [AccoMontage2](https://github.com/zhaojw1998/AccoMontage) -- melody harmonization (Li Yi et al., Music-X-Lab, NYU Shanghai)
- [GAME](https://github.com/openvpi/GAME) -- vocal-to-MIDI transcription (openvpi)
- [SoulX-Singer](https://github.com/nicollassistant/SoulX-Singer) -- zero-shot singing voice synthesis (Soul AILab)
- [CSL-L2M](https://lichaiustc.github.io/CSL-L2M/) -- lyrics-to-melody generation
- [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) -- base audio generation model (Stability AI)
- [Singing-Vocal-Beat-Tracking](https://github.com/SingingVocalBeatTracking) -- singing beat detection

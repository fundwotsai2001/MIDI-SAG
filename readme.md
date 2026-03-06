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
```
./inference.sh
```



#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
root_dir=$(dirname "$script_dir")

cd $root_dir || exit
export PYTHONPATH=$root_dir:$PYTHONPATH

model_path=pretrained_models/SoulX-Singer/model.pt
config=soulxsinger/config/soulxsinger.yaml
prompt_wav_path=example/audio/en_prompt.mp3
prompt_metadata_path=example/audio/en_prompt.json
target_metadata_path=example/audio/sample01.json
phoneset_path=soulxsinger/utils/phoneme/phone_set.json
save_dir=example/generated/sample01
control=score  # melody or score

python -m cli.inference \
    --device cuda \
    --model_path $model_path \
    --config $config \
    --prompt_wav_path $prompt_wav_path \
    --prompt_metadata_path $prompt_metadata_path \
    --target_metadata_path $target_metadata_path \
    --phoneset_path $phoneset_path \
    --save_dir $save_dir \
    --control $control \
    --auto_shift \
    --pitch_shift 0
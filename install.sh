conda install cmake -y
pip install "pip<24.1"
pip install torch==2.5.1 
pip install torchaudio==2.5.1
pip install pytorch-fast-transformers==0.4.0 --no-build-isolation
pip install pyaudio
pip install -r requirements.txt
pip install torchcodec
pip install chorder
pip install miditok
pip install -U huggingface_hub

# Download the SoulX-Singer SVS model
hf download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer
# Download models required for preprocessing
hf download Soul-AILab/SoulX-Singer-Preprocess --local-dir pretrained_models/SoulX-Singer-Preprocess

# wget https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip -O /tmp/some_model.zip                                                         
# unzip /tmp/some_model.zip -d SOME/pretrained/                                                                                                    
# rm /tmp/some_model.zip 
# wget https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip -O /tmp/RMVPE.zip      
# unzip /tmp/RMVPE.zip -d SOME/pretrained/                                                                                                    
# rm /tmp/RMVPE.zip  


# wget https://github.com/openvpi/GAME/releases/download/v1.0.0/GAME-1.0-medium.zip -O /tmp/game_model.zip                                                         
# unzip /tmp/game_model.zip -d GAME/  
# gdown 1o6eUCYwUcIzZeqEycar6AHTSQ-vYcL_q --folder

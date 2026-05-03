conda install cmake -y
pip install "pip<24.1"
pip install torch==2.5.1 torchaudio==2.5.1                                                                                                                                               
pip install pytorch-fast-transformers==0.4.0 --no-build-isolation
pip install pyaudio
pip install -r requirements.txt
pip install torchcodec
pip install chorder
pip install miditok
pip install "huggingface-hub>=0.24.0,<1.0"
pip install nemo_toolkit[asr] 
pip install openai-whisper 
# # Download the SoulX-Singer SVS model
hf download Soul-AILab/SoulX-Singer --local-dir MIDI-SAG_checkpoints/SoulX-Singer
Download models required for preprocessing
hf download Soul-AILab/SoulX-Singer-Preprocess --local-dir MIDI-SAG_checkpoints/SoulX-Singer-Preprocess

                                                                                                 
wget https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip -O /tmp/RMVPE.zip      
unzip /tmp/RMVPE.zip                                                                                            
rm /tmp/RMVPE.zip  


wget https://github.com/openvpi/GAME/releases/download/v1.0.0/GAME-1.0-medium.zip -O /tmp/game_model.zip                                                         
unzip /tmp/game_model.zip -d GAME/  

gdown 1o6eUCYwUcIzZeqEycar6AHTSQ-vYcL_q --folder
gdown 1ylVTiDd_fwif2ISQzn9bxxYSkfEzXjQu -O ./MIDI-SAG_checkpoints/pretrained_CSLL2M.pt

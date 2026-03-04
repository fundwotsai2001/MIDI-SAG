pip install "pip<24.1"
pip install torch==2.5.1 
pip install torchaudio==2.5.1
pip install pytorch-fast-transformers==0.4.0 --no-build-isolation
pip install omegaconf==2.0.6 --no-deps
pip install fairseq==0.12.2 --no-deps
pip install pyaudio
pip install -r requirements.txt
pip install torchcodec
pip install chorder
pip install miditok

wget https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip -O /tmp/some_model.zip                                                         
unzip /tmp/some_model.zip -d SOME/pretrained/                                                                                                    
rm /tmp/some_model.zip 
wget https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip -O /tmp/RMVPE.zip      
unzip /tmp/RMVPE.zip -d SOME/pretrained/rmvpe                                                                                                 
rm /tmp/RMVPE.zip  
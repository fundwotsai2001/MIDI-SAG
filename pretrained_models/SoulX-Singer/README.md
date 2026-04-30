---

language:
- en
- zh
library_name: huggingface_hub
license: apache-2.0
pipeline_tag: text-to-speech
tags:
- text-to-audio
- music
- singing-voice-synthesis
- svs
- zero-shot

---

<div align="center">
    <b><em> Towards High-Quality Zero-Shot Singing Voice Synthesis</em></b>
    </p>
    <p>
    <img src="assets/soulx-logo.png" alt="SoulX-Singer_Logo" style="height: 80px;">
    </p>
    <p>
    </p>
    <a href="https://soul-ailab.github.io/soulx-singer/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Singer"><img src='https://img.shields.io/badge/Github-Page-green' alt="Github"></a>
    <a href="https://arxiv.org/abs/2602.07803"><img src="https://img.shields.io/badge/arXiv-2602.07803-b31b1b" alt="arXiv"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Singer/blob/main/assets/technical-report.pdf"><img src='https://img.shields.io/badge/Report-Github?label=Technical&color=red' alt="technical report"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Singer"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0"></a>
</div>

**SoulX-Singer** is a high-fidelity, zero-shot singing voice synthesis model that enables users to generate realistic singing voices for unseen singers. It supports melody-conditioned (F0 contour) and score-conditioned (MIDI notes) control for precise pitch, rhythm, and expression.

For more details, please refer to the paper: [SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis](https://arxiv.org/abs/2602.07803).

## Sample Usage

### 1. Set Up Environment

```bash
git clone https://github.com/Soul-AILab/SoulX-Singer.git
cd SoulX-Singer
conda create -n soulxsinger -y python=3.10
conda activate soulxsinger
pip install -r requirements.txt
```

### 2. Download Pretrained Models

```bash
pip install -U huggingface_hub

# Download the SoulX-Singer SVS model
hf download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer

# Download models required for preprocessing
hf download Soul-AILab/SoulX-Singer-Preprocess --local-dir pretrained_models/SoulX-Singer-Preprocess
```

### 3. Run Inference

```bash
bash example/infer.sh
```

## License

We use the Apache 2.0 license. Researchers and developers are free to use the codes and model weights of our SoulX-Singer. Check the license at [LICENSE](LICENSE) for more details.


## Usage Disclaimer
This project provides a singing voice synthesis model for vocal generation capable of zero-shot voice cloning, intended for academic research, educational purposes, and legitimate applications, such as personalized vocal synthesis and assistive technologies.

Please note:
We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.


## Citation

```bibtex
@misc{soulxsinger,
      title={SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis}, 
      author={Jiale Qian and Hao Meng and Tian Zheng and Pengcheng Zhu and Haopeng Lin and Yuhang Dai and Hanke Xie and Wenxiao Cao and Ruixuan Shang and Jun Wu and Hongmei Liu and Hanlin Wen and Jian Zhao and Zhonglin Jiang and Yong Chen and Shunshun Yin and Ming Tao and Jianguo Wei and Lei Xie and Xinsheng Wang},
      year={2026},
      eprint={2602.07803},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2602.07803}, 
}
```

## Contact Us
If you are interested in leaving a message to our work, feel free to email qianjiale@soulapp.cn or menghao@soulapp.cn or wangxinsheng@soulapp.cn

You’re welcome to join our WeChat or Soul APP group for technical discussions, updates.
<p align="center">


  <br>
  <span style="display: inline-block; margin-right: 10px;">
    <img src="assets/soul_wechat01.jpg" width="500" alt="WeChat Group QR Code"/>
  </span>
</p>



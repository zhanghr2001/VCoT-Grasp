# VCoT-Grasp: Grasp Foundation Models with Visual Chain-of-Thought Reasoning for Language-driven Grasp Generation


<!-- [![arXiv](https://img.shields.io/badge/arXiv-2409.14163-b31b1b.svg)](https://arxiv.org/abs/2409.14163) -->


Authors: Haoran Zhang, [Shuanghao Bai](https://baishuanghao.github.io/), [Wanqi Zhou](https://scholar.google.com/citations?user=3Q_3PR8AAAAJ&hl=zh-CN), Yuedi Zhang, Qi Zhang, Pengxiang Ding, Cheng Chi, Donglin Wang, [Badong Chen](https://scholar.google.com/citations?user=mq6tPX4AAAAJ&hl=zh-CN&oi=ao).


## Highlights

![main figure](model.jpg)
> **<p align="justify"> Abstract:** *Source-free domain generalization (SFDG) tackles the challenge of adapting models to unseen target domains without access to source domain data. 
To deal with this challenging task, recent advances in SFDG have primarily focused on leveraging the text modality of vision-language models such as CLIP. 
These methods involve developing a transferable linear classifier based on diverse style features extracted from the text and learned prompts or deriving domain-unified text representations from domain banks. 
However, both style features and domain banks have limitations in capturing comprehensive domain knowledge.
In this work, we propose Prompt-Driven Text Adapter (PromptTA) method, which is designed to better capture the distribution of style features and employ resampling to ensure thorough coverage of domain knowledge. 
To further leverage this rich domain information, we introduce a text adapter that learns from these style features for efficient domain information storage.
Extensive experiments conducted on four benchmark datasets demonstrate that PromptTA achieves state-of-the-art performance.* </p>

<details>
  
<summary>Main Contributions</summary>

1) We propose PromptTA, a novel adapter-based framework for SFDG that incorporates a text adapter to effectively leverage rich domain information.
2) We introduce style feature resampling that ensures comprehensive coverage of textual domain knowledge.
3) Extensive experiments demonstrate that our PromptTA achieves the state of the art on DG benchmarks.
   
</details>


## Installation 
Our code is tested on Ubuntu 22.04 LTS with cuda version 12.1. Follow the below steps to create environment and install dependencies.

* Setup conda environment.
```bash
# Create conda environment
conda create -y -n grasp python=3.9
conda activate grasp

# Install torch, refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

* Clone repository and install requirements.
```bash
git clone https://github.com/zhanghr2001/VCoT-Grasp.git
cd VCoT-Grasp

# Install requirements, version details are described in the file
pip install -r requirements.txt
```

## Data Preparation
Download Grasp-Anything

```bash
bash data_prepare/grasp_anything/download.sh
```

Modify the data path in constants.py
Our filter algorithm can be found in data_prepare/yolo_world.


## Training and Evaluation
Scripts for training and evaluation are in [scripts folder](scripts/). Modify *DATA* to your dataset directory before running.

```bash
# training
bash scripts/train.sh

# evaluation
bash scripts/eval.sh
```


<!-- ## Citation
If our code is helpful to your research or projects, please consider citing:
```bibtex
@misc{zhang2024prompttapromptdriventextadapter,
      title={PromptTA: Prompt-driven Text Adapter for Source-free Domain Generalization}, 
      author={Haoran Zhang and Shuanghao Bai and Wanqi Zhou and Jingwen Fu and Badong Chen},
      year={2024},
      eprint={2409.14163},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.14163}, 
}
``` -->


## Acknowledgements

Our style of readme refers to [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment). 
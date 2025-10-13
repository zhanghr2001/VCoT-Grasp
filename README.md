<div align="center">   
  
# VCoT-Grasp: Grasp Foundation Models with Visual Chain-of-Thought Reasoning for Language-driven Grasp Generation

</div>


<div align="center">   
  
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.05827) &ensp; [![Project](https://img.shields.io/badge/Project-Page-blue?logo=homepage&logoColor=white)](https://zhanghr2001.github.io/VCoT-Grasp.github.io/) &ensp; [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/zhanghr2001/VCoT-Grasp/)

</div>

This is the official repository for [VCoT-Grasp: Grasp Foundation Models with Visual Chain-of-Thought Reasoning for Language-driven Grasp Generation](https://arxiv.org/abs/2510.05827).


## 🌟 Highlights

![main figure](assets/model.png)
> **<p align="justify"> Abstract:** *Robotic grasping is one of the most fundamental tasks in robotic manipulation, and grasp detection/generation has long been the subject of extensive research. Recently, language-driven grasp generation has emerged as a promising direction due to its practical interaction capabilities. However, most existing approaches either lack sufficient reasoning and generalization capabilities or depend on complex modular pipelines. Moreover, current grasp foundation models tend to overemphasize dialog and object semantics, resulting in inferior performance and restriction to single-object grasping.
To maintain strong reasoning ability and generalization in cluttered environments, we propose VCoT-Grasp, an end-to-end grasp foundation model that incorporates visual chain-of-thought reasoning to enhance visual understanding for grasp generation. VCoT-Grasp adopts a multi-turn processing paradigm that dynamically focuses on visual inputs while providing interpretable reasoning traces.
For training, we refine and introduce a large-scale dataset, VCoT-GraspSet, comprising 167K synthetic images with over 1.36M grasps, as well as 400+ real-world images with more than 1.2K grasps, annotated with intermediate bounding boxes. Extensive experiments on both VCoT-GraspSet and real robot demonstrate that our method significantly improves grasp success rates and generalizes effectively to unseen objects, backgrounds, and distractors.* </p>


## 🤗 Model Zoo

| Model ID | Description | Params | DType | Link |
|:----------:|:-------------:|:--------:|:-------:|:------:|
| `zhanghr2001/VCoT-Grasp` | VCoT-Grasp with MLP head | 3B | bfloat16 | 🤗 [Link](https://huggingface.co/zhanghr2001/VCoT-Grasp/) |

TODO: We will release more pretrained models with a variety of heads.

Run the following command to download the checkpoint.
```bash
# Download model weights to checkpoints/vcot
huggingface-cli download zhanghr2001/VCoT-Grasp --local-dir checkpoints/vcot
```

The results are presented below. FT and ZS denote Fine-tuning and Zero-shot, respectively, with the number in parentheses indicating the number of object categories.

<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="3">VCoT-GraspSet</th>
    <th colspan="3">Real-World (FT)</th>
    <th rowspan="1">Real-World (ZS)</th>
    <th colspan="2">Real-World (FT,Generalization)</th>
  </tr>
  <tr>
    <th>Seen (367)</th>
    <th>Unseen (21)</th>
    <th>Avg.</th>
    <th>Seen (15)</th>
    <th>Unseen (15)</th>
    <th>Avg.</th>
    <th>Unseen (15)</th>
    <th>Background (5)</th>
    <th>Distractors (5)</th>
  </tr>
  <tr><td>LGD</td><td>0.39</td><td>0.13</td><td>0.20</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>CLIP-Fusion</td><td>0.52</td><td>0.14</td><td>0.21</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>GG-CNN + CLIP</td><td>0.56</td><td>0.18</td><td>0.27</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>GR-ConvNet + CLIP</td><td>0.71</td><td>0.33</td><td>0.45</td><td>0.68</td><td>0.55</td><td>0.62</td><td>0.44</td><td>0.60</td><td>0.48</td></tr>
  <tr><td>RT-Grasp</td><td>0.59</td><td>0.45</td><td>0.51</td><td>0.6</td><td>0.53</td><td>0.57</td><td>0.48</td><td>0.56</td><td>0.52</td></tr>
  <tr><td>VCoT-Grasp w/ MLP Head</td><td> <b>0.73</b> </td><td> <b>0.52</b> </td><td> <b>0.61</b> </td><td> <b>0.76</b> </td><td> <b>0.71</b> </td><td> <b>0.74</b> </td><td> <b>0.60</b></td><td> <b>0.84</b> </td><td> <b>0.64</b> </td></tr>
  <tr><td>VCoT-Grasp w/ LM Head</td><td> <b>0.84</b> </td><td> <b>0.59</b> </td><td> <b>0.69</b> </td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</table>



## 🛠️ Installation 
Our code is tested on Ubuntu 22.04 LTS and CUDA 12.1.

* Setup conda environment.
```bash
# Create conda environment
conda create -y -n grasp python=3.9
conda activate grasp

# Install torch, refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

* Install [FlashAttention](https://github.com/Dao-AILab/flash-attention) for training.
```bash
# Adjust MAX_JOBS according to your RAM.
pip install ninja
MAX_JOBS=4 pip install flash-attn --no-build-isolation
# Alternatively, it's recommended to download .whl file from https://github.com/Dao-AILab/flash-attention/releases and directly install the .whl file.
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

* Clone repository and install requirements.
```bash
# Clone repository
git clone https://github.com/zhanghr2001/VCoT-Grasp.git
cd VCoT-Grasp

# Install requirements, version details are described in the file
pip install -r requirements.txt
```

## 🚀 Quick Start
Run the following command to run the example demo.
```bash
# Run inference using assets/demo.jpg
python demo.py
```

## 💿 Data Preparation
Download source data from Grasp Anything.

```bash
# download and unzip
bash data_prepare/grasp_anything/download.sh
```

Also modify the path in constants.py to the downloaded data.
```python
grasp_anything_rgb_root = "<path>/grasp_anything/image"
grasp_anything_mask_root = "<path>/grasp_anything/mask"
grasp_anything_planar_grasp_root = "<path>/grasp_anything/grasp_label_positive"
```
Our VCoT-GraspSet can be found at [split/vcot](split/vcot). 
<!-- Our filter algorithm can be found in data_prepare/yolo_world. -->


## 🔥 Training
Launch training using 🤗[Accelerate](https://github.com/huggingface/accelerate). Training configs can be found in [accelerate_configs](accelerate_configs/).

```bash
# training
bash scripts/train.sh
```

## 📊 Evaluation
```bash
# evaluation
bash scripts/eval.sh
```


## 📑 Citation
If our code is helpful to your research or projects, please consider citing:
```bibtex
article{zhang2025vcot,
  title={VCoT-Grasp: Grasp Foundation Models with Visual Chain-of-Thought Reasoning for Language-driven Grasp Generation},
  author={Zhang, Haoran and Bai, Shuanghao and Zhou, Wanqi and Zhang, Yuedi and Zhang, Qi and Ding, Pengxiang and Chi, Cheng and Wang, Donglin and Chen, Badong},
  journal={arXiv preprint arXiv:2510.05827},
  year={2025}
}
```


## 🙏 Acknowledgements
Our baseline implementation builds upon [LGD](https://github.com/Fsoft-AIC/LGD). Our model is based on [Paligemma 2](https://huggingface.co/google/paligemma2-3b-mix-224). We extend our sincere thanks to the authors for their publicly available code and valuable contributions to the research community.

<div align="center">

<h1>UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement</h1>

<a href="https://arxiv.org/pdf/2512.21185"><img src="https://img.shields.io/badge/arXiv-2512.21185-b31b1b.svg?style=flat-square" alt="arXiv"></a>
<a href="https://pku-yuangroup.github.io/UltraShape-1.0/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page"></a>
<a href="https://huggingface.co/infinith/UltraShape"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=flat-square" alt="HuggingFace Models"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>

</div>

<br/>

<div align="center">
  <img src="asset/images/teaser.png" width="100%" alt="UltraShape 1.0 Teaser" />
</div>

<br/>

## 📖 Abstract

In this report, we introduce **UltraShape 1.0**, a scalable 3D diffusion framework for high-fidelity 3D geometry generation. The proposed approach adopts a **two-stage generation pipeline**: a coarse global structure is first synthesized and then refined to produce detailed, high-quality geometry.

To support reliable 3D generation, we develop a comprehensive data processing pipeline that includes a novel **watertight processing method** and **high-quality data filtering**. This pipeline improves the geometric quality of publicly available 3D datasets by removing low-quality samples, filling holes, and thickening thin structures, while preserving fine-grained geometric details. 

To enable fine-grained geometry refinement, we decouple spatial localization from geometric detail synthesis in the diffusion process. We achieve this by performing **voxel-based refinement** at fixed spatial locations, where voxel queries derived from coarse geometry provide explicit positional anchors encoded via **RoPE**, allowing the diffusion model to focus on synthesizing local geometric details within a reduced, structured solution space.

Extensive evaluations demonstrate that UltraShape 1.0 performs competitively with existing open-source methods in both data processing quality and geometry generation.

## 🔥 News
* **[2025-12-25]** 📄 We released the technical report of **UltraShape 1.0** on arXiv.
* **[2025-12-26]** 🚀 We released the inference code and pre-trained models.
* **[2025-12-31]** 🚀 We released the training code.

## 🗓️ To-Do List
- [x] Release inference code
- [x] Release pre-trained weights (Hugging Face)
- [x] Release training code
- [ ] Release data processing scripts

## 🛠️ Installation & Usage

### 1. Environment Setup
```bash
git clone https://github.com/PKU-YuanGroup/UltraShape-1.0.git
cd UltraShape
# 1. Create and activate the environment
conda create -n ultrashape python=3.10
conda activate ultrashape

# 2. Install PyTorch (CUDA 12.1 recommended)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install cubvh (Required for MC acceleration)
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation

# For Training & Sampling (Optional)
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
```
⬇️ Model Weights

Please download the pre-trained weights from Hugging Face [ [infinith/UltraShape](https://huggingface.co/infinith/UltraShape/tree/main) ] and place them in your checkpoint directory (e.g., ./checkpoints/).


### 2. Generate Coarse Mesh

First, use Hunyuan3D-2.1 to generate a coarse mesh from your input image.

Repository: [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)

Follow the instructions in the Hunyuan3D-2.1 repository to obtain the initial mesh file (e.g., .glb or .obj).

### 3. Generate Refined Mesh

Once you have the coarse mesh, use the provided script to run the refinement stage.

Run the inference script:
```bash
sh scripts/run.sh
```

**image**: Path to the reference image.

**mesh**: Path to the coarse mesh.

**output_dir**: Directory to save the refined result.

**ckpt**: Path to the downloaded UltraShape checkpoint.

### 4. Data Preparation & Training

First, prepare the data, including watertight meshes and rendered images.
Then, run the sampling script as follows:
```
python scripts/sampling.py \
    --mesh_json data/mesh_paths.json \
    --output_dir data/sample
```

Here, mesh_json is a list containing the file paths of the watertight meshes.


The multi-node training script is:
```
sh train.sh [node_idx]
```

**training_data_list**: the folder containing train.json and val.json, which store the ID lists for datasets.

**sample_pcd_dir**: the directory containing the sampled .npz files.

**image_data_json**: the file paths of the rendered images.

You can switch between VAE and DiT training in train.sh, and specify the output directory and configuration file there as well.

## 🔗 BibTeX

If you found this repository helpful, please cite our reports:

```bibtex
@article{jia2025ultrashape,
    title={UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement},
    author={Jia, Tanghui and Yan, Dongyu and Hao, Dehao and Li, Yang and Zhang, Kaiyi and He, Xianyi and Li, Lanjiong and Chen, Jinnan and Jiang, Lutao and Yin, Qishen and Quan, Long and Chen, Ying-Cong and Yuan, Li},
    journal={arxiv preprint arXiv:2512.21185},
    year={2025}
}
```

## Acknowledgements

Our code is built upon the excellent work of [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1). The core idea of our method is greatly inspired by [LATTICE](https://arxiv.org/abs/2512.03052). We deeply appreciate the contributions of these works to the 3D generation community. Please also consider citing **Hunyuan3D 2.1** and **LATTICE**:

- **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)**
- **[Lattice3D](https://lattice3d.github.io/)**

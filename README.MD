
# Multispectral Demosaicing via Dual Cameras

**SaiKiran Tedla\*, Junyong Lee\*, Beixuan Yang, Mahmoud Afifi, Michael S. Brown**  
Samsung AI Center Toronto & York University

📄 [Paper (PDF)](https://arxiv.org/pdf/2503.22026)  
📄 [Project Page & Dataset](https://ms-demosaic.github.io)

---

## 📌 Citation

If you use our dataset or code, please cite:

```bibtex
@inproceedings{TedlaLee2025Multispectral,
  title={{Multispectral Demosaicing via Dual Cameras}},
  author={{Tedla, SaiKiran and Lee, Junyong and Yang, Beixuan and Afifi, Mahmoud and Brown, Michael S}},
  booktitle={{Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}},
  year={{2025}}
}
```

---

## 🚀 Getting Started

This section describes how to train and test our unified demosaicing and denoising model.

### 🔧 Environment Setup

```bash
conda env create -f p312.yml
conda activate p312
```

- Copy the **DNG** provided in the dataset link into `PyTorch/utilities/`.  
  This DNG is used for visualization and computing the DeltaE metric.  
- Create a [Weights & Biases (wandb)](https://wandb.ai/) account for experiment tracking.  
  Update the `wandb` settings in the YAML config files accordingly.

---

### 📂 Dataset

- Download the dataset and visualization DNG from our [project page](https://ms-demosaic.github.io).  
- Unzip all dataset files into the **same directory**.  
- Update the config files with the correct dataset paths.  
- The **train/val/test split** is specified in `PyTorch/utilities/multi_spectral_dataset.py`.  
- Use `PyTorch/utilities/color_space_converter` for visualization.  
  We provide a sample DNG used to extract metadata for rendering.

---

### 🏋️‍♂️ Training

**⚠️ Important:**  
Before training, **set `test: false`** in the corresponding config file, update dataset paths, and configure your `wandb` project/entity.

We provide configs for both **full-size** and **super-resolution** experiments from our paper.

**Stage 1 – Train RGB and MS networks:**
```bash
python PyTorch/runner.py --config PyTorch/configs/stage1_full_ms.yaml
python PyTorch/runner.py --config PyTorch/configs/stage1_full_rgb.yaml
```

**Stage 2 – Train enhancement/fusion network:**
```bash
python PyTorch/runner.py --config PyTorch/configs/stage2_naf_full.yaml
```

---

### 🧪 Testing

**⚠️ Important:**  
Before testing, **set `test: true`** in the config file and provide paths to the pretrained checkpoints (available on the [project page](https://ms-demosaic.github.io)). Base_models (stage 1) and models (stage 2) folders should by placed inside PyTorch. 

For example, you may run:
```bash
python PyTorch/runner.py --config PyTorch/configs/stage2_naf_full.yaml
```

---

### 📜 Notes

- All config files are initially in **test mode**. Switch to **train mode** by setting `test: false`. We provide configs for training stage 1 and stage 2 networks.
- Checkpoints for reproducing paper results are available on the [project page](https://ms-demosaic.github.io). Base_models (stage 1) and models (stage 2) folders should by placed inside PyTorch. 
- For visualization, the provided DNG and color space converter will help render multispectral data.  

---

### 📨 Contact

For questions or issues, please reach out via the [project page](https://ms-demosaic.github.io) or open an issue on the repository.

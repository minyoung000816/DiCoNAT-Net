# DiCoNAT-Net

**A Dilated Neighborhood Attention Transformer with Dual-Path Architecture for Motion Deblurring**

DiCoNAT-Net is a PyTorch-based image deblurring network designed to restore images degraded by motion blur.  
The network combines a **Dual-Path Encoder–Decoder**, **Dilated Neighborhood Attention (DiNAT)**, and a **two-stage cascaded refinement strategy** to model directional blur patterns and long-range dependencies while keeping the model compact.

## Overview

Motion blur caused by camera shake or rapid object motion can severely degrade image quality and structural details. CNN-based deblurring networks are effective at extracting local features, but their fixed receptive fields make it difficult to model long-range dependencies. Transformer-based approaches improve global context modeling, but conventional self-attention can introduce substantial computational overhead.

DiCoNAT-Net addresses these limitations through three main ideas:

- **Dual-Path Encoder–Decoder**  
  Two directional paths process the original image and a 90°-rotated image to learn horizontal and vertical blur characteristics separately.

- **DiNAT-based Decoder**  
  Dilated Neighborhood Attention captures local and broader contextual relationships using neighborhood attention with different dilation rates.

- **Cascaded Refinement**  
  Two consecutive deblurring stages progressively remove residual blur and refine fine structural details.

## Network Architecture

<p align="center">
  <img src="assets/architecture.png" alt="DiCoNAT-Net architecture" width="900">
</p>

The first stage performs directional feature extraction and initial blur removal. Its restored output is then passed to the second stage for further refinement. Each stage uses a dual-path encoder–decoder structure with residual connections.

## ResBlock and DiNAT Block

<p align="center">
  <img src="assets/dinat_resblock.png" alt="ResBlock and DiNAT Block" width="850">
</p>

The encoder uses **ResBlocks** for local feature extraction, while the decoder uses **DiNAT blocks** to model feature relationships over wider spatial regions. In the proposed configuration, local attention uses a dilation rate of 1, while larger dilation rates are used to progressively capture broader context.

## Quantitative Results

The model was trained on the **GoPro** dataset and additionally evaluated on **HIDE** and **REDS** to examine generalization under different motion-blur conditions.

| Dataset | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | VIF ↑ | MSE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| GoPro | **32.86** | **0.9603** | **0.9765** | **0.5970** | **41.33** | **0.0859** |
| HIDE | **30.66** | **0.9365** | **0.9670** | **0.5328** | **74.58** | **0.1158** |
| REDS | **27.19** | **0.8736** | **0.9243** | **0.6987** | **195.82** | **0.2013** |

On the GoPro dataset, DiCoNAT-Net achieved **32.86 dB PSNR** and **0.0859 LPIPS**. Compared with MRDNet in the reported experiments, this corresponds to an improvement of **1.15 dB in PSNR** and **0.0154 in LPIPS**.

The proposed model contains approximately **7.5M parameters**. In the Transformer-based comparison reported in the manuscript, this is approximately **54.55% fewer parameters** than CTMS, the most compact Transformer-based benchmark in that comparison.

<p align="center">
  <img src="assets/parameter_performance.png" alt="Parameter and PSNR comparison" width="720">
</p>

## Qualitative Results

### GoPro Example 1

<p align="center">
  <img src="assets/gopro_qualitative_1.png" alt="GoPro qualitative comparison 1" width="900">
</p>

### GoPro Example 2

<p align="center">
  <img src="assets/gopro_qualitative_2.png" alt="GoPro qualitative comparison 2" width="900">
</p>

The qualitative comparisons illustrate the restoration of fine structures such as repeated line patterns and window boundaries under motion blur.

## Downstream Object Detection

To evaluate whether deblurring improves downstream computer-vision performance, restored GoPro images were also evaluated using YOLOv8. In the reported experiment, deblurring with DiCoNAT-Net improved **mAP50 by 9.5%** relative to the blurred input, with improvements also observed for the People, Car, and Potted Plant classes.

<p align="center">
  <img src="assets/object_detection.png" alt="Object detection comparison" width="900">
</p>

## Dataset Structure

The current dataloader expects paired blurry and sharp images with matching filenames.

```text
dataset/
├── train/
│   ├── blur/
│   └── sharp/
├── test/
│   ├── blur/
│   └── sharp/
└── valid/              # optional
    ├── blur/
    └── sharp/
```

For the original experiments, the GoPro dataset contains **2,103 training pairs** and **1,111 test pairs**.

## Training

The implementation uses PyTorch. The main experimental settings reported in the manuscript are:

- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: `4`
- Training epochs: `3000`
- Training crop size: `256 × 256`
- Hardware used for the reported experiments: NVIDIA GeForce RTX 4090

Run training with:

```bash
python main.py --mode train --data_dir /path/to/dataset
```

Training checkpoints and outputs are stored under:

```text
results/DiCoNATNet/
```

## Evaluation

The default evaluation code expects the checkpoint at:

```text
results/DiCoNATNet/weights/Best.pkl
```

Run evaluation with:

```bash
python main.py \
    --mode test \
    --data_dir /path/to/dataset \
    --test_model ./weights/Best.pkl
```

Pretrained weights are not included in this repository.

## Core Dependencies

The project uses the following major libraries:

- Python
- PyTorch
- torchvision
- NATTEN
- NumPy
- SciPy
- scikit-image
- tqdm
- TensorBoard
- fvcore (for FLOPs analysis)

> Exact package versions from the original experiment should be added if the original environment information is available.

## Project Structure

```text
DiCoNAT-Net/
├── assets/
│   ├── architecture.png
│   ├── dinat_resblock.png
│   ├── parameter_performance.png
│   ├── gopro_qualitative_1.png
│   ├── gopro_qualitative_2.png
│   └── object_detection.png
├── data/
│   ├── __init__.py
│   ├── data_augment.py
│   └── data_load.py
├── models/
│   ├── DiCoNATNet.py
│   └── layers.py
├── eval.py
├── flops.py
├── main.py
├── train.py
├── utils.py
├── valid.py
├── .gitignore
└── README.md
```

## Related Manuscript

**A Dilated Neighborhood Attention Transformer with Dual-Path Architecture for Motion Deblurring**  
Minyoung Lee, Ho Sub Lee

The manuscript introduces the proposed DiCoNAT framework and reports experiments on GoPro, HIDE, and REDS, together with ablation studies, model-complexity comparisons, and downstream object-detection evaluation.

## Acknowledgements

Parts of the training and evaluation pipeline were developed with reference to the public **XYDeblur** implementation:

- [XY-Single-Image-Deblurring](https://github.com/Tarak200/XY-Single-Image-Deblurring)

DiCoNAT-Net introduces the proposed dual-path directional architecture, DiNAT-based decoding, and cascaded refinement design used in this work.

## Notes

- The repository does not include the GoPro, HIDE, or REDS datasets.
- The repository does not currently include pretrained model weights.
- Input image height and width should be compatible with the network's downsampling/upsampling structure.

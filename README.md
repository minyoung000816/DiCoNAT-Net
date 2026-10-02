# DiCoNAT-Net

**A Dilated Neighborhood Attention Transformer with Dual-Path Architecture for Motion Deblurring**

PyTorch implementation of **DiCoNAT-Net**, a motion deblurring network that combines direction-aware feature learning, Dilated Neighborhood Attention (DiNAT), and cascaded refinement.

## Highlights

* **32.86 dB PSNR / 0.9603 SSIM** on the GoPro test set
* **7.5M parameters**
* Direction-aware **Dual-Path Encoder–Decoder**
* **DiNAT-based decoder** for local and long-range dependency modeling
* Evaluated on **GoPro, HIDE, and REDS**
* Additional validation through **ablation studies** and **YOLOv8 object-detection experiments**

## Key Contributions

DiCoNAT-Net was designed to address the limited receptive field of CNN-based deblurring models and the high computational cost of conventional Transformer self-attention.

* **Dual-Path Encoder–Decoder**  
The original image and a 90°-rotated image are processed through separate directional paths to learn horizontal and vertical blur characteristics.
* **DiNAT-based Decoder**  
Dilated Neighborhood Attention is applied in the decoder to capture both local information and wider contextual dependencies with reduced attention cost.
* **Two-Stage Cascaded Refinement**  
Two consecutive deblurring stages progressively remove residual blur and refine structural details.

The proposed network was implemented in PyTorch and evaluated through quantitative and qualitative comparisons, ablation studies, model-complexity analysis, and object-detection experiments.

## Network Architecture

<p align="center">
  <img src="assets/architecture.png" alt="DiCoNAT-Net architecture" width="900">
</p>

Each stage consists of a Dual-Path Encoder–Decoder. The first stage performs initial restoration, and the second stage further refines the intermediate result.

### ResBlock and DiNAT Block

<p align="center">
  <img src="assets/dinat\_resblock.png" alt="ResBlock and DiNAT Block" width="850">
</p>

The encoder uses **ResBlocks** for local feature extraction, while the decoder uses **DiNAT blocks** for neighborhood-based attention. Local attention uses a dilation rate of 1, while larger dilation rates of 9, 18, and 36 are used to capture wider spatial context.

## Results

|Dataset|PSNR ↑|SSIM ↑|MS-SSIM ↑|VIF ↑|MSE ↓|LPIPS ↓|
|-|-:|-:|-:|-:|-:|-:|
|GoPro|**32.86**|**0.9603**|**0.9765**|**0.5970**|**41.33**|**0.0859**|
|HIDE|**30.66**|**0.9365**|**0.9670**|**0.5328**|**74.58**|**0.1158**|
|REDS|**27.19**|**0.8736**|**0.9243**|**0.6987**|**195.82**|**0.2013**|

On the GoPro dataset, DiCoNAT-Net achieved **32.86 dB PSNR** and **0.0859 LPIPS**. In the reported experiments, this improved PSNR by **1.15 dB** and LPIPS by **0.0154** compared with MRDNet.

The model contains approximately 7.5M parameters. In the Transformer-based comparison, DiCoNAT-Net uses approximately 54.55% fewer parameters than CTMS, the most compact Transformer-based benchmark included in the evaluation.

<p align="center">
  <img src="assets/parameter\_performance.png" alt="Parameter and PSNR comparison" width="720">
</p>

## Qualitative Results

<p align="center">
  <img src="assets/gopro\_qualitative\_1.png" alt="GoPro qualitative comparison 1" width="900">
</p>

<p align="center">
  <img src="assets/gopro\_qualitative\_2.png" alt="GoPro qualitative comparison 2" width="900">
</p>

The examples show restoration of fine structures such as repeated line patterns and window boundaries under motion blur.

## Object Detection Evaluation

To evaluate the practical effectiveness of image deblurring, restored GoPro images were evaluated using **YOLOv8**.

In the reported experiment, DiCoNAT-Net improved **mAP50 by 9.5%** compared with the blurred input.

<p align="center">
  <img src="assets/object\_detection.png" alt="Object detection comparison" width="900">
</p>

## Project Structure

```text
DiCoNAT-Net/
├── assets/
├── data/
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

## Acknowledgements

Parts of the training and evaluation pipeline were developed with reference to the public **XYDeblur** implementation:

* [XY-Single-Image-Deblurring](https://github.com/Tarak200/XY-Single-Image-Deblurring)

DiCoNAT-Net introduces the proposed dual-path directional architecture, DiNAT-based decoding, and cascaded refinement design used in this work.


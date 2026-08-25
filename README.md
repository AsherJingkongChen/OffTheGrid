# Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting

[![Project Page](https://img.shields.io/badge/Project%20Page-Visit-blue)](https://arthurmoreau.github.io/OffTheGrid/)
[![Latest Paper](https://img.shields.io/badge/Latest%20Paper-Read-orange)](https://arxiv.org/abs/2512.15508)



<p align="center">
  <img src="https://arthurmoreau.github.io/OffTheGrid/assets/images/tease.png" width="70%" alt="The teaser figure of the paper, showing Off The Grid Gaussians">
</p>

Implementation of "Off The Grid: Detection of Primitives for Feed-Forward Gaussian Splatting" (CVPR 2026).
It is a pose-free feed-forward 3DGS pipeline, based on Large Reconstruction Models such as VGGT (denoted as encoder), and a convolutional decoder that detects gaussians primitives. This repository contains code to perform inference, reproduce paper results and train the model, including fine-tuning of the encoder.

## How to use

<details>
<summary>Installation</summary>

```
git clone https://gitlab-uk.rnd.huawei.com/ai-uk-team/enhanced-imaging-group/offthegrid.git
cd offthegrid

conda create -n offthegrid python=3.12
conda activate offthegrid

#Install torch and torchvision with CUDA adapted to your system
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/vggt.git --no-build-isolation
pip install git+https://github.com/nerfstudio-project/gsplat.git@3d4f9027f36d70ac8e89536e14c247de377f74b5 --no-build-isolationpip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

```

Install DepthAnything3 (optional): 
```
pip install hatchling
pip install --no-build-isolation git+https://github.com/ByteDance-Seed/Depth-Anything-3
```
</details>

<details>
<summary>Checkpoints</summary>

Download checkpoints of our model [here](https://huggingface.co/ArthurMoreau/OffTheGrid/tree/main).
</details>

<details>
<summary>Simple Inference</summary>

The `simple_inference.py` script runs the model on a collection of uncalibrated images and predict a 3D gaussian model. It can be used to generate novel view synthesis videos, a ply file, a COLMAP model, a mesh, and to visualize adaptive density outputs. It runs from a folder containing a sequence of images and subsample them following different available strategies. While it is compatible with any number of views (given enough VRAM), the model is trained and performs better with 1-12 views.


Run the example script:
```bash
bash scripts/simple_inference.sh
```

</details>


<details>
<summary>Data</summary>

We provide a 'minimal dataset' containing 3 scenes with 6 images each to try scripts before downloading data. You can download public datasets for evaluation and training using the following links:

**Test datasets:** [Charge](https://huggingface.co/charge-benchmark/Charge), [SCRREAM](https://github.com/Junggy/SCRREAM), [DL3DV](https://github.com/DL3DV-10K/Dataset), [MipNeRF360](https://jonbarron.info/mipnerf360/), [Tanks and Temples](https://www.tanksandtemples.org/download/), [7scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

**Training datasets:** [DL3DV](https://github.com/DL3DV-10K/Dataset), [ARKit](https://github.com/apple/ARKitScenes/blob/main/DATA.md), [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/), [Co3Dv2](https://github.com/facebookresearch/co3d), [Unreal](https://huggingface.co/datasets/fabiotosi92/UnrealStereo4K), [Kitti360](https://www.cvlibs.net/datasets/kitti-360/index.php), [WildRGBD](https://github.com/wildrgbd/wildrgbd), [BlendedMVS](https://github.com/yoyo000/blendedmvs), [DTU](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip)

</details>

<details>
<summary>Evaluation</summary>

The `evaluation.py` script runs the model on test datasets and computes rendering (PSNR, SSIM, LPIPS) and geometry (depth estimation, camera pose estimation, focal length estimation) metrics.

As the model is pose-free, ground truth camera poses can not be used for rendering evaluation. Similar to AnySplat, we circumvent the problem by performing inference 2 times: one with source views only, and one with source+target views. Camera poses are aligned up to a single scale factor and novel views are rendered from the first model.


Run the example script:
```bash
bash scripts/evaluation.sh
```

Results from the paper can be reproduced using 'dataset=full_eval' which performs evaluation on all datasets and averages metrics. Evaluation on a new custom dataset can be performed by implementing a dataloader.

</details>

<details>
<summary>Training</summary>

Our self-supervised training procedure does not require any 3d annotations. It is trained on video frames depicting static scenes. We start from a pretrained encoder (VGGT) and train the detection decoder from scratch.

**Usage:**
```bash
bash scripts/train_offthegrid_vggt.sh
```
Fine-tuning the encoder requires a GPU with a large amount of VRAM (>100GB).

We also provide a script
Extensive visualizations of the model are logged to TensorBoard.

</details>

<details>
<summary>Additional features</summary>

- DA3 Gaussian model baseline
- Generate  'Bullet time' effects videos with the 'src/bullet_time.py' script.  
- You can train the model with DA3 encoder.
- Some implemented features might not work properly (VGGT-omega encoder, EMA teacher).

One limitation of the model is the lack of view-dependent colours, integrating spherical harmonics would be a nice addition left as future work.

</details>

## Citation

```tex
@inproceedings{moreau2026off,
  title={Off the grid: Detection of primitives for feed-forward 3d gaussian splatting},
  author={Moreau, Arthur and Shaw, Richard and Nazarczuk, Michal and Shin, Jisu and Tanay, Thomas and Zhang, Zhensong and Xu, Songcen and P{\'e}rez-Pellitero, Eduardo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11756--11766},
  year={2026}
}
```
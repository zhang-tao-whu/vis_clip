<div align="center">

# [DVIS++: Improved Decoupled Framework for Universal Video Segmentation]()
[Tao Zhang](https://scholar.google.com/citations?user=3xu4a5oAAAAJ&hl=zh-CN), XingYe Tian, Yikang Zhou, 
[ShunPing Ji](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=zh-CN), Xuebo Wang, Xin Tao,

Yuan Zhang, Pengfei Wan, Zhongyuan Wang and 
[Yu Wu](https://scholar.google.com/citations?hl=zh-CN&user=23SZHUwAAAAJ)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-decoupled-video-instance-segmentation/video-instance-segmentation-on-ovis-1)](https://paperswithcode.com/sota/video-instance-segmentation-on-ovis-1?p=dvis-decoupled-video-instance-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-decoupled-video-instance-segmentation/video-panoptic-segmentation-on-vipseg)](https://paperswithcode.com/sota/video-panoptic-segmentation-on-vipseg?p=dvis-decoupled-video-instance-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-decoupled-video-instance-segmentation/video-instance-segmentation-on-youtube-vis-3)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-3?p=dvis-decoupled-video-instance-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-decoupled-video-instance-segmentation/video-instance-segmentation-on-youtube-vis-1)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-1?p=dvis-decoupled-video-instance-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-decoupled-video-instance-segmentation/video-instance-segmentation-on-youtube-vis-2)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-2?p=dvis-decoupled-video-instance-segmentation)

<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/radar.png" width="200"/>
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/radar_ov.png" width="200"/>
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/bar.png" width="400"/>
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/overview.png" width="800"/>
</div>

## News
- DVIS and DVIS++ achieved **1st place** in the VPS Track of the PVUW challenge at CVPR 2023. `2023.5.25`
- DVIS and DVIS++ achieved **1st place** in the VIS Track of the 5th LSVOS challenge at ICCV 2023. `2023.8.15`

## Features
- DVIS++ is a universal video segmentation framework that supports VIS, VPS and VSS.
- DVIS++ can run in both online and offline modes. 
- DVIS++ achieved SOTA performance on YTVIS 2019&2021&2022, OVIS, VIPSeg and VSPW datasets.
- OV-DVIS++ is the first open-vocabulary video universal segmentation framework with powerful zero-shot segmentation capability. 

## Demos
### VIS
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/34df9b7e.gif" width="200"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/52ee3d90.gif" width="600"/>
### VSS
### VPS
### Open-vocabulary demos

## Installation

See [Installation Instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for DVIS++](datasets/README.md).

See [Getting Started with DVIS++](GETTING_STARTED.md).

## Model Zoo

Trained models are available for download in the [DVIS++ Model Zoo](MODEL_ZOO.md).

## <a name="CitingDVIS"></a>Citing DVIS and DVIS++

```BibTeX
@article{DVIS,
  title={DVIS: Decoupled Video Instance Segmentation Framework},
  author={Zhang, Tao and Tian, Xingye and Wu, Yu and Ji, Shunping and Wang, Xuebo and Zhang, Yuan and Wan, Pengfei},
  journal={arXiv preprint arXiv:2306.03413},
  year={2023}
}
```

## Acknowledgement

This repo is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), 
[MinVIS](https://github.com/NVlabs/MinVIS), [VITA](https://github.com/sukjunhwang/VITA), 
[FC-CLIP](https://github.com/bytedance/fc-clip) and [DVIS](https://github.com/zhang-tao-whu/DVIS).
Thanks for their excellent works.
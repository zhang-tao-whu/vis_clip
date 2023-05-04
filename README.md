# [DVIS: Decoupled Video Instance Segmentation Framework]()

[Tao Zhang](https://scholar.google.com/citations?user=3xu4a5oAAAAJ&hl=zh-CN), XingYe Tian, [Yu Wu](https://scholar.google.com/citations?hl=zh-CN&user=23SZHUwAAAAJ), [ShunPing Ji](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=zh-CN), Xuebo Wang, Yuan Zhang, Pengfei Wan

## Features
- DVIS is a universal video segmentation framework that supports VIS, VPS and VSS.
- DVIS can run both in online and offline modes. 
- DVIS achieved SOTA performance on YTVIS, OVIS, VIPSeg and VSPW datasets.
- DVIS can be trained and inference on 11G GPUS.

## Demos

<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_0.gif" width="390"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_1.gif" width="360"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_2.gif" width="234"/>

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for DVIS](datasets/README.md).

See [Getting Started with DVIS](GETTING_STARTED.md).

## Model Zoo

Trained models are available for download in the [DVIS Model Zoo](MODEL_ZOO.md).

## Acknowledgement

This repo is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), [MinVIS](https://github.com/NVlabs/MinVIS) and [VITA](https://github.com/sukjunhwang/VITA).
Thanks for their excellent works.
## Visualization

### Testing on COCO

Command:

   ```
   python demo_video/demo_long_video.py \
       --config-file /path/to/config.yaml \
       --input /path/to/images \
       --output work_dirs/demo_out/ \
       --opts MODEL.WEIGHTS /path/to/weight.pth
   ```
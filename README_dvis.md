# [DVIS: Decoupled Video Instance Segmentation Framework]()

[Tao Zhang](https://scholar.google.com/citations?user=3xu4a5oAAAAJ&hl=zh-CN), XingYe Tian, [Yu Wu](https://scholar.google.com/citations?hl=zh-CN&user=23SZHUwAAAAJ), [ShunPing Ji](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=zh-CN), Xuebo Wang, Yuan Zhang, Pengfei Wan

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
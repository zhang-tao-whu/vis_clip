from thop import profile
import torch
import argparse
from detectron2.modeling import build_model
from thop import clever_format
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from minvis import add_minvis_config


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

video_size_dict = {'480p': [480, 853], '720p': [720, 1280]}
configs_dict = {'r50': 'configs/ovis/video_maskformer2_R50_bs32_8ep_frame_offline.yaml',
                'swinl': 'configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_offline.yaml'}
parser = argparse.ArgumentParser(description="FlexVIS GFlops")
parser.add_argument(
    "--windows_size",
    type=int,
    default=80,
    help="Windows size for semi-offline mode",
)
parser.add_argument(
    "--video_size",
    type=str,
    default='480p',
    help="Windows size for semi-offline mode",
)
parser.add_argument(
    "--backbone",
    type=str,
    default='r50',
    help="Windows size for semi-offline mode",
)

args = parser.parse_args()
args.config_file = configs_dict[args.backbone]
args.opts = []
input_size = video_size_dict[args.video_size]
cfg = setup_cfg(args)
model = build_model(cfg.clone())
model.eval()

# backbone GFlops
backbone = model.backbone
print(backbone)
input = torch.randn(1, 3, input_size[0], input_size[1]).to(model.device)
macs, params = profile(backbone, inputs=(input, ))
#macs, params = clever_format([macs, params], "%.3f")
print(macs)
print(params)
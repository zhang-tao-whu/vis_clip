import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image

def check_image(seg_file, seg_infos):
    pan_seg = np.array(Image.open(seg_file))
    pan_seg = np.uint32(pan_seg)
    pan_seg = pan_seg[:, :, 0] + pan_seg[:, :, 1] * 256 + pan_seg[:, :, 2] * 256 * 256
    unique_ids, ids_cnts = np.unique(pan_seg, return_counts=True)
    seg_ids, seg_ids_cnts = [], []
    for seg_info in seg_infos:
        seg_ids.append(seg_info['id'])
        seg_ids_cnts.append(seg_info['area'])
    for id in seg_ids:
        if id not in unique_ids:
            print('------------------------------------')
            print('segment infos id not found in png file')
            print(seg_file)
            print('unique_ids:', unique_ids)
            print('unique_cnts:', ids_cnts)
            print('seg_ids:', seg_ids)
            print('seg_ids_cnts:', seg_ids_cnts)
    for id in unique_ids:
        if id not in seg_ids:
            print('------------------------------------')
            print('png files id not found in segment infos')
            print(seg_file)
            print('unique_ids:', unique_ids)
            print('unique_cnts:', ids_cnts)
            print('seg_ids:', seg_ids)
            print('seg_ids_cnts:', seg_ids_cnts)
    return




json_path = 'output/inference/pred.json'
img_path = 'output/inference/pan_pred/'

with open(json_path, 'r') as f:
    json_file = json.load(f)

video_annotations = json_file['annotations']
for video_annotation in tqdm(video_annotations):
    video_id = video_annotation['video_id']
    images_annotations = video_annotation['annotations']
    for image_annotation in images_annotations:
        seg_file = os.path.join(img_path, video_id, image_annotation['file_name'].split('.')[0] + '.png')
        check_image(seg_file, image_annotation['segments_info'])

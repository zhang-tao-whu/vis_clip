import cv2
import os
import numpy as np
import tqdm

src = 'output/inference/pan_pred'
drc = 'output/inference/pan_pred_reverse'

videos = os.listdir(src)

for video in tqdm.tqdm(videos):
    images = os.listdir(os.path.join(src, video))
    if not os.path.exists(os.path.join(drc, video)):
        os.makedirs(os.path.join(drc, video))
    for image in images:
        seg = np.array(cv2.imread(os.path.join(src, video, image)))
        seg = seg[:, :, ::-1]
        cv2.imwrite(os.path.join(drc, video, image), seg)


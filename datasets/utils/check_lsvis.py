import json
import os
from PIL import Image

json_file = '../lsvis/train_instances.json'
with open(json_file, 'r') as f:
    json_file = json.load(f)

videoID2videos = {}
for item in json_file['videos']:
    id = item["id"]
    videoID2videos[id] = item

for anno in json_file['annotations']:
    video_id = anno['video_id']
    height, width = anno['height'], anno['width']
    video_info = videoID2videos[video_id]
    video_height, video_width = video_info['height'], video_info['width']
    images_files = video_info["file_names"]
    images_files = [os.path.join('../lsvis/train/JPEGImages', item) for item in images_files]
    assert video_height == height and video_width == width, print((height, width), (video_height, video_width))
    for i, file in enumerate(images_files):
        image = Image.open(file)
        image_width, image_height = image.size
        assert video_height == image_height and video_width == image_width, print((height, width),
                                                                                  (image_height, image_width),
                                                                                  anno['segmentations'][i]["size"])

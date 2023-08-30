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

print(json_file.keys())

wrong_videos_id = []
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
        if not (video_height == image_height and video_width == image_width):
            wrong_videos_id.append(video_id)
        if anno['segmentations'][i] is None:
            wrong_videos_id.append(video_id)
            # if anno['segmentations'][i] is None:
            #     print(i, file, video_info)
            #     print(anno['segmentations'][:i])
            # else:
            #     if not isinstance(anno['segmentations'][i], dict):
            #         print(anno['segmentations'][i], type(anno['segmentations'][i]))
            #     print(file, (height, width), (image_height, image_width), anno['segmentations'][i]["size"])

wrong_videos_id = set(wrong_videos_id)
print(wrong_videos_id)
videos_ = []
for video in json_file["videos"]:
    if video["id"] in wrong_videos_id:
        pass
    else:
        videos_.append(video)
print(len(videos_), '/', len(json_file["videos"]))
json_file["videos"] = videos_

annotations_ = []
for anno in json_file['annotations']:
    video_id = anno['video_id']
    if video_id not in wrong_videos_id:
        annotations_.append(anno)
print(len(annotations_), '/', len(json_file["annotations"]))
json_file["annotations"] = annotations_


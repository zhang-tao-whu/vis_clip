import os
import json

image_root = 'datasets/VIPSeg/VIPSeg_720P/images/'
test_file = 'datasets/VIPSeg/VIPSeg_720P/test.txt'

f = open(test_file, encoding='gbk')
test_video_ids = []
for line in f:
    test_video_ids.append(line.strip())

video_ids = os.listdir(image_root)

videos_infos = []
for video_id in test_video_ids:
    assert video_id in video_ids
    video_info = {'video_id': video_id}
    images_infos = []
    image_files = os.listdir(os.path.join(image_root, video_id))
    for image_file in image_files:
        images_infos.append({'id': image_file.split('.')[0], 'width': 1280, 'height': 720, 'file_name': image_file})
    video_info['images'] = images_infos
    videos_infos.append(video_info)

video_annotations = []
for video in videos_infos:
    video_annotation = {'video_id': video['video_id']}
    image_annotations = []
    for image in video['images']:
        image_annotations.append({'image_id': image['id'], 'file_name': image['file_name'], 'segments_info': []})
        



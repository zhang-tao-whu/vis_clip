import torch

weight = torch.load('./output/model_final.pth')

model_ = {}
for key in weight['model'].keys():
    if 'sem_seg_head.pixel_decoder.pixel_decoder.' in key:
        print(key)
        key_ = 'sem_seg_head.pixel_decoder.' + key[len('sem_seg_head.pixel_decoder.pixel_decoder.'):]
    else:
        key_ = key
    model_.update({key_: weight['model'][key]})

weight['model'] = model_
torch.save(weight, './output/model_final_converted.pth')
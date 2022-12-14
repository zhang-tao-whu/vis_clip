import torch

weight = torch.load('./output/model_final.pth')

model_ = {}
for key in weight['model'].keys():
    print(key)
    if 'sem_seg_head.pixel_decoder.' in key:
        key_ = 'sem_seg_head.' + key[len('sem_seg_head.pixel_decoder.'):]
    else:
        key_ = key
    model_.update({key_: weight['model'][key]})

weight['model'] = model_
torch.save(weight, './output/model_final_converted.pth')
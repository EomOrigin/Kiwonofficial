import os
import torch
import shutil
import numpy as np
from collections import OrderedDict

def save_checkpoint(state, gpus, is_best=False,
                    model_path = f'./ckpt/',
                    model_name = f'checkpoint.pth.tar'):
    
    if(len(gpus) > 1):
        old_state_dict = state['state_dict']
        
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('.module.')
            if (flag != -1):
                k = k.replace('.module.', '.')
            new_state_dict[k] = v
        state['state_dict'] = new_state_dict

    os.makedirs(model_path, exist_ok=True)
    torch.save(state, f'{model_path}/{model_name}')
    if is_best:
        shutil.copyfile(f'{model_path}/{model_name}', f'{model_path}/model_best.pth.tar')

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

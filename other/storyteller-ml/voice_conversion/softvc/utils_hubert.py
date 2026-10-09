# Torch Script Inference Hubert
import copy
from typing import Optional, Tuple
import random

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import torch.nn.utils as utils

from torch.nn.utils.weight_norm import WeightNorm
from typing import Dict, Any
from collections import OrderedDict

def remove_weight_norm(module):
    module_list = [mod for mod in module.children()]
    if len(module_list) == 0:
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook.remove(module)
                del module._forward_pre_hooks[k]
    else:
        for mod in module_list:
            remove_weight_norm(mod)


def modify_model(state_dict:Dict[str,Any]):
    new_state_dict = OrderedDict()
    for k in state_dict['acoustic-model'].keys():
        value = state_dict['acoustic-model'][k]
        name = k[7:] # remove `module.` because they used ddp training to save
        new_state_dict[name] = value
    return new_state_dict

import json
import pathlib

import torch
from torch import nn
from safetensors import safe_open

unet_mapping = json.load(open('unet_lora_mapping.json'))


class LoraLinear(torch.nn.Module):
    def __init__(self, old_linear, up=None, down=None):
        super().__init__()
        self.old_linear = old_linear
        self.up = up
        self.down = down
        self.alpha = 1.0
        self.new_linear = None

    def forward(self, x):
        if self.new_linear is None:
            self.new_linear = nn.Linear(self.down.shape[1], self.up.shape[0])
            self.new_linear.weight.data = self.up @ self.down
            self.new_linear = self.new_linear.to('cuda').to(torch.float32)
        x1 = self.new_linear(x) * (self.alpha / float(self.up[1].size()[0])) if self.alpha is not None else 1
        x2 = self.old_linear(x)
        return x1 + x2


class LoraConv2d(torch.nn.Module):
    def __init__(self, old_conv2d, up=None, down=None):
        super().__init__()
        self.old_conv2d = old_conv2d
        self.up = up
        self.down = down
        self.alpha = None
        self.new_conv2d = None

    def forward(self, x):
        if self.new_conv2d is None:
            self.new_conv2d = nn.Conv2d(self.down.shape[1], self.up.shape[0], 1)
            self.new_conv2d.weight.data = (self.up.squeeze() @ self.down.squeeze()).unsqueeze(-1).unsqueeze(-1)
            self.new_conv2d = self.new_conv2d.to('cuda').to(torch.float32)
        x1 = self.new_conv2d(x) * (self.alpha / float(self.up[1].size()[0])) if self.alpha is not None else 1
        x2 = self.old_conv2d(x)
        return x1 + x2


def _get_nested_attr(obj, attr_path):
    elements = attr_path.split('.')
    for elem in elements:
        if elem.isdigit():  # If the element is a digit, access it as an index
            obj = obj[int(elem)]
        else:  # Otherwise, access it as an attribute
            obj = getattr(obj, elem)
    return obj


def _set_nested_attr(obj, attr_path, value):
    elements = attr_path.split('.')
    for elem in elements[:-1]:
        if elem.isdigit():
            obj = obj[int(elem)]
        else:
            obj = getattr(obj, elem)
    setattr(obj, elements[-1], value)


def _replace_underscores(input_string, exceptions):
    # Split the input string by underscores
    parts = input_string.split('_')

    # Reconstruct the string, replacing underscores with dots
    # except in the specified exceptions
    output_parts = []
    skip_next = False
    for i, part in enumerate(parts):
        if skip_next:
            skip_next = False
            continue

        # Check if this part combined with the next part is in exceptions
        if i < len(parts) - 1 and '_'.join([part, parts[i + 1]]) in exceptions:
            output_parts.append('_'.join([part, parts[i + 1]]))
            skip_next = True
        else:
            output_parts.append(part)

    return '.'.join(output_parts)


def apply_lora(unet, text_encoder, lora_weights):
    for param_name, param_value in lora_weights.items():
        if 'unet' in param_name:
            # Determine layer type
            if 'lora_down' in param_name:
                layer_type = 'down'
            elif 'lora_up' in param_name:
                layer_type = 'up'
            elif 'alpha' in param_name:
                layer_type = 'alpha'
            else:
                raise ValueError(f'Unknown layer type for {param_name}')

            base_param_name = param_name.split('.')[0]
            replaced_name = _replace_underscores(base_param_name, exceptions=[
                'up_blocks', 'mid_block', 'down_blocks', 'transformer_blocks',
                'to_k', 'to_q', 'to_v', 'to_out', 'to_in', 'proj_in', 'proj_out',
            ])

            for diffusers_name, sd_name in unet_mapping.items():
                base_diffusers = diffusers_name.rsplit('.', 1)[0]
                base_sd = sd_name.rsplit('.', 1)[0]
                if base_diffusers in replaced_name:
                    replaced_name = replaced_name.replace(base_diffusers, base_sd)

            replaced_name = replaced_name.replace('lora.unet', 'diffusion_model')
            param_value = param_value.to('cuda').to(torch.float32)
            curModule = _get_nested_attr(unet, replaced_name)

            # Update module type if needed
            if isinstance(curModule, torch.nn.Linear):
                _set_nested_attr(unet, replaced_name, LoraLinear(curModule))
            elif isinstance(curModule, torch.nn.Conv2d):
                _set_nested_attr(unet, replaced_name, LoraConv2d(curModule))
            elif not isinstance(curModule, (LoraLinear, LoraConv2d)):
                raise ValueError(f'Unknown layer type: {type(curModule)}')

            if layer_type == 'down':
                _set_nested_attr(unet, replaced_name + '.down', param_value)
            elif layer_type == 'up':
                _set_nested_attr(unet, replaced_name + '.up', param_value)
            elif layer_type == 'alpha':
                _set_nested_attr(unet, replaced_name + '.alpha', param_value)
        elif 'te_text_model' in param_name:
            # Determine layer type
            if 'lora_down' in param_name:
                layer_type = 'down'
            elif 'lora_up' in param_name:
                layer_type = 'up'
            elif 'alpha' in param_name:
                layer_type = 'alpha'
            else:
                raise ValueError(f'Unknown layer type for {param_name}')

            base_param_name = param_name.split('.')[0]
            replaced_name = base_param_name.replace('lora_te', 'transformer')
            replaced_name = _replace_underscores(replaced_name, exceptions=[
                'text_model', 'self_attn', 'k_proj', 'v_proj', 'q_proj', 'out_proj',
            ])
            curModule = _get_nested_attr(text_encoder, replaced_name)
            if isinstance(curModule, torch.nn.Linear):
                _set_nested_attr(text_encoder, replaced_name, LoraLinear(curModule))
            elif not isinstance(curModule, LoraLinear):
                raise ValueError(f'Unknown layer type: {type(curModule)}')
            param_value = param_value.to('cuda').to(torch.float32)
            if layer_type == 'down':
                _set_nested_attr(text_encoder, replaced_name + '.down', param_value)
            elif layer_type == 'up':
                _set_nested_attr(text_encoder, replaced_name + '.up', param_value)
            elif layer_type == 'alpha':
                _set_nested_attr(text_encoder, replaced_name + '.alpha', param_value)


def load_lora(filepath):
    path = pathlib.Path(filepath)
    if path.suffix == '.safetensors':
        try:
            print(f'Loading LORA weights using safetensors from {filepath}')
            lora_weights = {}
            with safe_open(filepath, framework="pt", device='cuda') as f:
                for key in f.keys():
                    lora_weights[key] = f.get_tensor(key)
        except Exception as e:
            raise Exception(f'Error loading LORA weights: {str(e)}')
    else:
        try:
            print(f'Loading LORA weights using torch from {filepath}')
            lora_weights = torch.load(filepath, map_location='cuda')
        except Exception as e:
            raise Exception(f'Error loading LORA weights from {filepath}: {str(e)}')
    return lora_weights

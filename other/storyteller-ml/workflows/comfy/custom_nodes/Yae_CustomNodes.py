import sys
import os
import torch
from PIL import Image, ImageDraw, ImageOps
import math
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


class YaeCombineConditionings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "conditioning1": ("CONDITIONING", ),
                     "conditioning2": ("CONDITIONING", ),
                     },
                }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, **kwargs):
        res = []
        for k, v in kwargs.items():
            if v != 0:
                res += v

        return (res, )




NODE_CLASS_MAPPINGS = {
    "YaeCombineConditionings": YaeCombineConditionings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YaeCombineConditionings": "Yae's Combine Conditionings"
}





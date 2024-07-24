import os
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
from torchinfo import summary
from ptflops import get_model_complexity_info

from core.pipeline import Pipeline
from core.model.omnistitch import Model as omnistitch
from fvcore.nn import FlopCountAnalysis, flop_count_table


"""
if you want to check flops with omnistitch, we have to change forward function
delete parameter of img1 and set img1 to img0.copy (ex. forward(self, img0): img1 = img0.clone())
our network include the cupy calculation, so it was the best we could do :(

Result by 4 * 3 * width * height
batch size 4 means that Four instances are required to provide an omnidirectional view.
Computational complexity: 312.99 GMac, Number of parameters: 4.99 M
"""

def test_flops(model_name="omnistitch"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    # GV360 dataset image size = 480 * 576 ~ 480 * 784
    width, height = 480, 576
    skiplevel = 0
    img0 = torch.randn(4, 3, width, height)
    img0 = img0.to(device)
    img1 = torch.randn(4, 3, width, height)
    img1 = img1.to(device)
    PYR_LEVEL = math.ceil(math.log2((width+32)/480) + 3)
    SKIP_LEVEL = PYR_LEVEL-3
    print(f"pyr_level : {PYR_LEVEL}, skip_level : {SKIP_LEVEL}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    if model_name not in ("omnistitch"):
        raise ValueError("model_name must be one of ('omnistitch')")
    
    Model = omnistitch(PYR_LEVEL, skiplevel)
    input_shape = (3, height, width)
    macs, params = get_model_complexity_info(Model.to(device).eval(), input_shape, as_strings=True, print_per_layer_stat=True, verbose=True)
    print(f'Computational complexity: {macs}, and 1 GFlops = 2 * GMacs')
    print(f'Number of parameters: {params}')

if __name__ == "__main__":
    model_name = "omnistitch"
    test_flops(model_name)
    
# CUDA_VISIBLE_DEVICES=1 python3 -m scripts.flops
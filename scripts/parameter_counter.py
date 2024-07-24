import os
import sys
import torch
import torch.nn as nn

from core.model.vsla_model import Model as vsla_model
from core.model.omnistitch import Model as omnistitch

"""
The Omnistitch has 4989761 trainable parameters
The VSLA has 39610473 trainable parameters
"""

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    model = vsla_model()
    num_parameters = count_parameters(model)
    print(f"The Omnistitch has {num_parameters} trainable parameters")
    
# CUDA_VISIBLE_DEVICES=1 python3 -m scripts.parameter_counter
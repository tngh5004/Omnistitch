import torch
import math
import numpy
import torch.nn.functional as F
import torch.nn as nn
# from core.model.lpvs_vsla import Model as vsla_model
from core.model.omnistitch import Model as omnistitch

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    model = omnistitch()
    num_parameters = count_parameters(model)
    print(f"The omnistitch has {num_parameters} trainable parameters")
    
# The omnistitch has 4989761 trainable parameters
# The vsla_like has 39610473 trainable parameters
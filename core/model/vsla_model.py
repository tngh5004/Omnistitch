import torch
import math
import numpy
import torch.nn.functional as F
import torch.nn as nn

from ..utils import correlation
from ..model.softsplat import softsplat
from ..model import vsla_network_parts as vsla_parts

"""

Video Stitching for Linear Camera Arrays(VSLA) : https://arxiv.org/abs/1907.13622
Super slomo : https://github.com/avinashpaliwal/Super-SloMo
"""

#**************************************************************************************************#
# => Unified model
#**************************************************************************************************#
class Model(nn.Module):
    def __init__(self, pyr_level=1, nr_lvl_skipped=0):
        super(Model, self).__init__()
        self.flowComp = vsla_parts.UNet(6, 4)
        self.flowComp.to(torch.device("cuda"))
        self.ArbTimeFlowIntrp = vsla_parts.UNet(20, 5)
        self.ArbTimeFlowIntrp.to(torch.device("cuda"))

    def forward(self, img0, img1, pyr_level=1, nr_lvl_skipped=0):
        # Calculate flow between reference frames I0 and I1
        flowOut = self.flowComp(torch.cat((img0, img1), dim=1))
        
        # Extracting flows between I0 and I1 - F_0_1 and F_1_0
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]
        
        # scaling factor for VSLA
        batch_size, channels, height, width = F_0_1.size()
        scaling_factor_01 = torch.linspace(0, 1, steps=480, device=torch.device("cuda")).view(1, 1, 1, 480)
        scaling_factor_10 = torch.linspace(1, 0, steps=480, device=torch.device("cuda")).view(1, 1, 1, 480)
        
        F_t_0 = F_0_1 * scaling_factor_01.expand(batch_size, channels, height, width)
        F_t_1 = F_1_0 * scaling_factor_10.expand(batch_size, channels, height, width)
        
        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0 = softsplat.backwarp_(img0, F_t_0)
        g_I1_F_t_1 = softsplat.backwarp_(img1, F_t_1)
        
        # Calculate optical flow residuals and visibility maps
        intrpOut = self.ArbTimeFlowIntrp(torch.cat((img0, img1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
        
        # Extract optical flow residuals and visibility maps
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0
        
        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0_f = softsplat.backwarp_(img0, F_t_0_f)
        g_I1_F_t_1_f = softsplat.backwarp_(img1, F_t_1_f)
        
        # wCoeff = model.getWarpCoeff(trainFrameIndex, device)
        Ft_p = (0.5 * V_t_0 * g_I0_F_t_0_f + 0.5 * V_t_1 * g_I1_F_t_1_f) / (0.5 * V_t_0 + 0.5 * V_t_1)
        
        convenience = 0

        return Ft_p, flowOut, convenience

if __name__ == "__main__":
    pass
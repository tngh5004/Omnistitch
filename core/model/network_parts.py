import torch
import torch.nn as nn
from torch.nn import functional as F

def pad_to_input(x, i2):
    """Apply bottom and right padding caused Upsampling mismatch"""
    target_height = i2.size(-2) - x.size(-2)
    target_width = i2.size(-1) - x.size(-1)
    x = nn.functional.pad(x, (0, target_width, 0, target_height), mode='constant', value=0)
    return x

    
class DoubleConv(nn.Module):
    """Triple Convolution with LeakyReLU & BatchNorm"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace = False, negative_slope=0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace = False, negative_slope=0.1),
        )
        
    def forward(self, x):
        return self.double_conv(x)
    

class DoubleConv_PR(nn.Module):
    """Double Convolution with PReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=out_channels)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class OnebyOne_PR(nn.Module):
    """Double Convolution with PReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.onebyone_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=out_channels),
        )
    
    def forward(self, x):
        return self.onebyone_conv(x)

class TripleConv_LR(nn.Module):
    """Triple Convolution with LeakyReLU"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.triple_conv_lru = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
        
    def forward(self, x):
        return self.triple_conv_lru(x)

class Triple_Stair_Conv_LR(nn.Module):
    """Triple Convolution with LeakyReLU"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.triple_conv_lru = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2,
                    kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*4,
                    kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=in_channels*4, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
        
    def forward(self, x):
        return self.triple_conv_lru(x)
    
class Quadruple_Conv_LR(nn.Module):
    """Triple Convolution with LeakyReLU"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.quadruple_conv_lru = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
        
    def forward(self, x):
        return self.quadruple_conv_lru(x)
    
class TripleConv_PR(nn.Module):
    """Triple Convolution with PReLU"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.triple_conv_pru = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=stride, padding=1),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=out_channels),
        )
        
    def forward(self, x):
        return self.triple_conv_pru(x)
    

class Down(nn.Module):
    """Downscaling with stride_2 convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace = False, negative_slope=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace = False, negative_slope=0.1),
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Decoder with Upsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, (in_channels // 2))
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_ConvTrans2d(nn.Module):
    """Decoder with ConvTranspose2d"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4, stride=2, padding=1, bias=True),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=out_channels)
        )
        
    def forward(self, x):
        return self.upconv(x)

class InConv(nn.Module):
    """Enter the encoder network"""
    def __init__(self, out_channels, img_metric=False):
        super().__init__()

        if img_metric:
            img_metric = out_channels // 3
            # For img color consistency (3 to 12)
            self.conv_img = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            # For img background importance (1 to 4)
            self.conv_metric = nn.Conv2d(in_channels=1, out_channels=img_metric, kernel_size=3, stride=1, padding=1)
            out_channels = out_channels + img_metric
        else:
            metric_out = out_channels // 6
            # For img color consistency (3 to 12)
            self.conv_img = nn.Conv2d(in_channels=out_channels, out_channels=(out_channels // 2), kernel_size=3, stride=1, padding=1)
            # For img background importance (1 to 4)
            self.conv_metric = nn.Conv2d(in_channels=1, out_channels=metric_out, kernel_size=3, stride=1, padding=1)
            out_channels = (out_channels // 2) + metric_out
        self.R = nn.ReLU(inplace = False)
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.PR = nn.PReLU(num_parameters=out_channels)

    def forward(self, x1, x2):
        x2 = self.conv_img(x2)
        x1 = self.conv_metric(x1)
        x = torch.cat([self.R(x1), self.R(x2)], dim=1)
        x = self.conv(x)
        return self.PR(x)


class OutConv(nn.Module):
    """Exist the decoder network"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.outconv(x)
    

class Pred_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pred_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

class Metric_network(nn.Module):
    """Softmax-splatting for image & feature depth metric"""
    def __init__(self, n_channels=12, img_metric=True):
        super(Metric_network, self).__init__()
        
        self.n_channels = n_channels
        self.img_metric = img_metric
        
        if img_metric:
            in_channels = n_channels + (n_channels // 3)
        else:
            in_channels = (n_channels//2) + (n_channels // 6)
            
        self.inconv = (InConv(n_channels, img_metric))
        self.down1 = (TripleConv_LR(in_channels, in_channels * 2, stride=2))
        self.down2 = (TripleConv_LR(in_channels * 2, in_channels * 2, stride=2))
        self.up1 = (Up(in_channels * 4 , in_channels))
        self.up2 = (Up(in_channels * 2, in_channels))
        self.outconv = (OutConv(in_channels, 1))

    def forward(self, i1, i2):
        x1 = self.inconv(i1, i2)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        if x.size()[2:] != i2.size()[2:]:
            pad_to_input(x, i2)
        return self.outconv(x)

if __name__=='__main__':
    pass
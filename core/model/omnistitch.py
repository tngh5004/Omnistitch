import torch
import math
import numpy
import torch.nn.functional as F
import torch.nn as nn

from ..utils import correlation
from ..model.softsplat import softsplat
from ..model import network_parts

"""
Due to the tight page limit of ACM MM, the detailed implementation is described in comments. 
By the way, the comments are for understanding and there is no difference in the main content.
"""

#**************************************************************************************************#
# => Feature Extraction
#**************************************************************************************************#
class FeatureExtractor(nn.Module):
        """
        The feature encoder basically consists of a three-stage(fixed) feature pyramid 
        shared by the flow estimator and the synthesis network.
        """
        def __init__(self):
                super(FeatureExtractor, self).__init__()
                self.conv_stage0 = network_parts.Triple_Stair_Conv_LR(in_channels=3, out_channels=24)
                self.conv_stage1 = network_parts.TripleConv_LR(in_channels=24, out_channels=48, stride=2)
                self.conv_stage2 = network_parts.TripleConv_LR(in_channels=48, out_channels=96, stride=2)

        def forward(self, img):
                C0 = self.conv_stage0(img)
                C1 = self.conv_stage1(C0)
                C2 = self.conv_stage2(C1)
                return [C0, C1, C2]


#**************************************************************************************************#
# => Flow Estimation
#**************************************************************************************************#
class FlowEstimator(nn.Module):
        """
        Bi-directional optical flow estimator
        1) construct partial cost volume with the CNN features from the stage 2 of the feature pyramid;
        2) estimate bi-directional flows, by feeding cost volume, CNN features for
        both warped images, CNN feature and estimated flow from previous iteration.
        """
        def __init__(self):
                super(FlowEstimator, self).__init__()
                # (4*2 + 1) ** 2 + 96 * 2 + 96 + 4 = 373
                self.conv_layer1 = nn.Sequential(
                        nn.Conv2d(in_channels=373, out_channels=224,
                        kernel_size=1, stride=1, padding=0),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.conv_layer2 = nn.Sequential(
                        nn.Conv2d(in_channels=224, out_channels=192,
                        kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.conv_layer3 = nn.Sequential(
                        nn.Conv2d(in_channels=192, out_channels=160,
                        kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.conv_layer4 = nn.Sequential(
                        nn.Conv2d(in_channels=160, out_channels=128,
                        kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.conv_layer5 = nn.Sequential(
                        nn.Conv2d(in_channels=128, out_channels=96,
                        kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.conv_layer6 = nn.Sequential(
                        nn.Conv2d(in_channels=96, out_channels=4,
                        kernel_size=3, stride=1, padding=1))


        def forward(self, feature_L, feature_R, last_feat, last_flow, warping_network):
                warped_feature_L, warped_feature_R, feature_cost = warping_network('flow_estimation', last_flow, feature_L, feature_R)
                input_feat = torch.cat([feature_cost, warped_feature_L, warped_feature_R, last_feat, last_flow], 1)
                feat = self.conv_layer1(input_feat)
                feat = self.conv_layer2(feat)
                feat = self.conv_layer3(feat)
                feat = self.conv_layer4(feat)
                feat = self.conv_layer5(feat)
                flow = self.conv_layer6(feat)

                return flow, feat, warping_network




#**************************************************************************************************#
# => Frame Synthesis
#**************************************************************************************************#
class SynthesisNetwork(nn.Module):
        """
        Synthesize the two warped images into a single image.
        1) Synthesize stitched image, by feeding original image pair, warped image pair,
        estimated bi-directional flows and stitched image from previous iteration.
        """
        def __init__(self):
                super(SynthesisNetwork, self).__init__()
                input_channels = 9+4+6
                output_channels = 48
                self.encoder_conv = network_parts.DoubleConv_PR(in_channels=input_channels, out_channels=output_channels)
                self.encoder_down1 = network_parts.TripleConv_PR(in_channels=48 + 24 + 24, out_channels=output_channels * 2, stride=2)
                self.encoder_down2 = network_parts.TripleConv_PR(in_channels=96 + 48 + 48, out_channels=output_channels * 4, stride=2)
                self.decoder_up1 = network_parts.Up_ConvTrans2d(in_channels=192 + 96 + 96, out_channels=output_channels * 2)
                self.decoder_up2 = network_parts.Up_ConvTrans2d(in_channels=96 + 96, out_channels=output_channels)
                self.decoder_conv = network_parts.DoubleConv_PR(in_channels=48 + 48, out_channels=output_channels)
                self.pred = nn.Conv2d(in_channels=48, out_channels=5, kernel_size=3, stride=1, padding=1)

        def forward(self, last_i, i0, i1, c0_pyr, c1_pyr, bi_flow_pyr, warping_network):
                warped_img0, warped_img1, warped_c0, warped_c1, flow_0t_1t = warping_network("Synthetic", bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1)
                input_feat = torch.cat(
                        (last_i, warped_img0, warped_img1, i0, i1, flow_0t_1t), 1)
                s0 = self.encoder_conv(input_feat)
                s1 = self.encoder_down1(torch.cat((s0, warped_c0, warped_c1), 1))
                warped_c0, warped_c1 = warping_network("Synthetic", bi_flow_pyr[1], c0_pyr[1], c1_pyr[1])
                s2 = self.encoder_down2(torch.cat((s1, warped_c0, warped_c1), 1))
                warped_c0, warped_c1 = warping_network("Synthetic_feat", bi_flow_pyr[2], c0_pyr[2], c1_pyr[2])
                x = self.decoder_up1(torch.cat((s2, warped_c0, warped_c1), 1))
                x = self.decoder_up2(torch.cat((x, s1), 1))
                x = self.decoder_conv(torch.cat((x, s0), 1))

                # prediction
                refine = self.pred(x)
                refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
                refine_mask0 = torch.sigmoid(refine[:, 3:4])
                refine_mask1 = torch.sigmoid(refine[:, 4:5])
                merged_img = (warped_img0 * refine_mask0 + \
                        warped_img1 * refine_mask1)
                merged_img = merged_img / (refine_mask0 + \
                        refine_mask1)
                interp_img = merged_img + refine_res
                interp_img = torch.clamp(interp_img, 0, 1)

                extra_dict = {}
                extra_dict["refine_res"] = refine_res
                extra_dict["refine_mask0"] = refine_mask0
                extra_dict["refine_mask1"] = refine_mask1
                extra_dict["warped_img0"] = torch.clamp(warped_img0, 0, 1)
                extra_dict["warped_img1"] = torch.clamp(warped_img1, 0, 1)
                extra_dict["merged_img"] = merged_img

                return interp_img, extra_dict, warping_network

class WarpingNetwork(nn.Module):
        """
        It's used to warp features and images, and the warping technique is softmax splatting.
        1) It is used in the flow estimator of the model and in the synthesis network. 
        2) softmax splatting uses a network that estimates the depth metric, and we use the same network 
        with shared weights to efficiently manage parameters by looking only at images or features.
        3) The second encoder in the convolutional network uses average splatting 
        because there was no performance difference between the splatting techniques.
        """
        def __init__(self):
                super(WarpingNetwork, self).__init__()
                self.alpha_i = nn.Parameter(-torch.ones(1))
                self.alpha_f = nn.Parameter(-torch.ones(1))
                self.img_metric_network = network_parts.Metric_network(n_channels=12, img_metric=True)
                self.feat_metric_network = network_parts.Metric_network(n_channels=96, img_metric=False)
                
        def forward(self, Module, bi_flow, feature_L=None, feature_R=None, i0=None, i1=None):
                flow_0t = bi_flow[:, :2] * 0.5
                flow_1t = bi_flow[:, 2:4] * 0.5
                if (Module == 'flow_estimation'):
                        tenMetricleft = torch.nn.functional.l1_loss(input=feature_L, target=softsplat.backwarp_(tenIn=feature_R, tenFlow=flow_0t*0.25), reduction='none').mean([1], True) ### softmax
                        tenMetricright = torch.nn.functional.l1_loss(input=feature_R, target=softsplat.backwarp_(tenIn=feature_L, tenFlow=flow_1t*0.25), reduction='none').mean([1], True)
                        warped_feature_L = softsplat.FunctionSoftsplat(
                                tenInput=feature_L, tenFlow=flow_0t*0.25,
                                tenMetric=(self.alpha_f * self.feat_metric_network(tenMetricleft,feature_L)).clip(-20.0, 20.0), strType='softmax')
                        warped_feature_R = softsplat.FunctionSoftsplat(
                                tenInput=feature_R, tenFlow=flow_1t*0.25,
                                tenMetric=(self.alpha_f * self.feat_metric_network(tenMetricright,feature_R)).clip(-20.0, 20.0), strType='softmax')
                        feature_cost = F.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=warped_feature_L, tenSecond=warped_feature_R),negative_slope=0.1, inplace=False)
                        return warped_feature_L, warped_feature_R, feature_cost
                elif (Module == 'Synthetic_feat'):
                        tenMetricleft = torch.nn.functional.l1_loss(input=feature_L, target=softsplat.backwarp_(tenIn=feature_R, tenFlow=flow_0t), reduction='none').mean([1], True) ### softmax
                        tenMetricright = torch.nn.functional.l1_loss(input=feature_R, target=softsplat.backwarp_(tenIn=feature_L, tenFlow=flow_1t), reduction='none').mean([1], True)
                        warped_feature_L = softsplat.FunctionSoftsplat(
                                tenInput=feature_L, tenFlow=flow_0t,
                                tenMetric=(self.alpha_f * self.feat_metric_network(tenMetricleft,feature_L)).clip(-20.0, 20.0), strType='softmax')
                        warped_feature_R = softsplat.FunctionSoftsplat(
                                tenInput=feature_R, tenFlow=flow_1t,
                                tenMetric=(self.alpha_f * self.feat_metric_network(tenMetricright,feature_R)).clip(-20.0, 20.0), strType='softmax')
                        return warped_feature_L, warped_feature_R
                elif (Module == 'last_interp'):
                        tenMetricleft = torch.nn.functional.l1_loss(input=i0, target=softsplat.backwarp_(tenIn=i1, tenFlow=flow_0t), reduction='none').mean([1], True) ### softmax
                        tenMetricright = torch.nn.functional.l1_loss(input=i1, target=softsplat.backwarp_(tenIn=i0, tenFlow=flow_1t), reduction='none').mean([1], True)
                        warped_img0 = softsplat.FunctionSoftsplat(
                                tenInput=i0, tenFlow=flow_0t,
                                tenMetric=(self.alpha_i * self.img_metric_network(tenMetricleft,i0)).clip(-20.0, 20.0), strType='softmax')
                        warped_img1 = softsplat.FunctionSoftsplat(
                                tenInput=i1, tenFlow=flow_1t,
                                tenMetric=(self.alpha_i * self.img_metric_network(tenMetricright,i1)).clip(-20.0, 20.0), strType='softmax')
                        return warped_img0, warped_img1
                elif (Module == 'Synthetic'):
                        warped_feature_L = softsplat.FunctionSoftsplat(
                                tenInput=feature_L, tenFlow=flow_0t,
                                tenMetric=None, strType='average')
                        warped_feature_R = softsplat.FunctionSoftsplat(
                                tenInput=feature_R, tenFlow=flow_1t,
                                tenMetric=None, strType='average')
                        if (i0 is None):
                                return warped_feature_L, warped_feature_R
                        tenMetricleft = torch.nn.functional.l1_loss(input=i0, target=softsplat.backwarp_(tenIn=i1, tenFlow=flow_0t), reduction='none').mean([1], True) ### softmax
                        tenMetricright = torch.nn.functional.l1_loss(input=i1, target=softsplat.backwarp_(tenIn=i0, tenFlow=flow_1t), reduction='none').mean([1], True)
                        warped_img0 = softsplat.FunctionSoftsplat(
                                tenInput=i0, tenFlow=flow_0t,
                                tenMetric=(self.alpha_i * self.img_metric_network(tenMetricleft,i0)).clip(-20.0, 20.0), strType='softmax')
                        warped_img1 = softsplat.FunctionSoftsplat(
                                tenInput=i1, tenFlow=flow_1t,
                                tenMetric=(self.alpha_i * self.img_metric_network(tenMetricright,i1)).clip(-20.0, 20.0), strType='softmax')
                        flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
                        return warped_img0, warped_img1, warped_feature_L, warped_feature_R, flow_0t_1t
                        
                
        

#**************************************************************************************************#
# => Unified model
#**************************************************************************************************#
class Model(nn.Module):
        """
        The Unified model is designed to iterate over one pipeline for a (specified) number of pyramid levels.
        However, the flow estimator is only used repeatedly in the top three pyramids, 
        and if there are more pyramids than that, it interpolates the flow estimated in the previous pyramid.
        This structure is heavily inspired by UPR-NET, a flow-based interpolate model.
        Uniquely, In the last step of the entire pipeline, We made sure to set the flow to skip 
        (the minimum number of pyramids is 4) to minimize blurring artifacts.
        Because our tasks are harder to find flow in than the tasks covered by the interpolation model.
        """
        def __init__(self, pyr_level=4, nr_lvl_skipped=1):
                super(Model, self).__init__()
                self.pyr_level = pyr_level
                self.nr_lvl_skipped = nr_lvl_skipped
                self.feature_encoder = FeatureExtractor()
                self.motion_estimator = FlowEstimator()
                self.synthesis_network = SynthesisNetwork()
                self.warping_network = WarpingNetwork()
        
        def forward_one_lvl(self, img0, img1, last_feat, last_flow, last_interp=None, skip_me=False):
                # context feature extraction
                feat0_pyr = self.feature_encoder(img0)
                feat1_pyr = self.feature_encoder(img1)
                warping_network = self.warping_network

                # bi-directional flow estimation
                if not skip_me:
                        flow, feat, warping_network = self.motion_estimator(
                                feat0_pyr[-1], feat1_pyr[-1],
                                last_feat, last_flow, warping_network)
                else:
                        flow = last_flow
                        feat = last_feat

                # frame synthesis
                ## optical flow is estimated at 1/4 resolution
                ori_resolution_flow = F.interpolate(
                        input=flow, scale_factor=4.0,
                        mode="bilinear", align_corners=False)

                ## consturct 3-level flow pyramid for synthesis network
                bi_flow_pyr = []
                tmp_flow = ori_resolution_flow
                bi_flow_pyr.append(tmp_flow)
                for i in range(2):
                        tmp_flow = F.interpolate(
                                input=tmp_flow, scale_factor=0.5,
                                mode="bilinear", align_corners=False) * 0.5
                        bi_flow_pyr.append(tmp_flow)

                ## merge warped frames as initial interpolation for frame synthesis
                if last_interp is None:
                        with torch.no_grad():
                                warped_img0, warped_img1 = warping_network("last_interp", bi_flow=ori_resolution_flow, i0=img0, i1=img1)
                        last_interp = (warped_img0 * 0.5) +  (warped_img1 * 0.5)
                ## do synthesis
                interp_img, extra_dict, warping_network = self.synthesis_network(
                        last_interp, img0, img1, feat0_pyr, feat1_pyr, bi_flow_pyr, warping_network)
                
                return flow, feat, interp_img, extra_dict

        def forward(self, img0, img1, pyr_level=None, nr_lvl_skipped=None):
                if pyr_level is None: pyr_level = self.pyr_level
                if nr_lvl_skipped is None: nr_lvl_skipped = self.nr_lvl_skipped
                N, _, H, W = img0.shape
                bi_flows = []
                interp_imgs = []
                skipped_levels = [] if nr_lvl_skipped == 0 else\
                        list(range(pyr_level))[::-1][-nr_lvl_skipped:]

                # The original input resolution corresponds to level 0.
                for level in list(range(pyr_level))[::-1]:
                        # scale down to the current level's
                        if level != 0:
                                scale_factor = 1 / 2 ** level
                                img0_this_lvl = F.interpolate(
                                        input=img0, scale_factor=scale_factor,
                                        mode="bilinear", align_corners=False)
                                img1_this_lvl = F.interpolate(
                                        input=img1, scale_factor=scale_factor,
                                        mode="bilinear", align_corners=False)
                        else:
                                img0_this_lvl = img0
                                img1_this_lvl = img1

                        # skip motion estimation, directly use up-sampled optical flow
                        skip_me = False
                        # the lowest-resolution pyramid level
                        if level == pyr_level - 1:
                                last_flow = torch.zeros(
                                        (N, 4, H // (2 ** (level+2)), W //(2 ** (level+2)))
                                        ).to(img0.device)
                                last_feat = torch.zeros(
                                        (N, 96, H // (2 ** (level+2)), W // (2 ** (level+2)))
                                        ).to(img0.device)
                                last_interp = None
                        # skip some levels for both motion estimation and frame synthesis
                        elif level in skipped_levels[:-1]:
                                continue
                        # last level (original input resolution), only skip motion estimation
                        elif (level == 0) and len(skipped_levels) > 0:
                                if len(skipped_levels) == pyr_level:
                                        last_flow = torch.zeros((N, 4, H // 4, W // 4)).to(img0.device)
                                        last_interp = None
                                else:
                                        resize_factor = 2 ** len(skipped_levels)
                                        last_flow = F.interpolate(
                                                input=flow, scale_factor=resize_factor,
                                                mode="bilinear", align_corners=False) * resize_factor
                                        last_interp = F.interpolate(
                                                input=interp_img, scale_factor=resize_factor,
                                                mode="bilinear", align_corners=False)
                                        skip_me = True
                        # last level (original input resolution), motion estimation + frame
                        # synthesis
                        else:
                                last_flow = F.interpolate(input=flow, scale_factor=2.0,
                                        mode="bilinear", align_corners=False) * 2
                                last_feat = F.interpolate(input=feat, scale_factor=2.0,
                                        mode="bilinear", align_corners=False) * 2
                                last_interp = F.interpolate(
                                        input=interp_img, scale_factor=2.0,
                                        mode="bilinear", align_corners=False)

                        flow, feat, interp_img, _extra_dict = self.forward_one_lvl(
                                img0_this_lvl, img1_this_lvl,
                                last_feat, last_flow, last_interp, skip_me=skip_me)
                        bi_flows.append(
                                F.interpolate(input=flow, scale_factor=4.0,
                                        mode="bilinear", align_corners=False))
                        interp_imgs.append(interp_img)

                # directly up-sample estimated flow to full resolution
                # with bi-linear interpolation
                bi_flow = F.interpolate(
                        input=flow, scale_factor=4.0,
                        mode="bilinear", align_corners=False)

                return interp_img, bi_flow, _extra_dict

if __name__ == "__main__":
        pass
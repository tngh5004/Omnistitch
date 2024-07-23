import os
import shutil
import math
import numpy as np
import cv2
from core.utils import flow_viz
import argparse
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core.pipeline import Pipeline
from core.dataset import GV360_wogt
from core.utils.pytorch_msssim import ssim_matlab
import lpips
import random

warnings.filterwarnings("ignore")

"""
1) This is code to measure the metric between two images by extracting only the overlapping regions of the two images 
when comparing the performance of existing image stitching algorithms.
2) Note that our model is an end-to-end model, so unlike other stitching models 
that split the warping and composition process, the warped image provided by the code below is a by-product of the process.
3) However, this code is implemented for the convenience of researchers in the field.

Although for images without ground truth, most papers use this comparison method.
but in my opinion it's a really weird metric to use. 
Simply replacing all overlap regions with the same pixels can give you the best score.
You can compare the numbers you see when running this code with the numbers you see when running benchmark_GV360.py.
Fortunately, many researchers are working on alternative metrics, so those interested should look them up :)
"""

def build_dir(dir):
        if os.path.exists(dir):
                shutil.rmtree(dir)
                os.makedirs(dir)
        else:
                os.makedirs(dir)


def evaluate(ppl, data_root, batch_size, nr_data_worker):
        dataset = GV360_wogt(data_root=data_root)
        val_data = DataLoader(dataset, batch_size=batch_size, num_workers=nr_data_worker, pin_memory=True)

        psnr_list = []
        ssim_list = []
        lpips_vgg_list = []
        
        precision = 0
        
        nr_val = val_data.__len__()
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)
        
        SAVE_MAKS_DIR = SAVE_DIR + '/mask'
        SAVE_LEFT_DIR = SAVE_DIR + '/left'
        SAVE_RIGHT_DIR = SAVE_DIR + '/right'
        SAVE_LEFT_OV_DIR = SAVE_DIR + '/left_ov'
        SAVE_RIGHT_OV_DIR = SAVE_DIR + '/right_ov'
        
        build_dir(SAVE_DIR)
        build_dir(SAVE_MAKS_DIR)
        build_dir(SAVE_LEFT_DIR)
        build_dir(SAVE_RIGHT_DIR)
        build_dir(SAVE_LEFT_OV_DIR)
        build_dir(SAVE_RIGHT_OV_DIR)
                
                
        for i, data in enumerate(val_data):
                data_gpu = data[0] if isinstance(data, list) else data
                data_gpu = data_gpu.to(DEVICE, non_blocking=True) / 255.
                img0 = data_gpu[:, :3]
                img1 = data_gpu[:, 3:6]
                
                n, c, h, w = img0.shape
                PYR_LEVEL = math.ceil(math.log2((w+32)/480) + 3)
                divisor = 2 ** (PYR_LEVEL-1+2)
                
                if (h % divisor != 0) or (w % divisor != 0):
                    ph = ((h - 1) // divisor + 1) * divisor
                    pw = ((w - 1) // divisor + 1) * divisor
                    padding = (0, pw - w, 0, ph - h)
                    img0 = F.pad(img0, padding, "constant", 0.5)
                    img1 = F.pad(img1, padding, "constant", 0.5)
                
                with torch.no_grad():
                        pred, _, extra_dict = ppl.inference_test(img0, img1, pyr_level=PYR_LEVEL, nr_lvl_skipped=PYR_LEVEL-3)
                        warped_img0 = extra_dict["warped_img0"]
                        warped_img0 = warped_img0[:, :, :h, :w]
                        warped_img1 = extra_dict["warped_img1"]
                        warped_img1 = warped_img1[:, :, :h, :w]
                        
                batch_psnr = []
                batch_ssim = []
                batch_lpips_vgg = []
                
                # iterate by batch_size
                for j in range(img0.shape[0]):
                        # stitched output
                        this_pred = pred[j]
                        interp_img = (this_pred * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(SAVE_DIR, f'pred_{precision}.png'), interp_img)
                        
                        # each warped image
                        this_warped_img0 = warped_img0[j]
                        interp_warpimg0 = (this_warped_img0 * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(SAVE_LEFT_DIR, f'warpimg0_{precision}.png'), interp_warpimg0)
                        this_warped_img1 = warped_img1[j]
                        interp_warpimg1 = (this_warped_img1 * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(SAVE_RIGHT_DIR, f'warpimg1_{precision}.png'), interp_warpimg1)
                        
                        # overlap mask
                        this_mask0 = torch.where(this_warped_img0 > 0.01, 1.0, 0.0)
                        this_mask1 = torch.where(this_warped_img1 > 0.01, 1.0, 0.0)
                        this_overlap_mask = torch.where((this_mask0 * this_mask1).sum(dim=0, keepdim=True) > 0.0, 1.0, 0.0)
                        interp_overlap_mask = (this_overlap_mask * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(SAVE_MAKS_DIR, f'overlapmask_{precision}.png'), interp_overlap_mask)
                        
                        # each warped image with overlapping regions
                        warped_left = this_warped_img0 * this_overlap_mask
                        overlap_warpimg0 = ((warped_left) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(SAVE_LEFT_OV_DIR, f'img_{precision}.png'), overlap_warpimg0)
                        warped_right = this_warped_img1 * this_overlap_mask
                        overlap_warpimg1 = ((warped_right) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(SAVE_RIGHT_OV_DIR, f'img_{precision}.png'), overlap_warpimg1)
                        
                        ssim = ssim_matlab(
                                warped_left.unsqueeze(0),
                                warped_right.unsqueeze(0)).cpu().numpy()
                        ssim = float(ssim)
                        ssim_list.append(ssim)
                        batch_ssim.append(ssim)
                        psnr = -10 * math.log10(
                                torch.mean(
                                        (warped_right - warped_left) * (warped_right - warped_left)
                                        ).cpu().data)
                        psnr_list.append(psnr)
                        batch_psnr.append(psnr)
                        
                        loss_vgg = loss_fn_vgg(warped_right, warped_left).cpu().numpy()
                        lpips_vgg_list.append(loss_vgg)
                        batch_lpips_vgg.append(loss_vgg)
                        precision += 1
                        
                print('batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}; lpips_vgg: {:.4f}'.format(i, nr_val, 
                        np.mean(batch_psnr), np.mean(batch_ssim), np.mean(batch_lpips_vgg)))
                
        psnr = np.array(psnr_list).mean()
        print('average psnr: {:.4f}'.format(psnr))
        ssim = np.array(ssim_list).mean()
        print('average ssim: {:.4f}'.format(ssim))
        lpips_vgg = np.array(lpips_vgg_list).mean()
        print('average lpips_vgg: {:.4f}'.format(lpips_vgg))

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='benchmark on GV360 dataset without reference')
        #**********************************************************#
        
        # => args for dataset and data loader
        parser.add_argument('--data_root', type=str, default='/home/sooho/workspace/data/GV360_test', \
                help='root dir of GV360 testset')
        parser.add_argument('--save_root', type=str, default='./demo/omnistitch/woGT',
                help='root dir of predicted result')
        parser.add_argument('--batch_size', type=int, default=4,
                help='batch size for data loader')
        parser.add_argument('--nr_data_worker', type=int, default=1,
                help='number of the worker for data loader')
        #**********************************************************#
        
        # => args for model
        parser.add_argument('--pyr_level', type=int, default=4,
                help='the number of pyramid levels of Omnistitch in testing')
        parser.add_argument('--model_name', type=str, default="omnistitch",
                help='model name, default is omnistitch')
        parser.add_argument('--model_file', type=str,
                default="./train-log-/Omnistitch/trained-models/model.pkl",
                help='weight of Omnistitch')
        #**********************************************************#
        
        # => init the benchmarking environment
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.demo = True
        torch.backends.cudnn.benchmark = True
        #**********************************************************#
        
        # => init the pipeline and start to benchmark
        args = parser.parse_args()
        SAVE_DIR = args.save_root
        
        model_cfg_dict = dict(
                load_pretrain = True,
                model_name = args.model_name,
                model_file = args.model_file
                )
        ppl = Pipeline(model_cfg_dict)
        
        print("Omnistitch benchmarking without reference...")
        evaluate(ppl, args.data_root, args.batch_size, args.nr_data_worker)
        print(f"{args.data_root}")
        print(f"{args.model_name}")
    
# CUDA_VISIBLE_DEVICES=0 python benchmark_woGT.py
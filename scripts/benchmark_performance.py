import os
import sys
import math
import numpy as np
import cv2
import argparse
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lpips
import random

from core.utils import flow_viz
from core.pipeline import Pipeline
from core.dataset import GV360
from core.utils.pytorch_msssim import ssim_matlab


warnings.filterwarnings("ignore")

def evaluate(ppl, data_root, batch_size, nr_data_worker=1):
        dataset = GV360(data_root=data_root, val=True)
        val_data = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=nr_data_worker, pin_memory=True)
        
        psnr_list = []
        ssim_list = []
        lpips_vgg_list = []
        
        nr_val = val_data.__len__()
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)
                
        for i, data in enumerate(val_data):
                data_gpu = data[0] if isinstance(data, list) else data
                data_gpu = data_gpu.to(DEVICE, non_blocking=True) / 255.
                
                img0 = data_gpu[:, :3]
                img1 = data_gpu[:, 3:6]
                gt = data_gpu[:, 6:9]
                
                n, c, h, w = img0.shape
                divisor = 2 ** (PYR_LEVEL-1+2)

                if (h % divisor != 0) or (w % divisor != 0):
                    ph = ((h - 1) // divisor + 1) * divisor
                    pw = ((w - 1) // divisor + 1) * divisor
                    padding = (0, pw - w, 0, ph - h)
                    img0 = F.pad(img0, padding, "constant", 0.5)
                    img1 = F.pad(img1, padding, "constant", 0.5)
                
                with torch.no_grad():
                        pred, _ = ppl.inference(img0, img1, pyr_level=PYR_LEVEL, nr_lvl_skipped=PYR_LEVEL - 3)
                        pred = pred[:, :, :h, :w]
                        
                batch_psnr = []
                batch_ssim = []
                batch_lpips_vgg = []
                
                curnum = i
                for j in range(gt.shape[0]):
                        this_gt = gt[j]
                        this_pred = pred[j]
                        interp_img = (this_pred * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        ssim = ssim_matlab(
                                this_pred.unsqueeze(0),
                                this_gt.unsqueeze(0)).cpu().numpy()
                        ssim = float(ssim)
                        ssim_list.append(ssim)
                        batch_ssim.append(ssim)
                        psnr = -10 * math.log10(
                                torch.mean(
                                        (this_gt - this_pred) * (this_gt - this_pred)
                                        ).cpu().data)
                        psnr_list.append(psnr)
                        batch_psnr.append(psnr)
                        
                        loss_vgg = loss_fn_vgg(this_gt, this_pred).cpu().numpy()
                        lpips_vgg_list.append(loss_vgg)
                        batch_lpips_vgg.append(loss_vgg)
                
                print('batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}; lpips_vgg: {:.4f}'.format(i, nr_val,
                np.mean(batch_psnr), np.mean(batch_ssim), np.mean(batch_lpips_vgg)))

        psnr = np.array(psnr_list).mean()
        print('average psnr: {:.4f}'.format(psnr))
        ssim = np.array(ssim_list).mean()
        print('average ssim: {:.4f}'.format(ssim))
        lpips_vgg = np.array(lpips_vgg_list).mean()
        print('average lpips_vgg: {:.4f}'.format(lpips_vgg))

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='benchmark on GV360 without image')
        #**********************************************************#
        
        # => args for dataset and data loader
        parser.add_argument('--data_root', type=str, default='/home/sooho/workspace/data/GV360_test', \
                help='root dir of GV360 testset')
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
        PYR_LEVEL = args.pyr_level
        
        model_cfg_dict = dict(
                load_pretrain = True,
                model_name = args.model_name,
                model_file = args.model_file,
                pyr_level = args.pyr_level,
                nr_lvl_skipped = args.pyr_level - 3 
                )
        ppl = Pipeline(model_cfg_dict)
        
        print("Omnistitch benchmarking...")
        evaluate(ppl, args.data_root, args.batch_size, args.nr_data_worker)
        print(f"{args.model_file}")
        print(f"{args.data_root}")
        print(f"{args.model_name}")

# CUDA_VISIBLE_DEVICES=1 python3 -m scripts.benchmark_performance
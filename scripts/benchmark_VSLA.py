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
from core.dataset import VSLA_SRMTEST
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

"""
The VSLA model uses the trained weights provided by super slomo. 
The valid difference between the models is that they use a scaling factor for the predicted flow. 
For the algorithm and training method, please refer to the VSLA paper. 
Hopefully, the VSLA_SRMTEST class in dataset.py and this file (benchmark_VSLA.py) will help you understand.
Video Stitching for Linear Camera Arrays(VSLA) : https://arxiv.org/pdf/1907.13622
Super Slomo : https://github.com/avinashpaliwal/Super-SloMo
"""

def evaluate(ppl, data_root, batch_size, nr_data_worker=1):
        dataset = VSLA_SRMTEST(data_root=data_root, crop_size=192, val=True)
        val_data = DataLoader(dataset, batch_size=batch_size, num_workers=nr_data_worker, pin_memory=True)

        psnr_list = []
        ssim_list = []
        lpips_vgg_list = []
        
        precision = 0
        
        nr_val = val_data.__len__()
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)
        
        SAVE_DIR = '/home/sooho/workspace/demo/benchmark_result_VSLA_00'
        if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
                
        for i, data in enumerate(val_data):
                data_gpu = data[0] if isinstance(data, list) else data
                data_gpu = data_gpu.to(DEVICE, non_blocking=True) / 255.

                img0 = data_gpu[:, :3]
                img1 = data_gpu[:, 3:6]
                gt = data_gpu[:, 6:9]
                y = 480 // 2
                w = 192 // 2
                left_cropped_image = data[:,:3]
                left_cropped_image_ = left_cropped_image[:,:,:,:y-w]
                img0 = img0[:,:,:,y-w:y+w]
                right_cropped_image = data[:,3:6]
                right_cropped_image_ = right_cropped_image[:,:,:,y+w:]
                img1 = img1[:,:,:,y-w:y+w]
                
                with torch.no_grad():
                        pred, _ = ppl.inference(img0, img1, pyr_level=1, nr_lvl_skipped=0)

                batch_psnr = []
                batch_ssim = []
                batch_lpips_vgg = []
                
                for j in range(gt.shape[0]):
                        this_gt = gt[j]
                        this_pred = pred[j]
                        interp_img = (this_pred * 255).byte().cpu().numpy().transpose(1, 2, 0)
                        left_cropped_image = (left_cropped_image_[j]).byte().permute(1, 2, 0)
                        right_cropped_image = (right_cropped_image_[j]).byte().permute(1, 2, 0)
                        
                        wholeimage = np.concatenate([left_cropped_image, interp_img],axis=1)
                        wholeimage = np.concatenate([wholeimage, right_cropped_image],axis=1)
                        wholeimage_tensor = torch.from_numpy(wholeimage.transpose(2, 0, 1)).float().to(this_gt.device) / 255.
                        
                        #cv2.imwrite(os.path.join(SAVE_DIR, f'pred_{precision}.png'), wholeimage)
                        ssim = ssim_matlab(
                                wholeimage_tensor.unsqueeze(0),
                                this_gt.unsqueeze(0)).cpu().numpy()
                        ssim = float(ssim)
                        ssim_list.append(ssim)
                        batch_ssim.append(ssim)
                        psnr = -10 * math.log10(
                                torch.mean(
                                        (this_gt - wholeimage_tensor) * (this_gt - wholeimage_tensor)
                                        ).cpu().data)
                        psnr_list.append(psnr)
                        batch_psnr.append(psnr)
                        
                        loss_vgg = loss_fn_vgg(this_gt, wholeimage_tensor).cpu().numpy()
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
        parser = argparse.ArgumentParser(description='benchmark on GV360 dataset')
        #**********************************************************#
        # => args for dataset and data loader
        parser.add_argument('--data_root', type=str, default='/home/sooho/workspace/data/GV360_test', \
                help='root dir of GV360 testset')
        parser.add_argument('--save_root', type=str, default='./demo/omnistitch/GV360',
                help='root dir of predicted result')
        parser.add_argument('--batch_size', type=int, default=1,
                help='batch size for data loader')
        parser.add_argument('--nr_data_worker', type=int, default=1,
                help='number of the worker for data loader')
        #**********************************************************#
        # => args for model
        parser.add_argument('--pyr_level', type=int, default=1,
                help='Deprecated, but used for code uniformity')
        parser.add_argument('--model_name', type=str, default="vsla_like",
                help='model name, one of (omnistitch, vsla_like)')
        # We do not provide a weight file, but we do provide code that can be trained
        parser.add_argument('--model_file', type=str,
                default="./train-log-/VSLA_l2vgg_crop192/trained-models/model.pkl",
                help='weight of VSLA_like')

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

        model_cfg_dict = dict(
                load_pretrain = True,
                model_size = args.model_size,
                model_file = args.model_file
                )
        ppl = Pipeline(model_cfg_dict)

        # resolution-aware parameter for inference
        PYR_LEVEL = args.pyr_level

        print("benchmarking on LPSFOV...")
        evaluate(ppl, args.data_root, args.batch_size, args.nr_data_worker)
        print(f"{args.data_root}")
        print(f"{args.model_size}")
    
# CUDA_VISIBLE_DEVICES=1 python3 -m scripts.benchmark_VSLA
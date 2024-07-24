import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import math
from importlib import import_module
from torch.nn import functional as F

from core.utils import flow_viz
from core.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_exp_env():
        if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
        if os.path.exists(SAVE_DIR):
                shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR)
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.demo = True


def interp_imgs(ppl, ori_img0, ori_img1):
        img0 = (torch.tensor(ori_img0.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
        img1 = (torch.tensor(ori_img1.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
        
        n, c, h, w = img0.shape
        divisor = 2 ** (PYR_LEVEL-1+2)
        
        if (h % divisor != 0) or (w % divisor != 0):
                ph = ((h - 1) // divisor + 1) * divisor
                pw = ((w - 1) // divisor + 1) * divisor
                padding = (0, pw - w, 0, ph - h)
                img0 = F.pad(img0, padding, "constant", 0.5)
                img1 = F.pad(img1, padding, "constant", 0.5)
                
        print("\nInitialization is OK! Begin to interp images...")
        
        interp_img, bi_flow, _extra_dict = ppl.inference_test(img0, img1, pyr_level=PYR_LEVEL, nr_lvl_skipped=PYR_LEVEL-3)
        interp_img = interp_img[:, :, :h, :w]
        bi_flow = bi_flow[:, :, :h, :w]

        overlay_input = (ori_img0 * 0.5 + ori_img1 * 0.5).astype("uint8")
        interp_img = (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        bi_flow = bi_flow[0].cpu().numpy().transpose(1, 2, 0)
        flow01 = bi_flow[:, :, :2]
        flow10 = bi_flow[:, :, 2:]
        flow01 = flow_viz.flow_to_image(flow01, convert_to_bgr=True)
        flow10 = flow_viz.flow_to_image(flow10, convert_to_bgr=True)
        bi_flow = np.concatenate([flow01, flow10], axis=1)
        
        cv2.imwrite(os.path.join(SAVE_DIR, '0-img0.png'), ori_img0)
        cv2.imwrite(os.path.join(SAVE_DIR, '1-img1.png'), ori_img1)
        cv2.imwrite(os.path.join(SAVE_DIR, '2-overlay-input.png'), overlay_input)
        cv2.imwrite(os.path.join(SAVE_DIR, '3-interp-img.png'), interp_img)
        cv2.imwrite(os.path.join(SAVE_DIR, '4-bi-flow.png'), bi_flow)
        if args.check_warped_img == True:
                warped_img0 = _extra_dict["warped_img0"][:, :, :h, :w]
                warped_img1 = _extra_dict["warped_img1"][:, :, :h, :w]
                warped_img0 = (warped_img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                warped_img1 = (warped_img1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                cv2.imwrite(os.path.join(SAVE_DIR, '5-warped-img0.png'), warped_img0)
                cv2.imwrite(os.path.join(SAVE_DIR, '6-warped-img1.png'), warped_img1)
        
        print("\nInterpolation is completed! Please see the results in %s" % (SAVE_DIR))


if __name__ == "__main__":
        parser = argparse.ArgumentParser(
                description="interpolate for given pair of images")
        parser.add_argument("--img0", type=str, required=True,
                help="file path of the left image")
        parser.add_argument("--img1", type=str, required=True,
                help="file path of the right image")
        parser.add_argument("--save_dir", type=str,
                default="./demo/output",
                help="dir to save interpolated frame")
        parser.add_argument('--model_name', type=str, default="omnistitch", # lpvs_K lpvs_H
                help='model name')
        parser.add_argument('--model_file', type=str,
                default="./train-log-/Omnistitch/trained-models/model.pkl",
                help='weight of Ours')
        parser.add_argument('--check_warped_img', type=bool,
                default=False,
                help='if you want to check each warped image without composition')

        args = parser.parse_args()

        #**********************************************************#
        # => parse args and init the training environment
        # global variable
        IMG0 = args.img0
        IMG1 = args.img1
        SAVE_DIR = args.save_dir

        # init env
        init_exp_env()

        #**********************************************************#
        # => read input frames and calculate the number of pyramid levels
        ori_img0 = cv2.imread(IMG0)
        ori_img1 = cv2.imread(IMG1)
        if ori_img0.shape != ori_img1.shape:
                ValueError("Please ensure that the input frames have the same size!")
        width = ori_img0.shape[1]
        PYR_LEVEL = math.ceil(math.log2((width+32)/480) + 3)

        #**********************************************************#
        # => init the pipeline and interpolate images
        model_cfg_dict = dict(
                load_pretrain = True,
                model_name = args.model_name,
                model_file = args.model_file
                )

        ppl = Pipeline(model_cfg_dict)
        ppl.eval()
        interp_imgs(ppl, ori_img0, ori_img1)

'''
CUDA_VISIBLE_DEVICES=1 python3 -m scripts.stitch_imgs \
--img0 /home/sooho/workspace/data/GV360_test/test_14m/00000_LD_L.jpg \
--img1 /home/sooho/workspace/data/GV360_test/test_14m/00000_LD_B.jpg \
--save_dir ./demo/output_d0
'''
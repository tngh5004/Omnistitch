import os
import sys
import torch
import time
import math

from core.pipeline import Pipeline

"""
The runtime batch_size is 4. This is because the model's goal is to create an omnidirectional view.
We used the RTX 4090 to measure the runtime of our model and other models. (CUDA Toolkit Version : 12.0)
In our main paper and rebuttal, the average runtime is written as 0.12s ± 0.02 because of the below result.
image size : 480 * 576 (Smallest size of GV360 testset) = average runtime: 0.110116 seconds
image size : 480 * 640 (Average size of GV360 testset) = average runtime: 0.127210 seconds
image size : 480 * 768 (Largest size of GV360 testset) = average runtime: 0.131544 seconds
"""

def test_runtime(model_name="omnistitch"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    
    width, height = 480, 640
    skiplevel = 0
    img0 = torch.randn(4, 3, width, height)
    img0 = img0.to(device)
    img1 = torch.randn(4, 3, width, height)
    img1 = img1.to(device)
    PYR_LEVEL = math.ceil(math.log2((width+32)/480) + 3)
    print(f"pyr_level : {PYR_LEVEL}, skip_level : {PYR_LEVEL - 3}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    if model_name not in ("omnistitch", "vsla_like"):
        raise ValueError("model_name must be one of ('omnistitch', 'vsla_like')")
    else:
        model_cfg_dict = dict(
                model_name = model_name,
                pyr_level=PYR_LEVEL,
                nr_lvl_skipped=PYR_LEVEL - 3,
                load_pretrain=False,
                )

    ppl = Pipeline(model_cfg_dict)
    ppl.device()
    ppl.eval()

    with torch.no_grad():
        for i in range(100):
            _, _ = ppl.inference(img0, img1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_stamp = time.time()
        for i in range(100):
            _, _ = ppl.inference(img0, img1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("average runtime: %4f seconds" % \
                ((time.time() - time_stamp) / 100))

if __name__ == "__main__":
    model_name = "omnistitch"
    test_runtime(model_name)
    
# CUDA_VISIBLE_DEVICES=1 python3 -m scripts.runtime
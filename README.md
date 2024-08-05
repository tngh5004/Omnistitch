# On updating...

# Omnistitch: Depth-aware Stitching Framework for Omnidirectional Vision with Multiple Cameras
This project is the official implementation of our ACM MM 2024 paper, OmniStitch: Depth-aware Stitching Framework for Omnidirectional Vision with Multiple Cameras

## Abstract
Omnidirectional vision systems provide a 360-degree panoramic view, enabling full environmental awareness in various fields, such as Advanced Driver Assistance Systems (ADAS) and Virtual Reality (VR). Existing omnidirectional stitching methods rely on a single specialized 360-degree camera. However, due to hardware limitations such as high mounting heights and blind spots, adapting these methods to vehicles of varying sizes and geometries is challenging. These challenges include limited generalizability due to the reliance on predefined stitching regions for fixed camera arrays, performance degradation from distance parallax leading to large depth differences, and the absence of suitable datasets with ground truth for multi-camera omnidirectional systems. To overcome these challenges, we propose a novel omnidirectional stitching framework and a publicly available dataset tailored for varying distance scenarios with multiple cameras. The framework, referred to as OmniStitch, consists of a Stitching Region Maximization (SRM) module for automatic adaptation to different vehicles with multiple cameras and a Depth-Aware Stitching (DAS) module to handle depth differences caused by distance parallax between cameras. In addition, we create and release an omnidirectional stitching dataset, called GV360, which provides ground truth images that maintain the perspective of the 360-degree FOV, designed explicitly for vehicle-agnostic systems. Extensive evaluations of this dataset demonstrate that our framework outperforms state-of-the-art stitching models, especially in handling varying distance parallax.

## Usage
### Python and Cuda environment
```
$ conda create -n Omnistitch python=3.9
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
$ pip install cupy-cuda12x
$ pip install loguru opencv_python scipy tensorboard tqdm lpips
```
For reference, [Pytorch](https://pytorch.org/get-started/locally/), [Cupy](https://docs.cupy.dev/en/stable/install.html)

### Train & Test
```
# Example for 3 GPU training
$ CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 --master_port=10000 train.py --exp_name test --resume True
```

```
# Example for 1 GPU training
$ CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=10000 train.py --exp_name test --resume True
```

```
# Example for tensorboard
$ tensorboard --logdir=/yourpath/train-log-/test/tensorboard
```

### Benchmark for performance on GV360 dataset & others
```
# Benchmark on GV360
$ CUDA_VISIBLE_DEVICES=1 python3 -m scripts.benchmark_performance
```

```
# Benchmark & Generate output
CUDA_VISIBLE_DEVICES=1 python3 -m scripts.benchmark_GV360
```

```
# Benchmark on Real dataset. (without ground truth)
# Comparing only the overlapping regions of the input images
CUDA_VISIBLE_DEVICES=1 python3 -m scripts.benchmark_GV360
```

### Testing Code for parameter, time, and complexity
```
# parameter
$ CUDA_VISIBLE_DEVICES=0 python3 -m scripts.parameter_counter
```

```
# runtime
$ CUDA_VISIBLE_DEVICES=0 python3 -m scripts.runtime
```

```
# flops
$ pip install ptflops fvcore
$ CUDA_VISIBLE_DEVICES=0 python3 -m scripts.flops
```
if you want to check flops with omnistitch, we have to change forward function of Model class.  
Delete parameter of img1 and set img1 to img0.copy (ex. forward(self, img0): img1 = img0.clone())  
Since we use the cupy calculation, it was the best we could :)

## GV360 dataset
We uploaded all of our GV360 datasets to [Hugging Face](https://huggingface.co/datasets/tngh5004/GV360)  
This dataset is for omnidirectional image stitching or video stitching.  
Also, it was collected with autonomous vehicles and ADAS in mind.  
The dataset was collected using [CARLA](https://github.com/carla-simulator/carla.git) simulator, powered by Unreal Engine.  

### Generation code with CARLA simulator
The code to generate the dataset is also available at this [link](https://github.com/tngh5004/Omnistitch/tree/main/scripts/GV360_generation_scripts).  
Please note that we only collected our dataset in a Windows environment.

## Acknowledgement
We borrow some codes from [UPR-NET](https://github.com/srcn-ivl/UPR-Net.git), [softmax-splatting](https://github.com/sniklaus/softmax-splatting.git), and [CARLA](https://github.com/carla-simulator/carla.git). We thank the authors for their excellent work. When using our code, please also pay attention to the licenses of UPR-NET, softmax-splatting, and CARLA.

## Citation

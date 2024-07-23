import cv2
import os
import json
import torch
import numpy as np
import random
from glob import glob
from PIL import ImageEnhance, Image
from torch.utils.data import DataLoader, Dataset
import fnmatch
from collections import defaultdict
cv2.setNumThreads(0)

# for training
class GV360(Dataset):
    def __init__(self, data_root, crop_size=480, val=False):
        self.data_root = data_root
        self.val = val
        self.crop_size = crop_size
        self.load_data()
    
    def __len__(self):
        return len(self.meta_data)
    
    def load_data(self):
        root_dirs = glob(os.path.join(self.data_root, '*'))
        # root_dirs[:] = (value for value in root_dirs \
        #                 if value != self.data_root + '/val')
        
        root_dirs.sort()
        frame_dict = defaultdict(tuple)
        index = 0
        
        for dir in root_dirs:
            img_files = glob(os.path.join(dir, "*.jpg"))
            img_files.sort()

            ld_files, rd_files, lu_files, ru_files = [], [], [], []
            
            ld_files = fnmatch.filter(img_files, '*LD*')
            rd_files = fnmatch.filter(img_files, '*RD*')
            lu_files = fnmatch.filter(img_files, '*LU*')
            ru_files = fnmatch.filter(img_files, '*RU*')

            for i in range(0, len(ld_files), 3):
                frame_dict[index] = (ld_files[i+2], ld_files[i+1], ld_files[i])
                index += 1
                frame_dict[index] = (rd_files[i+2], rd_files[i+1], rd_files[i])
                index += 1
                frame_dict[index] = (lu_files[i], lu_files[i+1], lu_files[i+2])
                index += 1
                frame_dict[index] = (ru_files[i], ru_files[i+1], ru_files[i+2])
                index += 1
        self.meta_data = frame_dict
    
    def randomcrop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1
    
    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = [os.path.join(imgpath[0]),
                    os.path.join(imgpath[1]),
                    os.path.join(imgpath[2])]
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1
    
    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        if self.val == False:
            # random crop
            img0, gt, img1 = self.randomcrop(img0, gt, img1, self.crop_size, self.crop_size)
            
            # random rotation
            if random.uniform(0, 1) < 0.5:
                rot_option = np.random.randint(1, 4)
                img0 = np.rot90(img0, rot_option)
                img1 = np.rot90(img1, rot_option)
                gt = np.rot90(gt, rot_option)
                
            # random channel reverse
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
                
            # random vertical flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)


# for evaulation that without reference, it used for overlap region comparison 
class GV360_wogt(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.load_data()
    
    def __len__(self):
        return len(self.meta_data)
    
    def load_data(self):
        root_dirs = glob(os.path.join(self.data_root, '*'))
        root_dirs.sort()
        frame_dict = defaultdict(tuple)
        index = 0
        
        for dir in root_dirs:
            img_files = glob(os.path.join(dir, "*.jpg"))
            img_files.sort()

            ld_files, rd_files, lu_files, ru_files = [], [], [], []
            
            ld_files = fnmatch.filter(img_files, '*LD*')
            rd_files = fnmatch.filter(img_files, '*RD*')
            lu_files = fnmatch.filter(img_files, '*LU*')
            ru_files = fnmatch.filter(img_files, '*RU*')

            for i in range(0, len(ld_files), 3):
                frame_dict[index] = (ld_files[i+2], ld_files[i])
                index += 1
                frame_dict[index] = (rd_files[i+2], rd_files[i])
                index += 1
                frame_dict[index] = (lu_files[i], lu_files[i+2])
                index += 1
                frame_dict[index] = (ru_files[i], ru_files[i+2])
                index += 1
        self.meta_data = frame_dict

    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = [os.path.join(imgpath[0]),
                    os.path.join(imgpath[1])]
        
        img0 = cv2.imread(imgpaths[0])
        img1 = cv2.imread(imgpaths[1])
        return img0, img1
    
    def __getitem__(self, index):
        img0, img1 = self.getimg(index)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1), 0)


# for VSLA-like approach
class VSLA_SRMTEST(Dataset):
    def __init__(self, data_root, crop_size, val=False):
        self.data_root = data_root
        self.val = val
        self.crop_size = crop_size
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        root_dirs = glob(os.path.join(self.data_root, '*'))
        
        root_dirs.sort()
        frame_dict = defaultdict(tuple)
        index = 0
        
        for dir in root_dirs:
            img_files = glob(os.path.join(dir, "*.jpg"))
            img_files.sort()

            ld_files, rd_files, lu_files, ru_files = [], [], [], []
            
            ld_files = fnmatch.filter(img_files, '*LD*')
            rd_files = fnmatch.filter(img_files, '*RD*')
            lu_files = fnmatch.filter(img_files, '*LU*')
            ru_files = fnmatch.filter(img_files, '*RU*')

            for i in range(0, len(ld_files), 3):
                frame_dict[index] = (ld_files[i+2], ld_files[i+1], ld_files[i])
                index += 1
                frame_dict[index] = (rd_files[i+2], rd_files[i+1], rd_files[i])
                index += 1
                frame_dict[index] = (lu_files[i], lu_files[i+1], lu_files[i+2])
                index += 1
                frame_dict[index] = (ru_files[i], ru_files[i+1], ru_files[i+2])
                index += 1
        self.meta_data = frame_dict

    def randomcrop(self, img0, gt, img1, w):
        # for augmentation
        # set 'h' to the minist height of the images in the dataset.
        ih, iw, _ = img0.shape
        y = iw // 2
        w = w // 2
        h = 576 
        x = np.random.randint(0, ih - h + 1)
        img0 = img0[x:x+h, y-w:y+w, :]
        img1 = img1[x:x+h, y-w:y+w, :]
        gt = gt[x:x+h, y-w:y+w, :]
        return img0, gt, img1
    
    def valcrop(self, img0, gt, img1, w):
        ih, iw, _ = img0.shape
        y = iw // 2
        w = w // 2
        img0 = img0[:, y-w:y+w, :]
        img1 = img1[:, y-w:y+w, :]
        gt = gt[:, y-w:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = [os.path.join(imgpath[0]),
                    os.path.join(imgpath[1]),
                    os.path.join(imgpath[2])]
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1
    
    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        if self.val == False:
            img0, gt, img1 = self.randomcrop(img0, gt, img1, self.crop_size)
        elif self.val == True:
            # without augmentation
            img0, gt, img1 = self.valcrop(img0, gt, img1, self.crop_size)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)
    


if  __name__ == "__main__":
    pass
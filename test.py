import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

BATCHSIZE_PER_CARD = 4

def pad_to_square_32(img):
    h, w, c = img.shape
    size = max(h, w)
    pad_size = ((size + 31) // 32) * 32  # smallest multiple of 32 >= size

    top = (pad_size - h) // 2
    bottom = pad_size - h - top
    left = (pad_size - w) // 2
    right = pad_size - w - left

    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, (h, w), (top, bottom, left, right)

def resize_mask_to_original(mask, original_size, pad_info):
    """
    Resize mask from padded size back to original image size
    
    Args:
        mask: numpy array of the mask from padded image
        original_size: tuple (height, width) of original image
        pad_info: tuple (top, bottom, left, right) padding information
    
    Returns:
        resized mask matching original image dimensions
    """
    orig_h, orig_w = original_size
    top, bottom, left, right = pad_info
    
    # Remove padding from mask
    mask_h, mask_w = mask.shape
    unpadded_mask = mask[top:mask_h-bottom, left:mask_w-right]
    
    # Resize to original dimensions if needed
    if unpadded_mask.shape != (orig_h, orig_w):
        resized_mask = cv2.resize(unpadded_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = unpadded_mask
    
    return resized_mask

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)
        img_padded, orig_size, pad_info = pad_to_square_32(img)
        
        img90 = np.array(np.rot90(img_padded))
        img1 = np.concatenate([img_padded[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        # Resize mask back to original image size
        mask2 = resize_mask_to_original(mask2, orig_size, pad_info)
        
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)
        img_padded, orig_size, pad_info = pad_to_square_32(img)
        
        img90 = np.array(np.rot90(img_padded))
        img1 = np.concatenate([img_padded[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        # Resize mask back to original image size
        mask2 = resize_mask_to_original(mask2, orig_size, pad_info)
        
        return mask2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)
        img_padded, orig_size, pad_info = pad_to_square_32(img)
        
        img90 = np.array(np.rot90(img_padded))
        img1 = np.concatenate([img_padded[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        # Resize mask back to original image size
        mask3 = resize_mask_to_original(mask3, orig_size, pad_info)
        
        return mask3
    
    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)
        img_padded, orig_size, pad_info = pad_to_square_32(img)
        
        img90 = np.array(np.rot90(img_padded))
        img1 = np.concatenate([img_padded[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        # Resize mask back to original image size
        mask3 = resize_mask_to_original(mask3, orig_size, pad_info)
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DinkNet34 segmentation model on an input image.")
    parser.add_argument('--img_path', type=str, default=None, help='Path to a single image')
    parser.add_argument('--ckpt', type=str, default='/content/log01_dink34.th', help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default='submits/log01_dink34/', help='Directory to save mask')
    args = parser.parse_args()

    solver = TTAFrame(DinkNet34)
    solver.load(args.ckpt)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.img_path:
        name = os.path.basename(args.img_path)
        mask = solver.test_one_img_from_path(args.img_path)
        mask[mask > 4.0] = 255
        mask[mask <= 4.0] = 0
        mask = np.concatenate([mask[:, :, None]] * 3, axis=2)
        out_path = os.path.join(args.out_dir, 'mask.png')
        cv2.imwrite(out_path, mask.astype(np.uint8))
        print(f"Mask saved to: {out_path}")
        print(f"Mask shape: {mask.shape}")
    else:
        # default batch mode
        source = 'dataset/valid/'
        val = os.listdir(source)
        tic = time()
        for i, name in enumerate(val):
            if i % 10 == 0:
                print(i / 10, '    ', '%.2f' % (time() - tic))
            tp = os.path.join(source, name)
            if tp.endswith((".png", ".jpg", ".tiff", ".tif")):
                mask = solver.test_one_img_from_path(tp)
                mask[mask > 4.0] = 255
                mask[mask <= 4.0] = 0
                mask = np.concatenate([mask[:, :, None]] * 3, axis=2)
                cv2.imwrite(os.path.join(args.out_dir, name[:-7] + 'mask.png'), mask.astype(np.uint8))

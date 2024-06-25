import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
import imageio
import sys 
sys.path.insert(1, './SINet-V2-main')
from utils.data_val import RGBtoGray_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
opt = parser.parse_args()

for _data_name in ['Image']:
    Images_path = './SINet-V2-main/Dataset/TrainValDataset/{}/'.format(_data_name)
    save_path = ('./SINet-V2-main/Dataset/TrainValDataset/GrayImages/')
    print("SINet_path",Images_path)
    print("save-path",save_path)

    os.makedirs(save_path, exist_ok=True)
    test_loader = RGBtoGray_dataset(Images_path, opt.testsize)

    for i in range(test_loader.size):
        RGBImage, name , index = test_loader.load_data()
        img = np.asarray(RGBImage)
        patch, channel, height, width = RGBImage.shape

        for c in range(channel):
          for h in range(height):
            for w in range(width):
              if img[:,c,h,w] <= 0.04045:
                img[:,c,h,w]=img[:,c,h,w] /12.92
              else:
                img[:,c,h,w]=((img[:,c,h,w]+0.055)/1.055)**2.4
         
        gray=0.2126*img[:,0,:,:] + 0.7152*img[:,1,:,:] + 0.0722*img[:,2,:,:]
        gray=gray[0,:,:]
        imageio.imwrite(save_path+'/'+name , gray)
        print(name)
        print(index,'of 10553 images in DUTS-TR')

#coding=utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_map(datapath):
    print(datapath)
    for name in os.listdir(datapath+'/TrainValDataset/GT'):
        mask = cv2.imread(datapath+'/TrainValDataset/GT/'+name,0)
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        if not os.path.exists(datapath+'/body-label/'):
            os.makedirs(datapath+'/body-label/')
        cv2.imwrite(datapath+'/body-label/'+name, body)

        if not os.path.exists(datapath+'/detail-label/'):
            os.makedirs(datapath+'/detail-label/')
        cv2.imwrite(datapath+'/detail-label/'+name, mask-body)


if __name__=='__main__':
    split_map('./SINet-V2-main/Dataset')


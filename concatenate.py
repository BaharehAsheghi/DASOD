import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

gridsize=8
path_img='./SINet-V2-main/Dataset/TrainValDataset/Image/'
path_body='./SINet-V2-main/Dataset/body-label/'
path_result='./SINet-V2-main/Dataset/TrainValDataset/ImageBL/'
if not os.path.exists(path_result):
  os.makedirs(path_result)
      
for img_name in os.listdir(path_img):
  print(img_name)

  img_orig = plt.imread(path_img + img_name)
  row, col, chan = img_orig.shape

  img_body= plt.imread(path_body + img_name[:-4]+'.png')*255

  image = np.zeros(shape=(row, col, chan+1), dtype=np.uint8)
  image[:,:,:3]=img_orig
  image[:,:,3]=img_body

  plt.imsave(path_result + img_name[:-4]+'.png', image, cmap='gray')

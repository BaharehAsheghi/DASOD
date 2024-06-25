# DASOD: Detail-Aware Salient Object Detection ##[IMAVIS 2024] ([Paper](https://www.sciencedirect.com/science/article/abs/pii/S0262885624002592))

## Download datasets and prepare the data
Download the following datasets and unzip them into the `Dataset` folder in SINet-V2-main:

* Training: [DUTS-TR](https://drive.google.com/file/d/1VtRvigy8fGpPb3hF-R-Zs8otNv1n9RG0/view?usp=sharing); the uploaded images are super-resolution images.

  Split ground-truth (GT) into body label and detail label using this command, which will be saved into `./SINet-V2-main/Dataset/body-label` and `./SINet-V2-main/Dataset/detail-label`
  ```
  python utils.py
  ```
  Concatenate super-resolution image with body label using this command, which will be saved into `./SINet-V2-main/Dataset/TrainValDataset/ImageBL`
  ```
  python concatenate.py
  ```

* Testing: [DUT-OMRON, DUTS-TE, ECSSD, HKU-IS, PASCAL-S](https://drive.google.com/file/d/1scvWyW7QgVU6keIuMnFpEIYFkU47ObdM/view?usp=sharing), and [SCAS](https://drive.google.com/file/d/1HWuo0xW9LWrKfRE91HABkLPzBGYaSDN2/view?usp=sharing)

** Note: If you get an error regarding the number of images or the Thumbs.db file during runtime, delete this file from all dataset folders using the following command (change the path for other folders):
  ```
  import os
  file_path = './SINet-V2-main/Dataset/TrainValDataset/Image/Thumbs.db'
  os.remove(file_path)
  ```

## Pseudo-mask generation stage

### Installation
We ran all the code in _Google Colab_ and installed these packages in the training phase.
```
pip install thop
pip install tensorboardX
pip install onnx
pip install timm
```
### Training and testing
Put [res2net101](https://drive.google.com/file/d/1MSGJ3XLCv6JWAItbi0FVFriX2ebSlxXW/view?usp=sharing) in the `.\SINet-V2-main\media\nercms\NERCMS\GepengJi\Medical_Seqmentation\CRANet\models` folder, and run these files in the `./SINet-V2-main` folder:
```
python MyTrain_Val.py
python MyTesting.py
```
The generated pseudo-masks are saved in the `./SINet-V2-main/res/SINet_V2` folder in the test phase.
## Refinement stage
### Data preparing
Run these files in the `./UR-COD` folder:
```
python loader_convert_rgbtogray.py
```
### Training and testing
```
python train.py 
python test.py 
```
The predicted saliency maps are saved in the `./UR-COD/results/UR-SINetv2` folder in the test phase.
## Results 
Predicted Saliency Maps: [DUT-OMRON, DUTS-TE, ECSSD, HKU-IS, PASCAL-S](https://drive.google.com/file/d/1Lag7Li1sPlVjGChJQFai9bwHbKvCaIiC/view?usp=sharing), and [SCAS](https://drive.google.com/file/d/1iMzKN_AeR0jBY_Razd5LnQ7HtgJPqqYK/view?usp=sharing)

## 😊
If you encounter any problems or experience difficulties while running this code, please do not hesitate to report issues in this repository or mail them to me.

My email is: bahareh.asheghi98@gmail.com

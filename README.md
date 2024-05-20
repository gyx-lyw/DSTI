# DSTI
official code for Dual-Branch Spatiotemporal Interaction Network for Video Crowd Counting. 

## Dataset
- Bus: [BaiduNetDisk](https://pan.baidu.com/s/1FR7PMrdhpNB2OgkY_QbbDw?pwd=ir6n), [GoogleDrive]().
- Canteen: [BaiduNetDisk](https://pan.baidu.com/s/18XtesjJTBolXMwHZFoazVw?pwd=yi7b), [GoogleDrive]().
- Classroom: [BaiduNetDisk](https://pan.baidu.com/s/1ZbD3aLNuu7syw86a7UQe-g?pwd=z3q8), [GoogleDrive](). 

## Install dependencies
torch >= 1.0 torchvision opencv numpy scipy.  

##  Take training and testing of Bus dataset for example:
1. Download Bus
2. Preprocess Bus to generate ground-truth density maps
```shell 
python generate_h5.py
```
3. Divide the last 10% of the training set into the validation set. The folder structure should look like this:
```shell 
Bus
├──train
    ├──1.img
    ├──1.h5
├──val
├──test
```
4. Train Bus
```shell 
python train_fuse.py (dataset path) (directory for saving model weights)
```
5. Test Bus
```shell 
python train_fuse.py (dataset path) (directory for saving model weights)
```
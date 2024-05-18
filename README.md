# Official code for Dual-Branch Spatiotemporal Interaction Network for Video Crowd Counting

### Install dependencies

torch >= 1.0 torchvision opencv numpy scipy  

###  Train and Test

1、 Pre-Process Data

```
python generate_h5.py
```

2、 Train model

```
python train_fuse.py --data_dir <directory of processed data> --save_dir <directory of log and model>
```

3、 Test Model
```
python test_fuse.py --data_dir <directory of processed data> --save_dir <directory of log and model>

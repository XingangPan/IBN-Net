## Instance-Batch Normalization Network

### Paper

Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. ["Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"](https://arxiv.org/abs/1807.09441), ECCV2018.  

### Introduction
<img align="middle" width="500" height="280" src="./utils/IBNNet.png">   

- IBN-Net carefully unifies instance normalization and batch normalization in a single deep network.  
- It provides an extremely simple way to increase both modeling and generalization capacity without adding model complexity.

### Requirements
- Pytorch 0.4.1 or higher

### Results

Top1/Top5 error on the ImageNet validation set are reported. You may get different results when training your models with different random seed.

| Model                     | origin         |  re-implementation      | IBN-Net     |
| -------------------       | ------------------ | ------------------ | ------------------ |
| DenseNet-121          | 25.0/-             | 24.96/7.85       | 24.47/7.25 [model](https://xingang.s3-ap-southeast-1.amazonaws.com/densenet121_ibn_a-e4af5cc1.pth)    |
| DenseNet-169          | 23.6/-              | 24.02/7.06      | 23.25/6.51 [model](https://xingang.s3-ap-southeast-1.amazonaws.com/densenet169_ibn_a-9f32c161.pth)    |
| ResNet-50                | 24.7/7.8          | 24.27/7.08       | 22.54/6.32  [model](https://xingang.s3-ap-southeast-1.amazonaws.com/resnet50_ibn_a-d9d0bb7b.pth)   |
| ResNet-101             | 23.6/7.1           | 22.48/6.23       | 21.39/5.59  [model](https://xingang.s3-ap-southeast-1.amazonaws.com/resnet101_ibn_a-59ea0ac6.pth)  |
| ResNeXt-101          | 21.2/5.6            | 21.31/5.74       | 20.88/5.42  [model](https://xingang.s3-ap-southeast-1.amazonaws.com/resnext101_ibn_a-6ace051d.pth)  |
| SE-ResNet-101       | 22.38/6.07        | 21.68/5.88       | 21.25/5.51   [model](https://xingang.s3-ap-southeast-1.amazonaws.com/se_resnet101_ibn_a-fabed4e2.pth)  |

### Before Start
1. Clone the repository  
    ```Shell
    git clone https://github.com/XingangPan/IBN-Net.git
    ```

2. Download [ImageNet](http://image-net.org/download-images) dataset (if you need to test or train on ImageNet). You may follow the instruction at [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) to process the validation set.

### Testing
1. Edit `test.sh`. Modify `model` and `data_path` to yours.  
    Options for `model`: resnet50_ibn_a, resnet50_ibn_b, resnet101_ibn_a, resnext101_ibn_a, se_resnet101_ibn_a, densenet121_ibn_a, densenet169_ibn_a.  
    
2. Run test script
    ```Shell
    sh test.sh
    ```
 
### Training
1. Edit `train.sh`. Modify `model` and `data_path` to yours.  
2. Run train script
    ```Shell
    sh train.sh
    ```

### Acknowledgement
This code is developed based on [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification).

### MXNet Implementation
https://github.com/bruinxiong/IBN-Net.mxnet

### Citing IBN-Net
```  
@inproceedings{pan2018IBN-Net,  
  author = {Xingang Pan, Ping Luo, Jianping Shi, and Xiaoou Tang},  
  title = {Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net},  
  booktitle = {ECCV},  
  year = {2018}  
}
```  

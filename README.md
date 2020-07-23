## Instance-Batch Normalization Network

### Paper

Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. ["Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"](https://arxiv.org/abs/1807.09441), ECCV2018.  

### Introduction
<img align="middle" width="500" height="280" src="./utils/IBNNet.png">   

- IBN-Net is a CNN model with domain/appearance invariance. It carefully unifies instance normalization and batch normalization in a single deep network.  
- It provides a simple way to increase both modeling and generalization capacity without adding model complexity.  
- IBN-Net is especially suitable for cross domain or person/vehicle re-identification tasks, see [michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) and [strong baseline for ReID](https://arxiv.org/pdf/1906.08332.pdf) for more details.

### Requirements
- Pytorch 0.4.1 or higher

### Results

Top1/Top5 error on the ImageNet validation set are reported. You may get different results when training your models with different random seed.

| Model                     | origin         |  re-implementation      | IBN-Net     |
| -------------------       | ------------------ | ------------------ | ------------------ |
| DenseNet-121          | 25.0/-             | 24.96/7.85       | 24.47/7.25 [[pre-trained model]](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/densenet121_ibn_a-e4af5cc1.pth)    |
| DenseNet-169          | 23.6/-              | 24.02/7.06      | 23.25/6.51 [[pre-trained model]](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/densenet169_ibn_a-9f32c161.pth)    |
| ResNet-18                | -          | 30.24/10.92     | 29.17/10.24  [[pre-trained model]](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth)   |
| ResNet-34                | -          | 26.70/8.58       | 25.78/8.19  [[pre-trained model]](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth)   |
| ResNet-50                | 24.7/7.8          | 24.27/7.08       | 22.54/6.32  [[pre-trained model]](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth)   |
| ResNet-101             | 23.6/7.1           | 22.48/6.23       | 21.39/5.59  [[pre-trained model]](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth)  |
| ResNeXt-101          | 21.2/5.6            | 21.31/5.74       | 20.88/5.42  [[pre-trained model]](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth)  |
| SE-ResNet-101       | 22.38/6.07        | 21.68/5.88       | 21.25/5.51   [[pre-trained model]](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth)  |

The rank1/mAP on two Re-ID benchmarks Market1501 and DukeMTMC-reID (from [michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)):

| Backbone | Market1501 | DukeMTMC-reID |
| --- | -- | -- |
| ResNet50 | 94.5 (85.9) | 86.4 (76.4) |
| ResNet101 | 94.5 (87.1) |  87.6 (77.6) |
| SeResNet50 | 94.4 (86.3) | 86.4 (76.5) |
| SeResNet101 | 94.6 (87.3) | 87.5 (78.0) |
| SeResNeXt50 | 94.9 (87.6) | 88.0 (78.3) |
| SeResNeXt101 | 95.0 (88.0) | 88.4 (79.0) |
| IBN-Net-a | 95.0 (88.2) | 90.1 (79.1) |

### Load IBN-Net from torch.hub
```python
import torch
model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
```

### Testing/Training on ImageNet
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

## Instance-Batch Normalization Network

### Paper

Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. ["Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"](https://arxiv.org/abs/1807.09441), ECCV2018.  

### Introduction
<img align="middle" width="500" height="280" src="./utils/IBNNet.png">   

- IBN-Net carefully unifies instance normalization and batch normalization in a single deep network.  
- It provides an extremely simple way to increase both modeling and generalization capacity without adding model complexity.
- IBN-Net works surprisingly well for Person Re-identification task, see [michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) and [strong baseline for ReID](https://arxiv.org/pdf/1906.08332.pdf) for more details.

### Requirements
- Pytorch 0.3.1 (master branch) or Pytorch 0.4.1 (0.4.1 branch)

### Results

Top1/Top5 error on the ImageNet validation set are reported. You may get different results when training your models with different random seed.

| Model                     | origin         |  re-implementation      | IBN-Net     |
| -------------------       | ------------------ | ------------------ | ------------------ |
| DenseNet-121          | 25.0/-             | 24.96/7.85       | 24.47/7.25              |
| DenseNet-169          | 23.6/-              | 24.02/7.06      | 23.25/6.51              |
| ResNet-50                | 24.7/7.8          | 24.27/7.08       | 22.54/6.32              |
| ResNet-101             | 23.6/7.1           | 22.48/6.23       | 21.39/5.59              |
| ResNeXt-101          | 21.2/5.6            | 21.31/5.74       | 20.88/5.42              |
| SE-ResNet-101       | 22.38/6.07        | 21.68/5.88       | 21.25/5.51              |

### Before Start
1. Clone the repository  
    ```Shell
    git clone https://github.com/XingangPan/IBN-Net.git
    ```

2. Download [ImageNet](http://image-net.org/download-images) dataset (if you need to test or train on ImageNet). You may follow the instruction at [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) to process the validation set.

### Testing
1. Download our pre-trained models and save them to `./pretrained`.   
    Download link: [Pretrained models for pytorch0.3.1](https://drive.google.com/open?id=1JxSo6unmvwkCavEqh42NDKYUG29HoLE0), [Pretrained models for pytorch0.4.1](https://drive.google.com/open?id=1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S)
2. Edit `test.sh`. Modify `model` and `data_path` to yours.  
    Options for `model`: densenet121_ibn_a, densenet169_ibn_a, resnet50_ibn_a_old, resnet50_ibn_a, resnet50_ibn_b, resnet101_ibn_a_old, resnet101_ibn_a, resnext101_ibn_a, se_resnet101_ibn_a.  
    (Note: For IBN-Net version of ResNet-50 and ResNet-101, our results in the paper are reported based on an slower implementation, corresponding to resnet50_ibn_a_old and resnet101_ibn_a_old here. We also provide a faster implementation, and the models are resnet50_ibn_a, resnet101_ibn_a, and all the rest. The top1/top5 error for resnet50_ibn_a and resnet101_ibn_a are 22.76/6.41 and 21.29/5.61 respectively.)  
3. Run test script
    ```Shell
    sh test.sh
    ```
 
### Training
1. Edit `train.sh`. Modify `model` and `data_path` to yours.  
2. Run train script
    ```Shell
    sh train.sh
    ```

This code is modified from [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification).

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

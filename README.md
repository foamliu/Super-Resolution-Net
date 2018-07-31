# 超分辨率网络

![apm](https://img.shields.io/apm/l/vim-mode.svg)

EDSR 的 Keras 实现。

## 原理

请参照论文 [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf)。

本代码参照了原作者的 Torch 实现：[NTIRE2017](https://github.com/LimBee/NTIRE2017) 和 jmiller656 的 Tensorflow 实现 [EDSR-Tensorflow](https://github.com/jmiller656/EDSR-Tensorflow).

EDSR (单尺度模型。 我们提供尺寸x4的模型):

![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/EDSR.png)


## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/imagenet.png)

按照 [说明](https://github.com/foamliu/ImageNet-Downloader) 下载 ImageNet 数据集。


## 如何使用


### 训练
```bash
$ python train.py
```

如果想可视化训练效果，请运行:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/learning_curve.png)

### 演示
下载 [预训练模型](https://github.com/foamliu/Super-Resolution-Net/releases/download/untagged-0b1ce773ce0ef13ac79a/model.16-9.0500.hdf5) 放入 "models" 目录然后执行:

```bash
$ python demo.py -s [2, 3, 4]
```

#### 尺度 scale=2

输入 | 输出 | PSNR | 目标 |
|---|---|---|---|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_out_x2.png)| 31.16740 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_out_x2.png)| 44.90785 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_out_x2.png)| 35.79408 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_out_x2.png)| 35.20883 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_out_x2.png)| 45.94723 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_out_x2.png)| 36.19514 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_out_x2.png)| 34.62096 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_out_x2.png)| 37.08404 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_out_x2.png)| 49.95246 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_gt_x2.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_image_x2.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_out_x2.png)| 35.17739 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_gt_x2.png)|

#### 尺度 scale=3

输入 | 输出 | PSNR | 目标 |
|---|---|---|---|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_out_x3.png)| 34.31166 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_out_x3.png)| 46.45506 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_out_x3.png)| 32.02418 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_out_x3.png)| 35.02696 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_out_x3.png)| 33.58175 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_out_x3.png)| 29.45888 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_out_x3.png)| 33.97808 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_out_x3.png)| 33.07943 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_out_x3.png)| 36.53109 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_gt_x3.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_image_x3.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_out_x3.png)| 30.85645 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_gt_x3.png)|

#### 尺度 scale=4

输入 | 输出 | PSNR | 目标 |
|---|---|---|---|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_out_x4.png)| 36.14261 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_out_x4.png)| 34.55068 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_out_x4.png)| 28.74133 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_out_x4.png)| 30.53005 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_out_x4.png)| 30.85291 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_out_x4.png)| 32.00084 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_out_x4.png)| 31.75487 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_out_x4.png)| 32.94882 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_out_x4.png)| 30.10912 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_gt_x4.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_image_x4.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_out_x4.png)| 32.22449 | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_gt_x4.png)|

### 模型评估
在 4268 张验证集图片上测得 PSNR 并求均值：x2=36.50111 dB, x3=33.88247 dB, x4=32.76783 dB。

```bash
$ python evaluate.py -s [2, 3, 4]
```

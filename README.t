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
$ python demo.py
```

输入 | 输出 | PSNR | 目标 |
|---|---|---|---|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_out.png)| $(psnr_0) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/0_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_out.png)| $(psnr_1) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/1_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_out.png)| $(psnr_2) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/2_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_out.png)| $(psnr_3) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/3_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_out.png)| $(psnr_4) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/4_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_out.png)| $(psnr_5) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/5_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_out.png)| $(psnr_6) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/6_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_out.png)| $(psnr_7) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/7_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_out.png)| $(psnr_8) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/8_gt.png)|
|![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_image.png) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_out.png)| $(psnr_9) | ![image](https://github.com/foamliu/Super-Resolution-Net/raw/master/images/9_gt.png)|

### 模型评估
在 4268 张验证集图片上测得 PSNR 并求均值。得到：$(avg_psnr) dB。

```bash
$ python evaluate.py
```

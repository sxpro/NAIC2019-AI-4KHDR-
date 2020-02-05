# 一、解题思路说明

## 待解决的问题

视频四倍超分+画质增强，要求将夹杂过曝/欠曝内容、未调色的540p SDR视频重建为高质量、已调色的4K SDR视频；
## 整体方案

从单帧图像的角度入手，提取视频的**关键帧**进行深度模型训练，通过**调色**和**超分**两阶段对低质量视频进行恢复。

### 数据处理

- 利用opencv对比赛提供的低质量`.mp4`视频文件进行关键帧抽帧处理，得到的结果保存成`.png`格式的图片，这些图片作为训练所使用的`input` ；再抽取这些图片所对应高质量视频中的关键帧当做`label`。

### 调色阶段

- 基于轻量级RCAN，先训练调色模型，此处将原始视频抽帧图像下采样4倍当做`label`。
- 此过程使用L1 Loss进行色彩恢复，下式可表示此过程：

$$
I_{color}^{c \times w \times h} = Ne{t_{color}}(I_{input}^{c \times w \times h})
$$

### 超分阶段

- 基于Adaptive RCAN，采用单帧图像超分处理方案，将一阶段调色后的结果当做输入，其对应的原始视频抽帧图像当做输出。
- 此过程使用L1 Loss + SSIM Loss进行图像超分辨率重建，下式表示过程：

$$
I_{final}^{c \times 2w \times 2h} = Ne{t_{sr}}(I_{color}^{c \times w \times h})
$$

## 主要创新点

* 调色模型

* 多损失函数 L1+SSIM

* 采用关键帧训练，减少数据量

# 二、数据和模型使用
## 预训练模型的使用情况
无
## 相关论文及预训练模型的下载链接
无

# 三、项目运行环境
## 项目
项目代码基于RBPN https://github.com/alterzero/RBPN-PyTorch


## 项目所需的工具包/框架

* python == 3.6.9
* numpy
* pytorch == 1.3.1
* scipy
* opencv
* pytorch_msssim
* tqdm
* tensorboardX
* ffmpeg 


## 项目运行的资源环境

* 1080Ti*1

# 四、项目运行办法


## 项目的文件结构
```
model
|  main.py                     #主模型训练文件
|  test.py                     #主模型测试文件0， 运行在卡0
|  test1.py                    #主模型测试文件1， 运行在卡1
|  test2.py                    #主模型测试文件2， 运行在卡2
|  test3.py                    #主模型测试文件3， 运行在卡3
|  data.py                     #主模型data文件
|  dataset.py                  #主模型dataset文件
|  loss.py                     #主模型损失函数文件
|  rcan.py                     #模型文件
|  mp4_png.py                  #视频图像互转参考文件
|  mp42png.py                  #视频转图像文件
|  png2mp4.py                  #图像转视频文件
|  creat_flist.py              #创建训练集test_data与测试集lr_images,hr_images的数据列表
|  create_color_flist.py       #创建调色后的训练集train_color与测试集test_color的数据列表
|  delete_error_frame.sh       #删除问题帧
|  readme.md	          
└───color                   #调色模型文件夹
	|   colornet.py              #模型文件
	|   data.py                  #调色模型data
	|   dataset.py               #调色模型dataset
	|   loss.py                  #损失函数文件
	|   main_color.py            #训练与测试的调色模型文件
└───Results                    #单帧图像结果存放（自动创建）
	└───16536366
		|  001.png
└───weights                    #主模型权重存放（自动创建）
  │   xx.pth
└───color_weights              #调色模型权重存放（自动创建）
  │   xx.pth
```

## 项目的运行步骤

已将所有路径设置为**相对路径**，如有路径问题，请将所有路径按照复现本地环境，设置为**绝对路径**，避免出错，谢谢！
`mp4_png.py`为参考文件，当`mp42png.py`与`png2mp4.py`的多线程出问题时，可作参考，采用循环

-----------


1. 运行**mp42png.py**, 生成关键帧数据。文件中有训练集的540p、4K数据与测试集的数据路径
```
python mp42png.py
```
- 删除一些问题帧
```
sh delete_error_frame.sh
```

---------------

2. 运行**create_flist.py**, 生成调色模型的训练和测试数据列表

```
python create_flist.py
```
- 生成`hr`数据列表：`train_hr.flist`, `lr`数据列表：`train_lr.flist`,  `test`数据列表：`test.flist`

-------------

3. 运行 **color/main_color.py**, 训练调色模型
```
python color/main_color.py --gpus 4 --batchSize （请设置较大的数值， 若导致显卡利用率降低可减少）
```
---------

4. 用调色模型对训练集`lr_images`和测试集`test_data`进行调色
```
python color/main_color.py --test_only True --pretrained True --testBatchSize （请设置较大的数值，若导致显卡利用率降低可减少）--gpus 4 
```
- 生成结果保存在`train_color/`下
- 请将代码中`test_flist`设置为 步骤**2**中得到的`test.flist`位置，`output`设置为训练集调色输出，如: `test_color`; `pretrained_sr`设置为 步骤3 中得到的最佳模型参数位置，如`model/color_weights/color_ColorNet_video_best.pth`

```
python color/main_color.py --test_only True --pretrained True --testBatchSize （请设置较大的数值， 若导致显卡利用率降低可减少）--gpus 4
```
- 生成结果保存在`test_color/`下

------------

5. 运行 **create_color_flist.py**, 生成用于超分模型训练的`lr`数据: `trian_color.flist`，测试的test数据：
```
python create_color_flist.py  
```

------------

6. 运行 **main.py**, 训练超分模型
```
python main.py --batchSize （请设置较大的数值， 若导致显卡利用率降低可减少）--gpus 4
```
- 权重保存在`model/weights/`下， 其中最佳模型名称应为 `4x_RCAN_image_best.pth`

--------

7. 运行 **test.py**, 测试程序
```
CUDA_VISIBLE_DEVICES=0 python test.py --self_ensemble True --testBatchSize （请设置较大的数值， 若速度减慢可调小）
CUDA_VISIBLE_DEVICES=1 python test1.py --self_ensemble True --testBatchSize （请设置较大的数值， 若速度减慢可调小）
CUDA_VISIBLE_DEVICES=2 python test2.py --self_ensemble True --testBatchSize （请设置较大的数值， 若速度减慢可调小）
CUDA_VISIBLE_DEVICES=3 python test3.py --self_ensemble True --testBatchSize （请设置较大的数值， 若速度减慢可调小）
```
- 生成的结果保存在`model/Results`文件夹下

---------

8. 运行 **png2mp4.py**, 生成最终结果mp4文件

```
python png2mp4.py
```

## 运行结果的位置

1. `model/Results/XX/xx.png` 图像文件
2. `answer/xx.mp4` 视频文件

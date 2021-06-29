---
layout:     post
title:      "『森林火灾检测』基于PaddleX实现森林火灾检测"
subtitle:   "使用Paddle作为开发套件，实现森林火灾检测"
date:       2021-06-27 12:00:00
author:     "thomas-yanxin"
header-img: "assets/owner/blog/header/post-bg-01.jpg"
thumbnail: /assets/owner/blog/thumbs/thumb01.png
tags: [technology]
category: [technology]
comments: true
share: true
---

# 效果预览
[B站链接](https://www.bilibili.com/video/BV1F44y1k7Cy/)
[AIStudio链接](https://aistudio.baidu.com/aistudio/projectdetail/1968964)


# 项目背景
<font face="楷体" size=3>&emsp;&emsp;2019年3月30日17时 ，凉山州木里县境内发生森林火灾，30名扑火人员牺牲。

&emsp;&emsp;2020年3月30日15时35分，凉山州西昌市经久乡和安哈镇交界的皮家山山脊处发生森林火灾，参与火灾扑救的19人牺牲、3人受伤。这起森林火灾造成各类土地过火总面积3047.7805公顷，综合计算受害森林面积791.6公顷，直接经济损失9731.12万元。

&emsp;&emsp;2020年4月14日17时35分，西藏自治区林芝市巴宜区尼西村附近发生森林火灾，3000余人历时4昼夜持续扑救，明火于4月18日17时全部扑灭。此次扑救，共扑打火线13公里，清理烟点4200余处、站杆倒木3900余根，开设防火隔离带3.5公里，实施了两次人工增雨。

&emsp;&emsp;2021年3月13日12时许，宁夏固原市原州区张易镇马场村二林沟荒山起火，扑火过程中造成2人死亡，6人受伤。火场共有3处着火点，在3个不同山头，过火面积500余亩。

&emsp;&emsp;2021年3月14日13时许，云南省昆明市盘龙区茨坝街道与龙泉街道交界处三丘田附近发生森林火灾。火情发生后，昆明市森林消防支队249名指战员迅速赶往现场处置，地方扑救力量（地方专业扑火队员256人、半专业扑火队员160人、干部群众155人）紧密配合扑救。

&emsp;&emsp;1950年以来，中国年均发生森林火灾13067起，受害林地面积653019公顷，因灾伤亡580人。其中1988年以前，全国年均发生森林火灾15932起，受害林地面积947238公顷，因灾伤亡788人（其中受伤678人，死亡110人）。1988年以后，全国年均发生森林火灾7623起，受害林地面积94002公顷，因灾伤亡196人（其中受伤142人，死亡54人），分别下降52.2%、90.1%和75.3%。

<font face="楷体" size=3>&emsp;&emsp;森林火灾由于在野外，人烟稀少，故而难以在初始阶段就发现并及时扑灭，更大程度上只有当火情演变为较大规模后，才能够被发现并扑救，而这时，扑救难度及损失程度已呈指数级上升。而如何在野外这样的恶劣环境下及时捕捉火情并通报是人工智能时代下防火灭火的新命题。  
&emsp;&emsp;本项目基于PaddleX,选取PPYOLO进行项目开发，并实现了windows端的部署，后期将结合PaddleSlim裁剪模型大小以及PaddleLite部署于树莓派上。

# 实地操作
<font face="楷体" size=3>1. 数据处理
2. 模型训练
3. 模型导出


```python
# 首先安装两个依赖包
!pip install paddlex
!pip install paddle2onnx
```

### 数据处理


```python
#解压数据集并将其移动至dataset中
!tar -xf /home/aistudio/data/data90352/fire_detection.tar 
```


```python
!mv VOC2020 dataset 
```

### 数据处理
<font face="楷体" size=3>&emsp;&emsp;在本数据集中，由于文件名及文件内容不符合PaddleX所提供的数据集读取API，故需要对其进行处理。观察数据集可知，有两个问题：一为标注文件的文件名中存在空格，这极大地影响了PaddleX数据集读取；二为标注文件中的内容需要进行对应性修改。


```python
# 修改.xml文件名，去掉文件名中的空格
# -*- coding: utf-8 -*-
import os
#设定文件路径
jpg_path='dataset/JPEGImages/'
anno_path = 'dataset/Annotations/'
i=1
#对目录下的文件进行遍历
for file in os.listdir(jpg_path):
#判断是否是文件
    if os.path.isfile(os.path.join(jpg_path,file))==True:
#设置新文件名
        main = file.split('.')[0]
        if " " in main:
            new_main = main.replace(' ','')
            new_main_jpg = new_main + '.jpg'
            new_main_anno = new_main + '.xml'
            print(os.path.join(jpg_path,new_main_jpg))
            print(os.path.join(anno_path,new_main_anno))

#         new_name=file.replace(file,"rgb_%d.jpg"%i)
# #重命名
            os.rename(os.path.join(jpg_path,main+'.jpg'),os.path.join(jpg_path,new_main_jpg))
            os.rename(os.path.join(anno_path,main+'.xml'),os.path.join(anno_path,new_main_anno))
            i+=1
#结束
print ("End")
```


```python
# 这里修改.xml文件中的<path>元素
!mkdir dataset/Annotations1
import xml.dom.minidom
import os

path = r'dataset/Annotations'  # xml文件存放路径
sv_path = r'dataset/Annotations1'  # 修改后的xml文件存放路径
files = os.listdir(path)
cnt = 1

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    item = root.getElementsByTagName('path')  # 获取path这一node名字及相关属性值
    for i in item:
        i.firstChild.data = '/home/aistudio/dataset/JPEGImages/' + str(cnt).zfill(6) + '.jpg'  # xml文件对应的图片路径

    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
    cnt += 1
```


```python
# 这里修改.xml文件中的<failname>元素
!mkdir dataset/Annotations2
import xml.dom.minidom
import os

path = r'dataset/Annotations1'  # xml文件存放路径
sv_path = r'dataset/Annotations2'  # 修改后的xml文件存放路径
files = os.listdir(path)

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    names = root.getElementsByTagName('filename')
    a, b = os.path.splitext(xmlFile)  # 分离出文件名a
    for n in names:
        n.firstChild.data = a + '.jpg'
    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
```

<font face="楷体" size=3>下面删除在数据集处理过程中所生成的冗余文件，并将其更改为适合PaddleX的数据集格式。


```python
!rm -rf dataset/Annotations
!rm -rf dataset/Annotations1
!mv dataset/Annotations2 dataset/Annotations
```

<font face="楷体" size=3>&emsp;&emsp;PaddleX非常**贴心**地为开发者准备了数据集划分工具，免去了开发者多写几行代码的需求。这里我们设置训练集、验证集、测试集划分比例为7：2：1。


```python
!paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/dataset/ --val_value 0.2 --test_value 0.1
```

### 模型训练

<font face="楷体" size=3>&emsp;&emsp;在使用PaddleX进行模型训练的过程中，我们使用目前PaddleX适配精度最高的PPYolo模型进行训练。其模型较大，预测速度比YOLOv3-DarkNet53更快，适用于服务端。大家也可以更改其他模型尝试一下。这里我训练了大概200个epoch(别问，问就是没算力了也懒得续点了……)当然看趋势还能涨！（有算力的童鞋可以试着调参或者继续往下面去试试）


```python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx


# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250), transforms.RandomDistort(),
    transforms.RandomExpand(), transforms.RandomCrop(), transforms.Resize(
        target_size=608, interp='RANDOM'), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'), transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/train_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    transforms=train_transforms,
    parallel_method='thread',
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/val_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    parallel_method='thread',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo
model = pdx.det.PPYOLO(num_classes=num_classes)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=540,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    save_interval_epochs=1,
    lr_decay_epochs=[270,320, 480],
    save_dir='output/ppyolo',
    resume_checkpoint='/home/aistudio/output/ppyolo/epoch_197',
    use_vdl=True)
```

### 模型导出
<font face="楷体" size=3>&emsp;&emsp;这里我们将训练过程中保存的模型导出为inference格式模型，其原因在于：PaddlePaddle框架保存的权重文件分为两种：支持前向推理和反向梯度的训练模型 和 只支持前向推理的推理模型。二者的区别是推理模型针对推理速度和显存做了优化，裁剪了一些只在训练过程中才需要的tensor，降低显存占用，并进行了一些类似层融合，kernel选择的速度优化。而导出的inference格式模型包括__model__、__params__和model.yml三个文件，分别表示模型的网络结构、模型权重和模型的配置文件（包括数据预处理参数等）。


```python
!paddlex --export_inference --model_dir=output/ppyolo/best_model --save_dir=./inference_model
```

### 模型部署
<font face="楷体" size=3>模型导出后我们可以采用Python端以视频流的形式部署。


```python
import cv2
import paddlex as pdx
from playsound import playsound

# 修改模型所在位置
predictor = pdx.deploy.Predictor('D:\\project\\python\\fire\\inference_model')
cap = cv2.VideoCapture('D:\\project\\python\\fire\\mda-kcwgejj7mfckc19e.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        result = predictor.predict(frame)
        score = result[0]['score']
        if score >= 0.3:
            print("*"*100)
            # 修改音频所在位置
            # playsound('D:\\project\\python\\cigarette\\cigarette.mp3')
        # print(result)
        vis_img = pdx.det.visualize(frame, result, threshold=0.3, save_dir=None)
        cv2.imshow('cigarette', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
```




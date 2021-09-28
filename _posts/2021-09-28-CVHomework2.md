---
title: 'Apple M1配置python环境，进行人脸检测'
date: 2021-09-04
permalink: /posts/2021/0928
tags:
  - DeepLearning
  - ComputerVision
---
在Apple M1的环境下安装miniforge3配置python环境，安装opencv等库，进行人脸检测。这也是计算机视觉课程的作业。
# 一、实验环境

- 操作系统：macOS Big Sur 11.6
- CPU：Apple M1

- Python版本：3.8.12
- pycharm：2021.2.2

- Anaconda：miniforge(由于ARM架构兼容问题，最终选择安装了miniforge)

<img src="/images/Blog21-09-28Image/w2_1.png" alt="image-20210928174909014" style="zoom: 33%;" />

#  二、环境配置

##  1.miniforge3的安装

1.由于Arm架构的问题，anaconda没有更好地兼容，所以我们只能下载miniforge3代替。

[下载链接](https://github.com/conda-forge/miniforge/releases)：https://github.com/conda-forge/miniforge/releases

[国内版本](https://gitee.com/photographer_adam/miniforge)：https://gitee.com/photographer_adam/miniforge

<img src="/images/Blog21-09-28Image/w2_2.png" alt="image-20210928175834003" style="zoom: 25%;" />

下载完成之后，进入对应的安装目录，打开终端：

`1.进入下载目录`

`cd ~/Downloads`

`2.安装`

`sh Miniforge3-MacOSX-arm64.sh`

`3.验证是否安装成功`

`conda --version`

安装结果如下：

<img src="/images/Blog21-09-28Image/w2_3.png" alt="image-20210928180729756" style="zoom: 25%;" />

## 2.创建虚拟python环境,导入对于的包

命令行执行命令如下：

`#创建名字为tf38-ai的环境，python版本为3.8`

`conda conda create -n tf38-ai python==3.8`

`#查看安装的环境`

`conda env list` 

`#激活对应环境`

`conda activate tf38-ai`

结果如下：

<img src="/images/Blog21-09-28Image/w2_4.png" alt="image-20210928181615699" style="zoom: 50%;" />

 环境的删除操作,以环境py39为例：

<img src="/images/Blog21-09-28Image/w2_13.png" alt="image-20210928181615699" style="zoom: 50%;" />

<img src="/images/Blog21-09-28Image/w2_14.png" alt="image-20210928181615699" style="zoom: 50%;" />

## 3.安装numpy,matplotlib,opencv

命令行执行如下操作,可成功安装matplotlib,numpy：

`conda install numpy matplotlib`

对于opencv,选择通过pycharm加载安装包的方式，以次点击>Preferences | Project: Desktop | Python Interpreter具体如下图：

<img src="/images/Blog21-09-28Image/w2_5.png" alt="image-20210928182753109" style="zoom: 25%;" />

搜索opencv进行安装即可：

<img src="/images/Blog21-09-28Image/w2_6.png" alt="image-20210928183014453" style="zoom: 25%;" />

可通过如下命令查看是否安装成功：

`conda list`

结果如下，全部安装成功。

<img src="/images/Blog21-09-28Image/w2_7.png" alt="image-20210928183137428" style="zoom: 25%;" />

<img src="/images/Blog21-09-28Image/w2_8.png" alt="image-20210928183220410" style="zoom: 25%;" />

#  三、人脸检测实验

## 1.下载对应的分类文件

[下载链接](https://github.com/opencv/opencv/tree/master/data/haarcascades)：https://github.com/opencv/opencv/tree/master/data/haarcascades

本工程用到的人脸检测文件[haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)和人脸眼睛检测文件[haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)。

<img src="/images/Blog21-09-28Image/w2_9.png" alt="image-20210928184004998" style="zoom: 25%;" />

## 2.基本流程

以静态图片检测人脸为例：

```python
#1.读取图像
faceImg = cv2.imread(image_name)

#2.转化为灰度图
gray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)

#3.加载人脸识别分类器
classifier1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier2 = cv2.CascadeClassifier('haarcascade_eye.xml')

#4.矩阵线条颜色
color1 = (255, 0, 0)
color2 = (0, 255, 0)

#5.进行识别，faceRects=人脸所在的坐标
faceRects1 = classifier1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32))
faceRects2 = classifier2.detectMultiScale(gray, scaleFactor=2.6, minNeighbors=3, minSize=(32, 32))

#6.框选出人脸
if len(faceRects1):
    for faceRect in faceRects1:
        x, y, w, h = faceRect
        # 框选出人脸   最后一个参数2是框线宽度
        cv2.rectangle(faceImg, (x, y), (x + h, y + w), color1, 2)
        cv2.putText(faceImg, "Face", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5,  color1 , 2, cv2.LINE_AA)

if len(faceRects2):
    for faceRect in faceRects2:
        x, y, w, h = faceRect
        # 框选出人脸   最后一个参数2是框线宽度
        cv2.rectangle(faceImg, (x, y), (x + h, y + w), color2, 2)
        cv2.putText(faceImg, "Eye", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color2, 2, cv2.LINE_AA)
#7.展示图片
cv2.imshow("faceImg",faceImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

##  3.实现效果

本项目分别实现了调用电脑摄像头进行**视频动态检测**以及**上传图片静态检测**两种方式。**针对人脸以及眼睛两个目标都做了检测**。

其中，GUI界面的实现调用了python中的tkinter。

运行代码，初始界面如下：

<img src="/images/Blog21-09-28Image/w2_10.png" alt="image-20210928184857542" style="zoom: 50%;" />

点击动态视频检测，结果如下：

<img src="/images/Blog21-09-28Image/image/w2_11.png" alt="iShot2021-09-25 17.05.45" style="zoom: 25%;" />

点击静态图片检测，结果如下：

<img src="/images/Blog21-09-28Image/w2_12.png" alt="image-20210928185909299" style="zoom: 50%;" />

## 4.完整代码

```python
import cv2
import tkinter as tk
from tkinter import filedialog


# 视频人脸检测
def VideoCapture():
    cap = cv2.VideoCapture(0)  # 开启摄像头

    # 循环读取图像
    while True:
        ok, faceImg= cap.read()  # 读取摄像头图像
        if ok is False:
            print('无法读取到摄像头！')
            break;
        gray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        color = (0, 255, 0)

        # 识别器进行识别
        faceRects = classifier.detectMultiScale(gray, scaleFactor=2.5, minNeighbors=3, minSize=(32, 32))

        if len(faceRects):
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 框选出人脸   最后一个参数2是框线宽度
                cv2.rectangle(faceImg, (x, y), (x + h, y + w), color, 2)
                cv2.putText(faceImg, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)

        # 转换灰色
        gray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)

        # 加载人眼识别分类器
        classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
        color = (255, 0, 0)

        # 识别器进行识别
        faceRects = classifier.detectMultiScale(gray, scaleFactor=2.6, minNeighbors=3, minSize=(32, 32))

        if len(faceRects):
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 框选出人脸   最后一个参数2是框线宽度
                cv2.rectangle(faceImg, (x, y), (x + h, y + w), color, 2)
                cv2.putText(faceImg, "Eye", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)

        # faceImg= cv2.cvtColor(faceImg,cv2.COLOR_RGB2BGR)
        cv2.imshow("faceImg", faceImg)
        # 展示图像

        k = cv2.waitKey(10)  # 键盘值
        if k == 27:  # 通过esc键退出摄像
            break

    # 关闭摄像头
    cap.release()
    cv2.destroyAllWindows()

def ImageDetect(image_name):
    #1.读取图像
    faceImg = cv2.imread(image_name)

    #2.转化为灰度图
    gray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)

    #3.加载人脸识别分类器
    classifier1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier2 = cv2.CascadeClassifier('haarcascade_eye.xml')

    #4.矩阵线条颜色
    color1 = (255, 0, 0)
    color2 = (0, 255, 0)

    #5.进行识别，faceRects=人脸所在的坐标
    faceRects1 = classifier1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32))
    faceRects2 = classifier2.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3, minSize=(32, 32))

    #6.框选出人脸
    if len(faceRects1):
        for faceRect in faceRects1:
            x, y, w, h = faceRect
            # 框选出人脸   最后一个参数2是框线宽度
            cv2.rectangle(faceImg, (x, y), (x + h, y + w), color1, 2)
            cv2.putText(faceImg, "Face", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5,  color1 , 2, cv2.LINE_AA)

    if len(faceRects2):
        for faceRect in faceRects2:
            x, y, w, h = faceRect
            # 框选出人眼   最后一个参数2是框线宽度
            cv2.rectangle(faceImg, (x, y), (x + h, y + w), color2, 2)
            cv2.putText(faceImg, "Eye", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color2, 2, cv2.LINE_AA)
    #7.展示图片
    cv2.imshow("faceImg",faceImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def chooseImage():
    # 打开选择文件夹对话框
    root = tk.Tk()
    root.withdraw()
    # 选择文件
    Filepath = filedialog.askopenfilename()
    ImageDetect(Filepath)


base = tk.Tk()

base.wm_title("FaceDetect")#负责标题
base.geometry('400x200')
lb = tk.Label(base,text="Choose the style you want")
# 方法-直接调用 run1()
btn1 = tk.Button(base, text='静态图片检测', command=chooseImage)
btn1.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

# 方法二利用 lambda 传参数调用run2()
btn2 = tk.Button(base, text='动态视频检测', command=VideoCapture)
btn2.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)
lb.pack()#给相应的组件指定布局

base.mainloop()
```
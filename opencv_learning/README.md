# 🎯 Python + OpenCV 计算机视觉学习项目

## 项目简介

这是一个**从零开始学习 Python + OpenCV** 的完整教学项目。项目的每个文件和每一行代码都附有**详细的中文注释**，帮助你理解计算机视觉的核心概念和 OpenCV 的实战用法。

## 项目架构

```
opencv_learning/
├── main.py                      # 🚀 主入口：按编号运行各个学习模块
├── requirements.txt             # 📦 依赖清单
├── src/                         # 📂 源代码目录
│   ├── basics/                  # 🧱 基础篇：图像读取、显示、基本操作
│   │   ├── io_display.py        #   图像的读取、显示、保存
│   │   ├── basic_ops.py         #   裁剪、缩放、旋转、翻转
│   │   └── color_spaces.py      #   颜色空间转换（RGB、HSV、灰度等）
│   ├── processing/              # 🔧 处理篇：图像滤波与变换
│   │   ├── filters.py           #   模糊、锐化、去噪滤波器
│   │   ├── edge_detection.py    #   Canny、Sobel、Laplacian 边缘检测
│   │   ├── morphological.py     #   膨胀、腐蚀、开闭运算
│   │   └── thresholding.py      #   二值化、自适应阈值、Otsu
│   ├── analysis/                # 🔍 分析篇：图像特征提取与分析
│   │   ├── contours.py          #   轮廓查找、绘制、形状识别
│   │   ├── histograms.py        #   直方图计算、均衡化、反向投影
│   │   └── feature_detection.py #   角点检测、SIFT、ORB 特征匹配
│   └── applications/            # 🏆 应用篇：综合实战项目
│       ├── face_detection.py    #   人脸检测（Haar Cascade + DNN）
│       ├── object_tracking.py   #   目标跟踪（光流法、Meanshift）
│       └── image_stitching.py   #   图像拼接（全景图合成）
├── utils/
│   └── helpers.py               # 🛠 工具函数：绘图辅助、计时等
├── data/                        # 📁 测试图片存放目录
└── outputs/                     # 📁 处理结果输出目录
```

## 学习路线

建议按以下顺序学习，每个模块都可以**独立运行**：

| 序号 | 模块 | 文件 | 学习目标 |
|------|------|------|----------|
| 1 | 图像读写 | `src/basics/io_display.py` | 了解 OpenCV 如何读取和显示图像 |
| 2 | 基本操作 | `src/basics/basic_ops.py` | 掌握裁剪、缩放、旋转等基本变换 |
| 3 | 颜色空间 | `src/basics/color_spaces.py` | 理解 RGB/HSV/灰度等颜色模型 |
| 4 | 图像滤波 | `src/processing/filters.py` | 学习模糊、锐化、降噪等滤波技术 |
| 5 | 边缘检测 | `src/processing/edge_detection.py` | 掌握 Canny/Sobel 等边缘提取算法 |
| 6 | 形态学 | `src/processing/morphological.py` | 理解膨胀/腐蚀/开闭运算 |
| 7 | 阈值化 | `src/processing/thresholding.py` | 掌握各种二值化方法 |
| 8 | 轮廓分析 | `src/analysis/contours.py` | 学习轮廓查找与形状分析 |
| 9 | 直方图 | `src/analysis/histograms.py` | 理解图像直方图与均衡化 |
| 10 | 特征检测 | `src/analysis/feature_detection.py` | 掌握角点检测与特征匹配 |
| 11 | 人脸检测 | `src/applications/face_detection.py` | 实战人脸检测 |
| 12 | 目标跟踪 | `src/applications/object_tracking.py` | 实战目标跟踪 |
| 13 | 图像拼接 | `src/applications/image_stitching.py` | 实战全景图合成 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备测试图片：在 data/ 目录下放一张名为 sample.jpg 的图片
#    或者代码会自动生成测试图案

# 3. 运行主程序，按菜单选择模块
python main.py

# 4. 也可以单独运行某个模块
python -m src.basics.io_display
```

## 核心概念速查

| 概念 | 一句话解释 |
|------|-----------|
| **图像在 OpenCV 中是什么？** | 一个 NumPy 三维数组 `[高度, 宽度, 通道数]`，每个像素值是 0-255 的整数 |
| **通道 (Channel)** | 彩色图有 BGR 三个通道（注意不是 RGB！），灰度图只有一个通道 |
| **卷积 (Convolution)** | 用一个"小窗口"（核）在图像上滑动，逐像素计算加权和——所有滤波的基础 |
| **阈值化 (Thresholding)** | 把灰度图变成黑白图：大于阈值的变白，小于的变黑 |
| **形态学 (Morphology)** | 基于形状的图像处理——膨胀让白色区域变大，腐蚀让白色区域变小 |
| **特征 (Feature)** | 图像中"有意义的点"——角点、边缘、斑块等，用于匹配和识别 |
| **轮廓 (Contour)** | 图像中相连的边界点组成的曲线，用于形状分析 |

## 学习建议

1. **逐模块运行**：按序号依次运行，理解每个概念后再进入下一个
2. **修改参数实验**：每个函数都有参数，试着改改看效果变化
3. **阅读注释**：代码中每一行都有中文注释，先读注释再看代码
4. **自己重写**：理解后尝试不看原代码自己重写一遍

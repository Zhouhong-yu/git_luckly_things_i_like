# =============================================================================
# 🔧 处理篇 [1/4] - filters.py
# 图像滤波：模糊、去噪、锐化
# =============================================================================
# 学完本模块你应该理解：
#   1. 什么是卷积（Convolution）——图像处理的核心运算
#   2. 各种模糊滤波器及其用途
#   3. 如何去除图像噪点
#   4. 如何锐化图像让边缘更清晰
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 卷积 (Convolution)
# =============================================================================
# 卷积是图像处理中最基本的运算——几乎所有滤波器都基于卷积。
#
# 什么是卷积？
#   想象你拿着一个"小窗口"（叫做"核" Kernel 或"滤波器" Filter），
#   在图像上从左到右、从上到下滑动。
#   每滑动一步，把小窗口和图像对应位置的像素值相乘再求和，
#   得到的结果就是输出图像的一个像素值。
#
# 举例：一个 3×3 的均值模糊核:
#         [1/9  1/9  1/9]
#    K =  [1/9  1/9  1/9]    每个权重都是 1/9
#         [1/9  1/9  1/9]    中心像素 = 它和周围8个邻居的平均值
#
# 核的大小和值决定了滤波效果:
#   - 全正数 → 模糊/平滑
#   - 有正有负 → 锐化/边缘检测
#   - 核越大 → 效果越强（但也越慢）
# =============================================================================

def add_noise(image: np.ndarray, noise_type: str = "gaussian") -> np.ndarray:
    """
    给图像添加噪声——用于演示去噪效果。

    噪声是图像中不想要的随机信号，来源于：
        - 相机传感器在低光环境下的热噪声
        - 传输过程中的信号干扰
        - 压缩导致的块效应

    参数:
        image:      原始图像
        noise_type: "gaussian" 高斯噪声 或 "salt_pepper" 椒盐噪声

    返回:
        添加噪声后的图像
    """
    noisy = image.copy().astype(np.float32)  # 转为浮点以避免计算溢出

    if noise_type == "gaussian":
        # 高斯噪声: 每个像素加上一个服从正态分布的随机值
        # 均值为0，标准差为25
        # np.random.normal(mean, std, shape) 生成正态分布随机数
        noise = np.random.normal(0, 25, image.shape).astype(np.float32)
        noisy = noisy + noise

    elif noise_type == "salt_pepper":
        # 椒盐噪声: 随机把一些像素变成纯黑(胡椒)或纯白(盐)
        # 就像撒了盐和胡椒粉在图像上
        amount = 0.02  # 噪声比例：2%的像素受影响
        # 生成随机掩模
        salt = np.random.random(image.shape[:2]) < amount / 2    # 1% 的像素变成白点
        pepper = np.random.random(image.shape[:2]) < amount / 2  # 1% 的像素变成黑点
        # 设置盐噪声（白点）
        noisy[salt] = 255
        # 设置胡椒噪声（黑点）
        noisy[pepper] = 0

    # 裁剪到 0-255 范围内（防止溢出）
    return np.clip(noisy, 0, 255).astype(np.uint8)


def demo_blur():
    """
    演示1: 各种模糊滤波器

    模糊（也叫平滑 Smoothing）是最常用的图像处理操作之一。
    主要用途：
        - 去除噪声
        - 减少细节（让算法更关注整体结构）
        - 美化皮肤（磨皮效果）
    """
    print("\n--- 📖 演示1: 图像模糊（平滑）---")
    img = load_image("sample.jpg")

    # --- 1. 均值滤波 (Averaging / Box Filter) ---
    # 把每个像素替换成它周围 k×k 区域的平均值
    # 简单粗暴，但会让图像变得模糊
    # cv2.blur(图像, (核宽度, 核高度))
    blur_mean = cv2.blur(img, (5, 5))  # 5×5 的均值核

    # --- 2. 高斯滤波 (Gaussian Blur) ---
    # 和均值滤波不同，高斯滤波给不同位置的像素不同权重：
    # 离中心越近权重越大（符合高斯分布/正态分布）
    # 效果比均值滤波更自然，是最常用的模糊方式
    # cv2.GaussianBlur(图像, (核宽, 核高), sigmaX)
    # sigmaX 控制高斯分布的"宽度"——越大越模糊
    # 如果 sigmaX=0，OpenCV 会根据核大小自动计算
    blur_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # --- 3. 中值滤波 (Median Blur) ---
    # 取 k×k 区域的中位数（而不是平均数）
    # 这是去除"椒盐噪声"的最佳方法！
    # 原理：椒盐噪声是极端的黑/白点，中位数会直接忽略这些极端值
    # cv2.medianBlur(图像, 核大小)  ← 注意核大小是单个整数
    blur_median = cv2.medianBlur(img, 5)

    # --- 4. 双边滤波 (Bilateral Filter) ---
    # "智能"模糊：只在颜色相近的区域模糊，颜色差异大的边缘保留
    # 这就是"磨皮不磨边"的原理——皮肤内部平滑，但五官轮廓保留
    # cv2.bilateralFilter(图像, 邻域直径, sigmaColor, sigmaSpace)
    #   sigmaColor: 颜色差异多大时停止平滑
    #   sigmaSpace: 空间距离的权重
    blur_bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    show_multiple_images([
        ("原图", img),
        ("均值滤波 (5x5)", blur_mean),
        ("高斯滤波 (5x5)", blur_gaussian),
        ("中值滤波 (5x5)", blur_median),
        ("双边滤波 (磨皮)", blur_bilateral),
    ], cols=2)
    cv2.destroyAllWindows()

    # --- 核大小对高斯模糊效果的影响 ---
    ksize_list = [3, 7, 15, 31]  # 不同核大小
    blur_sizes = []
    for k in ksize_list:
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        cv2.putText(blurred, f"kernel={k}x{k}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        blur_sizes.append((f"核={k}x{k}", blurred))

    show_multiple_images(blur_sizes, cols=2)
    cv2.destroyAllWindows()

    print("   💡 去椒盐噪声用中值滤波，日常模糊用高斯滤波，磨皮用双边滤波")


def demo_denoising():
    """
    演示2: 图像去噪实战

    先给图像加噪声，再用不同滤波器去除——直观理解各种滤波器的去噪能力。
    """
    print("\n--- 📖 演示2: 图像去噪对比 ---")
    img = load_image("sample.jpg")

    # --- 添加高斯噪声 ---
    noisy_gauss = add_noise(img, "gaussian")
    # 用不同滤波器去噪
    denoised_g_mean    = cv2.blur(noisy_gauss, (5, 5))                 # 均值
    denoised_g_gaussian = cv2.GaussianBlur(noisy_gauss, (5, 5), 0)    # 高斯
    denoised_g_median  = cv2.medianBlur(noisy_gauss, 5)               # 中值

    # --- 添加椒盐噪声 ---
    noisy_sp = add_noise(img, "salt_pepper")
    # 用不同滤波器去噪
    denoised_sp_mean    = cv2.blur(noisy_sp, (5, 5))
    denoised_sp_gaussian = cv2.GaussianBlur(noisy_sp, (5, 5), 0)
    denoised_sp_median  = cv2.medianBlur(noisy_sp, 5)

    show_multiple_images([
        ("高斯噪声原图", noisy_gauss),
        ("均值滤波", denoised_g_mean),
        ("高斯滤波", denoised_g_gaussian),
        ("中值滤波", denoised_g_median),
    ], cols=2)
    cv2.destroyAllWindows()

    show_multiple_images([
        ("椒盐噪声原图", noisy_sp),
        ("均值滤波", denoised_sp_mean),
        ("高斯滤波", denoised_sp_gaussian),
        ("中值滤波 ✅(最佳)", denoised_sp_median),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 中值滤波对椒盐噪声效果最好（它的设计天然适合移除极端值）")


def demo_sharpen():
    """
    演示3: 图像锐化

    锐化是模糊的反操作——让边缘更清晰、细节更突出。

    锐化的原理：
        原图 + (原图 - 模糊图) × 强度
        其中 (原图 - 模糊图) 提取了"被模糊掉"的高频信息（边缘和细节）
        把这些信息加回原图，边缘就增强了

    实现方式：自定义卷积核
        cv2.filter2D() 可以用任意自定义核对图像进行卷积
    """
    print("\n--- 📖 演示3: 图像锐化 ---")
    img = load_image("sample.jpg")

    # --- 方法1: 使用预定义的锐化核 ---
    # 核中正负权重交错：中心权重很大，周围权重为负
    # 这个核的效果是：增强中心像素和周围像素的差异
    kernel_sharpen_1 = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ], dtype=np.float32)
    # cv2.filter2D(图像, 输出深度(-1表示和输入相同), 卷积核)
    sharpened_1 = cv2.filter2D(img, -1, kernel_sharpen_1)

    # --- 方法2: 更强的锐化核 ---
    kernel_sharpen_2 = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    sharpened_2 = cv2.filter2D(img, -1, kernel_sharpen_2)

    # --- 方法3: USM (Unsharp Masking) 反锐化掩模 ---
    # 这是最专业的锐化方法，Photoshop 里也叫这个名
    # 步骤: 模糊 → 计算差值 → 加回原图
    blurred = cv2.GaussianBlur(img, (9, 9), 10)  # 先模糊
    # cv2.addWeighted(图1, 权重1, 图2, 权重2, 常数)
    # 输出 = 图1 × 权重1 + 图2 × 权重2 + 常数
    # 这里: 原图×1.5 + 模糊图×(-0.5) = 原图 + (原图-模糊图)×0.5
    sharpened_usm = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    show_multiple_images([
        ("原图", img),
        ("温和锐化", sharpened_1),
        ("强力锐化", sharpened_2),
        ("USM 反锐化掩模", sharpened_usm),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 锐化 = 原图 + 高频信息(原图-模糊图)，核心在卷积核的设计")


def demo_custom_kernel():
    """
    演示4: 自定义卷积核实验

    自己设计卷积核，直观感受不同核的效果。
    这是理解卷积神经网络 (CNN) 中"滤波器学习"的基础！
    """
    print("\n--- 📖 演示4: 自定义卷积核实验 ---")
    img = load_image("sample.jpg")

    # 转为灰度图，效果更明显
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 不同的自定义核 ---
    kernels = {
        "恒等 (什么也不做)": np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32),
        "边缘检测 (水平)": np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32),
        "边缘检测 (垂直)": np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32),
        "浮雕效果": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),
        "模糊 (3x3)": np.ones((3,3), dtype=np.float32) / 9,
        "锐化": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
    }

    results = [("原图（灰度）", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))]
    for name, kernel in kernels.items():
        # 应用卷积
        filtered = cv2.filter2D(gray, -1, kernel)
        # 转为三通道以便显示
        filtered_3ch = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        cv2.putText(filtered_3ch, name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        results.append((name, filtered_3ch))

    show_multiple_images(results, cols=3)
    cv2.destroyAllWindows()

    print("   💡 不同的卷积核 → 完全不同的效果，这就是 CNN 中"学习滤波器"的原理")


# =============================================================================
# 本节总结
# =============================================================================
# 滤波器一览:
#   均值滤波:     cv2.blur(img, (k, k))                 — 简单平均，效果一般
#   高斯滤波:     cv2.GaussianBlur(img, (k, k), sigma)  — 最常用，效果自然
#   中值滤波:     cv2.medianBlur(img, k)                — 去椒盐噪声专用
#   双边滤波:     cv2.bilateralFilter(img, d, sC, sS)   — 保边去噪（磨皮）
#   自定义卷积:   cv2.filter2D(img, -1, kernel)         — 万能方法
#
# 核心数学: 卷积 —— 核在图像上滑动，逐像素加权求和
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🔧 处理篇 [1/4]: 图像滤波")
    print("="*60)

    demo_blur()          # 演示1: 各种模糊
    demo_denoising()     # 演示2: 去噪实战
    demo_sharpen()       # 演示3: 锐化
    demo_custom_kernel() # 演示4: 自定义卷积核

    print("\n✅ 处理篇 [1/4] 完成！")


if __name__ == "__main__":
    main()

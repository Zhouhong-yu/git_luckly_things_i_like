# =============================================================================
# 🔍 分析篇 [2/3] - histograms.py
# 直方图分析：直方图计算、均衡化、反向投影、直方图比较
# =============================================================================
# 学完本模块你应该理解：
#   1. 什么是图像直方图——像素值分布的统计图
#   2. 如何计算和可视化直方图
#   3. 直方图均衡化——自动改善图像对比度
#   4. 反向投影——用直方图做目标检测
#   5. 直方图比较——判断两张图的相似度
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 图像直方图
# =============================================================================
# 直方图 (Histogram) = 统计图像中每种像素值出现了多少次。
#
# 对于灰度图（像素值 0-255）：
#   直方图是一个有 256 个柱子的柱状图
#   第 i 个柱子的高度 = 图像中像素值为 i 的像素有多少个
#
# 直方图能告诉你什么？
#   - 偏暗的图像: 直方图的柱子都在左侧（像素值小）
#   - 偏亮的图像: 直方图的柱子都在右侧（像素值大）
#   - 低对比度:   柱子集中在中间一小段（所有像素值差不多）
#   - 高对比度:   柱子分布在整个 0-255 范围
#   - 过曝:       右侧柱子"堆积"（大量像素值=255）
#   - 欠曝:       左侧柱子"堆积"（大量像素值=0）
#
# 彩色图的直方图: 三个通道各自有一个直方图
# =============================================================================

def demo_calc_histogram():
    """
    演示1: 计算和可视化直方图

    学习如何计算直方图并用不同方式展示它。
    """
    print("\n--- 📖 演示1: 计算图像直方图 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 方法1: cv2.calcHist() —— OpenCV 原生 ---
    # cv2.calcHist(images, channels, mask, histSize, ranges)
    #   images:   图像列表 [img]
    #   channels: 要计算的通道索引 [0] 表示第一个通道
    #             [0,1,2] 可以计算三个通道的直方图
    #   mask:     掩模——只计算掩模中白色区域的直方图，None=整张图
    #   histSize: 每个通道分多少"柱子"(bins)
    #   ranges:   像素值的范围 [0, 256]
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    print(f"   直方图形状: {hist.shape}")  # (256, 1) —— 256个bins
    print(f"   最暗像素(值=0)的个数: {hist[0][0]:.0f}")
    print(f"   最亮像素(值=255)的个数: {hist[255][0]:.0f}")

    # --- 方法2: numpy.histogram() —— NumPy 原生 ---
    hist_np, bins = np.histogram(gray.ravel(), 256, [0, 256])
    print(f"   NumPy 直方图前5项: {hist_np[:5]}")

    # --- 可视化直方图 ---
    try:
        import matplotlib
        matplotlib.use("TkAgg")  # 优先使用 Tkinter 后端
    except Exception:
        pass  # 如果 TkAgg 不可用，使用默认后端
    import matplotlib.pyplot as plt

    # 1. 灰度直方图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("灰度原图"), axes[0].axis("off")

    axes[1].plot(hist, color="black")
    axes[1].set_xlabel("像素值"), axes[1].set_ylabel("像素数量")
    axes[1].set_title("灰度直方图"), axes[1].set_xlim([0, 256])

    # 2. 彩色直方图
    colors = ("b", "g", "r")  # OpenCV 使用 BGR 顺序
    for i, color in enumerate(colors):
        hist_color = cv2.calcHist([img], [i], None, [256], [0, 256])
        axes[2].plot(hist_color, color=color, alpha=0.7)
    axes[2].set_xlabel("像素值"), axes[2].set_ylabel("像素数量")
    axes[2].set_title("BGR 彩色直方图"), axes[2].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()

    print("   💡 直方图 = 图像像素值的"人口普查"——告诉你每种亮度有多少像素")


def demo_histogram_equalization():
    """
    演示2: 直方图均衡化——自动改善对比度

    直方图均衡化 (Histogram Equalization) 是一种自动增强图像对比度的方法。
    它的核心思想: 重新映射像素值，让直方图变得更"均匀"。
    (原本集中在一块的柱子被"拉开"，分布到整个 0-255 范围)

    应用场景:
        - 医学图像增强（X光片、CT）
        - 曝光不足/过度的照片修正
        - 改善人脸识别的输入质量
    """
    print("\n--- 📖 演示2: 直方图均衡化 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 模拟低对比度图像 ---
    # 把像素值压缩到 50-150 范围（正常情况下应该是 0-255）
    low_contrast = ((gray.astype(np.float32) - gray.min()) /
                    (gray.max() - gray.min()) * 100 + 50).astype(np.uint8)

    # --- 标准直方图均衡化 ---
    # cv2.equalizeHist() 是全局均衡化
    # 它对整张图使用同一个映射函数
    equalized = cv2.equalizeHist(low_contrast)

    # --- CLAHE (自适应直方图均衡化) ---
    # 全局均衡化的问题: 在亮区和暗区都使用同一个映射
    # 可能会导致亮区过曝、暗区欠曝
    # CLAHE 的解决方案: 把图像分成小块，每个块独立做均衡化
    #                  然后再平滑拼接，限制对比度放大程度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clipLimit: 对比度限制阈值（防止噪声被放大）
    # tileGridSize: 分成多少个格子
    clahe_equalized = clahe.apply(low_contrast)

    # --- 可视化对比 ---
    try:
        import matplotlib
        matplotlib.use("TkAgg")  # 优先使用 Tkinter 后端
    except Exception:
        pass  # 如果 TkAgg 不可用，使用默认后端
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    images = [
        (gray, "原图"),
        (low_contrast, "低对比度"),
        (equalized, "全局均衡化"),
        (clahe_equalized, "CLAHE 自适应均衡化"),
    ]

    for i, (im, title) in enumerate(images):
        row, col = i // 2, i % 2
        axes[row][col].imshow(im, cmap="gray")
        axes[row][col].set_title(title)
        axes[row][col].axis("off")

    # 第5个子图放直方图对比
    axes[1][2].hist(low_contrast.ravel(), bins=256, range=[0, 256],
                    alpha=0.5, label="低对比度", color="red")
    axes[1][2].hist(equalized.ravel(), bins=256, range=[0, 256],
                    alpha=0.5, label="均衡化后", color="blue")
    axes[1][2].set_xlabel("像素值"), axes[1][2].set_ylabel("数量")
    axes[1][2].set_title("直方图对比"), axes[1][2].legend()

    plt.tight_layout()
    plt.show()

    print("   💡 均衡化 = 把像素值'拉开'到整个范围; CLAHE = 分块均衡化(更优)")


def demo_back_projection():
    """
    演示3: 反向投影——用直方图做目标定位

    反向投影 (Back Projection) 回答了这样一个问题：
    "图像中哪些像素的颜色和我感兴趣的目标相似？"

    原理:
        1. 计算目标(ROI)的颜色直方图
        2. 对整个图像，计算每个像素在目标直方图中出现的概率
        3. 概率高的区域就是"和目标颜色相似"的区域

    这常用于基于颜色的目标跟踪——比如跟踪一个红色的球。
    """
    print("\n--- 📖 演示3: 反向投影 — 颜色目标定位 ---")
    img = load_image("sample.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 步骤1: 选择一个 ROI（感兴趣区域）---
    # 假装我们选了一个红色区域作为跟踪目标
    # 使用颜色检测模拟
    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])
    roi_mask = cv2.inRange(hsv, lower, upper)
    roi = cv2.bitwise_and(img, img, mask=roi_mask)

    # --- 步骤2: 计算 ROI 的颜色直方图 ---
    # 只使用 H(色调) 和 S(饱和度) 通道，忽略 V(亮度)
    # 因为亮度容易受光照影响
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 只计算 mask 中白色区域的直方图
    roi_hist = cv2.calcHist(
        [roi_hsv], [0, 1], roi_mask,  # 通道0(H)和1(S)
        [180, 256],                    # H 分180个bin, S 分256个bin
        [0, 180, 0, 256]              # H 范围 0-180, S 范围 0-256
    )

    # --- 步骤3: 归一化直方图 ---
    # 把直方图的值缩放到 0-255 范围内（方便显示和后续使用）
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # --- 步骤4: 反向投影 ---
    # cv2.calcBackProject() 对全图计算每个像素属于目标的概率
    back_proj = cv2.calcBackProject(
        [hsv],          # 待搜索的图像（HSV）
        [0, 1],         # 使用的通道
        roi_hist,       # 目标直方图
        [0, 180, 0, 256],  # 每个通道的范围
        1               # 缩放因子
    )

    # --- 步骤5: 对反向投影结果做后处理 ---
    # 用一个圆形核做卷积，让结果更平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    back_proj = cv2.filter2D(back_proj, -1, kernel)

    # 阈值化找出最可能的区域
    _, thresh = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge([thresh, thresh, thresh])  # 转为三通道用于显示

    # 把结果叠加到原图上
    result = cv2.bitwise_and(img, thresh)

    roi_show = roi if roi.any() else img[:50, :50]
    back_proj_show = cv2.cvtColor(back_proj.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    thresh_show = cv2.cvtColor(thresh[:, :, 0], cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("原图", img),
        ("ROI (红色区域)", roi_show),
        ("反向投影 (概率图)", back_proj_show),
        ("阈值化结果 (目标位置)", result),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 反向投影 = '图像中哪里和我的目标颜色相似？' → 概率图")


def demo_histogram_comparison():
    """
    演示4: 直方图比较——量化两张图的相似度

    直方图可以作为图像的"指纹"，通过比较直方图来判断图像相似度。

    比较方法:
        - 相关性 (Correlation):      [0,1]，越大越相似
        - 卡方 (Chi-Square):        [0,∞)，越小越相似
        - 交集 (Intersection):      越大越相似
        - Bhattacharyya 距离:       [0,1]，越小越相似
    """
    print("\n--- 📖 演示4: 直方图比较 — 图像相似度 ---")
    img = load_image("sample.jpg")

    # 创建几个"变体"用于比较
    img_darker = (img * 0.5).astype(np.uint8)       # 变暗
    img_brighter = np.clip(img * 1.5, 0, 255).astype(np.uint8)  # 变亮
    img_flipped = cv2.flip(img, 1)                   # 水平翻转（内容变了）

    # 计算每张图的直方图
    def get_hist(image):
        """辅助函数: 计算三通道直方图并归一化"""
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()  # 展平为一维数组

    hist_orig = get_hist(img)
    hist_dark = get_hist(img_darker)
    hist_bright = get_hist(img_brighter)
    hist_flip = get_hist(img_flipped)

    # 比较方法
    methods = {
        "相关性 (越接近1越好)": cv2.HISTCMP_CORREL,
        "卡方 (越接近0越好)": cv2.HISTCMP_CHISQR,
        "交集 (越大越好)": cv2.HISTCMP_INTERSECT,
        "Bhattacharyya (越接近0越好)": cv2.HISTCMP_BHATTACHARYYA,
    }

    for name, method in methods.items():
        print(f"\n   [{name}]:")
        # cv2.compareHist(H1, H2, method)
        print(f"     原图 vs 暗图:  {cv2.compareHist(hist_orig, hist_dark, method):.4f}")
        print(f"     原图 vs 亮图:  {cv2.compareHist(hist_orig, hist_bright, method):.4f}")
        print(f"     原图 vs 翻转:  {cv2.compareHist(hist_orig, hist_flip, method):.4f}")

    show_multiple_images([
        ("原图", img),
        ("变暗 (50%)", img_darker),
        ("变亮 (150%)", img_brighter),
        ("水平翻转", img_flipped),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 直方图 = 图像'指纹'，比较直方图 = 比较两张图的颜色分布相似度")


# =============================================================================
# 本节总结
# =============================================================================
# 直方图计算:
#   cv2.calcHist([img], [channels], mask, [bins], [range])
#
# 直方图均衡化:
#   全局: cv2.equalizeHist(gray)                  — 简单但可能过度
#   CLAHE: cv2.createCLAHE().apply(gray)          — 自适应，推荐!
#
# 反向投影:
#   cv2.calcBackProject()                        — 颜色目标定位
#
# 直方图比较:
#   cv2.compareHist(H1, H2, method)              — 图像相似度
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🔍 分析篇 [2/3]: 直方图分析")
    print("="*60)

    demo_calc_histogram()           # 演示1: 计算直方图
    demo_histogram_equalization()   # 演示2: 均衡化
    demo_back_projection()          # 演示3: 反向投影
    demo_histogram_comparison()     # 演示4: 直方图比较

    print("\n✅ 分析篇 [2/3] 完成！")


if __name__ == "__main__":
    main()

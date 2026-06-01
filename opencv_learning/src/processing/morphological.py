# =============================================================================
# 🔧 处理篇 [3/4] - morphological.py
# 形态学操作：膨胀、腐蚀、开运算、闭运算、形态学梯度
# =============================================================================
# 学完本模块你应该理解：
#   1. 什么是形态学操作——基于形状的图像处理
#   2. 膨胀 (Dilation) 和腐蚀 (Erosion) 的原理
#   3. 开运算、闭运算的用途
#   4. 形态学梯度、顶帽、黑帽
#   5. 结构元素 (Structuring Element) 的作用
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 形态学操作
# =============================================================================
# 形态学 (Morphology) 意思是"关于形状的研究"。
# 形态学操作基于图像的"形状"来处理图像，主要用于二值图像（黑白图）。
#
# 核心思想: 用一个叫做"结构元素 (Structuring Element)"的小形状，
#           在图像上滑动，根据与图像内容的重叠情况来决定输出。
#
# 想象结构元素是一个"模板印章"：
#   - 膨胀 (Dilation):   印章覆盖到的任何地方都变成白色
#                        白色区域"膨胀"了 → 变大
#   - 腐蚀 (Erosion):    只有印章完全在白色区域内，中心才保留白色
#                        白色区域被"腐蚀"了 → 变小
#
# 结构元素形状:
#   - 矩形 (MORPH_RECT):   最常用
#   - 椭圆 (MORPH_ELLIPSE): 圆滑效果
#   - 十字 (MORPH_CROSS):   保留细线
# =============================================================================

def demo_erosion_dilation():
    """
    演示1: 腐蚀与膨胀——形态学的基本操作

    这两个操作是所有其他形态学操作的基础。

    腐蚀 (Erosion):
        - 白色区域边界被"侵蚀"
        - 小白色噪点消失
        - 两个接近的物体会分得更开
        - 公式: 输出 = min(结构元素覆盖区域的像素值)

    膨胀 (Dilation):
        - 白色区域向外"膨胀"
        - 小的黑色空洞被填满
        - 两个接近的物体会连在一起
        - 公式: 输出 = max(结构元素覆盖区域的像素值)
    """
    print("\n--- 📖 演示1: 腐蚀与膨胀 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 先转为二值图像 ---
    # 形态学操作通常在二值图像上进行
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # --- 创建结构元素 ---
    # cv2.getStructuringElement(形状, (宽, 高))
    # 它返回一个 numpy 数组，就是我们的"模板印章"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    print(f"   结构元素 (5x5 矩形):\n{kernel * 1}")  # *1 是为了让 True/False 变成 1/0

    # --- 腐蚀: cv2.erode() ---
    # 结构元素滑过图像，只有结构元素"完全被白色覆盖"时输出才是白色
    eroded = cv2.erode(binary, kernel, iterations=1)
    # iterations=1 表示做一次腐蚀。iterations=3 表示连续做三次（效果更强）

    # --- 膨胀: cv2.dilate() ---
    # 结构元素滑过图像，只要结构元素"碰到一点白色"输出就是白色
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # --- 对比不同迭代次数 ---
    eroded_3 = cv2.erode(binary, kernel, iterations=3)
    dilated_3 = cv2.dilate(binary, kernel, iterations=3)

    bin_show = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    eroded_show = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    dilated_show = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    eroded3_show = cv2.cvtColor(eroded_3, cv2.COLOR_GRAY2BGR)
    dilated3_show = cv2.cvtColor(dilated_3, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("二值原图", bin_show),
        ("腐蚀 1次 (白色变少)", eroded_show),
        ("腐蚀 3次 (消失更多)", eroded3_show),
        ("膨胀 1次 (白色变多)", dilated_show),
        ("膨胀 3次 (膨胀更厉害)", dilated3_show),
    ], cols=3)
    cv2.destroyAllWindows()

    print("   💡 腐蚀让白色区域变小（去噪点），膨胀让白色区域变大（填空洞）")


def demo_opening_closing():
    """
    演示2: 开运算与闭运算——腐蚀和膨胀的组合

    开运算 (Opening)  = 先腐蚀，再膨胀
        - 可以去掉小的白色噪点
        - 可以分开两个粘连的物体
        - 平滑物体边界
        - 像"打开"了两个物体之间的连接

    闭运算 (Closing)  = 先膨胀，再腐蚀
        - 可以填满小的黑色空洞
        - 可以连接两个相近的物体
        - 平滑物体边界
        - 像"关闭"了物体内部的缺口
    """
    print("\n--- 📖 演示2: 开运算与闭运算 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # --- 给二值图添加一些噪点来演示效果 ---
    # 添加白色噪点（椒盐噪声的"盐"）
    noisy = binary.copy()
    noise_mask = np.random.random(binary.shape[:2]) < 0.01
    noisy[noise_mask] = 255  # 白色小点
    # 添加黑色小洞
    hole_mask = np.random.random(binary.shape[:2]) < 0.01
    noisy[hole_mask] = 0  # 黑色小洞

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # --- 开运算: cv2.morphologyEx(..., cv2.MORPH_OPEN) ---
    # 先腐蚀（去掉小白点）→ 再膨胀（恢复物体原大小）
    opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)

    # --- 闭运算: cv2.morphologyEx(..., cv2.MORPH_CLOSE) ---
    # 先膨胀（填满小黑洞）→ 再腐蚀（恢复物体原大小）
    closed = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, kernel)

    # --- 两者结合: 先开后闭（常用降噪套路）---
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    noisy_show = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    opened_show = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    closed_show = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    cleaned_show = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("带噪声的二值图", noisy_show),
        ("开运算 (去白噪点)", opened_show),
        ("闭运算 (填黑空洞)", closed_show),
        ("开后闭 (干净!)", cleaned_show),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 开运算 = 去白色噪点；闭运算 = 填黑色空洞；结合使用 = 图像降噪")


def demo_morphology_ex():
    """
    演示3: 更多形态学操作

    除了开闭运算，还有：
        - 形态学梯度 = 膨胀 - 腐蚀 → 得到物体轮廓
        - 顶帽 (Top Hat) = 原图 - 开运算 → 提取比周围亮的区域
        - 黑帽 (Black Hat) = 闭运算 - 原图 → 提取比周围暗的区域
    """
    print("\n--- 📖 演示3: 形态学梯度、顶帽、黑帽 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # --- 形态学梯度 ---
    # 膨胀图 - 腐蚀图 = 物体的边界轮廓
    # 这比直接用 Canny 得到的边缘更"干净"（对于二值图）
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    # --- 顶帽 (Top Hat) ---
    # 原图 - 开运算结果 = 比周围亮的"突起"
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # --- 黑帽 (Black Hat) ---
    # 闭运算结果 - 原图 = 比周围暗的"凹陷"
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    bin_show = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    grad_show = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    tophat_show = cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)
    blackhat_show = cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("二值原图", bin_show),
        ("形态学梯度 (边界)", grad_show),
        ("顶帽 (亮区域)", tophat_show),
        ("黑帽 (暗区域)", blackhat_show),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 梯度=膨胀-腐蚀→边界；顶帽→亮斑；黑帽→暗斑")


def demo_kernel_comparison():
    """
    演示4: 不同结构元素的对比

    结构元素的形状和大小决定了形态学操作的效果。
    """
    print("\n--- 📖 演示4: 不同结构元素对比 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 不同形状的结构元素
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11))

    # 使用不同结构元素做开运算
    opened_rect = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_rect)
    opened_ellipse = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_ellipse)
    opened_cross = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_cross)

    bin_show = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    rect_show = cv2.cvtColor(opened_rect, cv2.COLOR_GRAY2BGR)
    ellipse_show = cv2.cvtColor(opened_ellipse, cv2.COLOR_GRAY2BGR)
    cross_show = cv2.cvtColor(opened_cross, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("二值原图", bin_show),
        ("矩形核 (MORPH_RECT)", rect_show),
        ("椭圆核 (MORPH_ELLIPSE)", ellipse_show),
        ("十字核 (MORPH_CROSS)", cross_show),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 矩形核保留直角，椭圆核产生圆滑效果，十字核保留细线结构")


# =============================================================================
# 本节总结
# =============================================================================
# 核心操作:
#   腐蚀:    cv2.erode(img, kernel)         — 白色缩小
#   膨胀:    cv2.dilate(img, kernel)        — 白色膨胀
#
# 组合操作 (cv2.morphologyEx):
#   开运算:  MORPH_OPEN   先腐蚀后膨胀  — 去白噪点
#   闭运算:  MORPH_CLOSE  先膨胀后腐蚀  — 填黑洞
#   梯度:    MORPH_GRADIENT 膨胀-腐蚀   — 物体轮廓
#   顶帽:    MORPH_TOPHAT  原图-开运算  — 亮区域
#   黑帽:    MORPH_BLACKHAT 闭运算-原图 — 暗区域
#
# 结构元素: cv2.getStructuringElement(形状, 大小)
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🔧 处理篇 [3/4]: 形态学操作")
    print("="*60)

    demo_erosion_dilation()    # 演示1: 腐蚀与膨胀
    demo_opening_closing()     # 演示2: 开闭运算
    demo_morphology_ex()       # 演示3: 梯度、顶帽、黑帽
    demo_kernel_comparison()   # 演示4: 结构元素对比

    print("\n✅ 处理篇 [3/4] 完成！")


if __name__ == "__main__":
    main()

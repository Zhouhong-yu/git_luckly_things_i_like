# =============================================================================
# 🔧 处理篇 [2/4] - edge_detection.py
# 边缘检测：Canny、Sobel、Laplacian
# =============================================================================
# 学完本模块你应该理解：
#   1. 什么是"边缘"——图像中像素值剧烈变化的地方
#   2. 梯度的概念——边缘检测的数学基础
#   3. Canny 边缘检测（最经典、最常用的算法）
#   4. Sobel 算子——计算图像梯度
#   5. Laplacian 算子——检测二阶导数过零点
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 什么是边缘？
# =============================================================================
# 边缘 (Edge) = 图像中像素值变化剧烈的地方
#
# 想象灰度图像是一座"山"（像素值 = 高度）：
#   - 平坦区域: 像素值差不多 → 没有边缘
#   - 斜坡:     像素值逐渐变化 → 弱边缘
#   - 悬崖:     像素值突然跳变 → 强边缘
#
# 怎么找到"悬崖"？用导数（梯度 Gradient）！
#
# 梯度 = 变化率 = 相邻像素的差值
#   水平梯度 Gx = 右边像素 - 左边像素（用 [-1, 0, 1] 核卷积）
#   垂直梯度 Gy = 下边像素 - 上边像素（用 [-1, 0, 1]ᵀ 核卷积）
#   梯度大小 = sqrt(Gx² + Gy²)  —— 变化有多剧烈
#   梯度方向 = atan2(Gy, Gx)     —— 变化的方向
#
# 梯度的计算也是通过卷积实现的！这就是为什么上一节学滤波很重要。
# =============================================================================


def demo_sobel():
    """
    演示1: Sobel 算子——计算图像的一阶导数（梯度）

    Sobel 算子是边缘检测的基础工具。
    它使用两个 3×3 的卷积核分别计算水平和垂直梯度：
        Gx 核（水平）：[-1 0 1; -2 0 2; -1 0 1]
        Gy 核（垂直）：[-1 -2 -1; 0 0 0; 1 2 1]

    为什么中间行/列权重是 2？
        离中心越近的像素应该权重越大（类似于高斯平滑）
        这样既计算了梯度，又有一定的降噪能力
    """
    print("\n--- 📖 演示1: Sobel 算子 — 图像梯度 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- cv2.Sobel() 参数详解 ---
    # cv2.Sobel(src, ddepth, dx, dy, ksize)
    #   src:    输入图像（通常是灰度图）
    #   ddepth: 输出图像的深度。cv2.CV_64F = 64位浮点数
    #           要用浮点数因为梯度可能是负数（从亮到暗 vs 从暗到亮）
    #   dx:     对 x 求导的阶数（1 = 一阶导数）
    #   dy:     对 y 求导的阶数
    #   ksize:  Sobel 核的大小（必须是奇数: 1,3,5,7...）

    # 水平梯度: 检测垂直边缘（因为水平方向的剧烈变化 = 垂直边缘）
    # dx=1, dy=0  → 只计算水平方向的变化率
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # 垂直梯度: 检测水平边缘
    # dx=0, dy=1  → 只计算垂直方向的变化率
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 取绝对值——我们关心变化有多大，不关心方向
    # cv2.convertScaleAbs() 把浮点数转为 uint8 并取绝对值
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # 合并水平和垂直梯度
    # cv2.addWeighted() 加权求和
    sobel_combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    grad_x_show = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
    grad_y_show = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)
    sobel_show = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("原图（灰度）", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
        ("水平梯度 Gx (垂直边缘)", grad_x_show),
        ("垂直梯度 Gy (水平边缘)", grad_y_show),
        ("合并梯度", sobel_show),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 Gx 检测垂直边缘（←→变化），Gy 检测水平边缘（↑↓变化）")


def demo_laplacian():
    """
    演示2: Laplacian 算子——二阶导数

    如果 Sobel 是"一阶导数"（速度），那 Laplacian 就是"二阶导数"（加速度）。

    一阶导数: 告诉我们"像素值在变化"
    二阶导数: 告诉我们"变化率在变化"——在边缘处，二阶导数会过零

    Laplacian 核: [[0,1,0], [1,-4,1], [0,1,0]]
    这是对图像求二阶导数的离散近似。

    优点: 对任何方向的边缘都敏感（各向同性）
    缺点: 对噪声非常敏感（二阶导数放大了噪声）
    """
    print("\n--- 📖 演示2: Laplacian 算子 — 二阶导数 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 先用高斯模糊降噪 ---
    # Laplacian 对噪声极其敏感，必须先去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- cv2.Laplacian() ---
    # 参数和 Sobel 类似，但没有 dx, dy 参数（Laplacian 对所有方向一视同仁）
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # 取绝对值
    abs_laplacian = cv2.convertScaleAbs(laplacian)

    # --- 对比: 有去噪 vs 无去噪 ---
    laplacian_noisy = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    abs_laplacian_noisy = cv2.convertScaleAbs(laplacian_noisy)

    gray_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lap_show = cv2.cvtColor(abs_laplacian, cv2.COLOR_GRAY2BGR)
    lap_noisy_show = cv2.cvtColor(abs_laplacian_noisy, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("原图", gray_show),
        ("Laplacian (先去噪)", lap_show),
        ("Laplacian (直接算=噪声很大!)", lap_noisy_show),
    ])
    cv2.destroyAllWindows()

    print("   💡 Laplacian 对噪声敏感 → 必须先高斯模糊！(这就是 LoG 的思想)")


def demo_canny():
    """
    演示3: Canny 边缘检测——最经典的边缘检测算法

    Canny 是 John Canny 在 1986 年提出的算法，至今仍是最广泛使用的边缘检测方法。
    它的输出是清晰的二值图（边缘=白，非边缘=黑），非常适合作为后续处理的输入。

    Canny 算法的四个步骤:
    ┌─────────────────────────────────────────────────────────────┐
    │ 步骤1: 高斯滤波    → 去除噪声                               │
    │ 步骤2: 计算梯度    → 用 Sobel 算子计算每个像素的梯度和方向   │
    │ 步骤3: 非极大值抑制 → 只保留梯度方向上最强的边缘（细化边缘）  │
    │ 步骤4: 双阈值+滞后  → 强边缘直接保留，弱边缘仅当与强边缘相连  │
    │                      时才保留（抑制虚假边缘）                │
    └─────────────────────────────────────────────────────────────┘

    两个阈值的作用:
        - 高阈值 (threshold2): 高于此值的肯定是边缘
        - 低阈值 (threshold1): 低于此值的肯定不是边缘
        - 介于两者之间的: 只有当它和"确定是边缘"的像素相邻时才保留
    """
    print("\n--- 📖 演示3: Canny 边缘检测 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- cv2.Canny(image, threshold1, threshold2) ---
    # threshold1: 低阈值（弱边缘的门槛）
    # threshold2: 高阈值（强边缘的门槛）
    # 推荐 ratio: threshold1 : threshold2 = 1:2 或 1:3

    # 不同阈值组合的效果对比
    canny_1 = cv2.Canny(gray, 50, 150)    # 低阈值=50,  高阈值=150  (标准)
    canny_2 = cv2.Canny(gray, 30, 100)    # 更低 → 检测到更多边缘（可能包含噪声）
    canny_3 = cv2.Canny(gray, 100, 200)   # 更高 → 只保留最显著的边缘
    canny_4 = cv2.Canny(gray, 200, 250)   # 非常高 → 只有极少数边缘

    gray_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    results = [("原图", gray_show)]
    for name, canny_img in [("低阈值 (30,100)", canny_2), ("标准 (50,150)", canny_1),
                              ("高阈值 (100,200)", canny_3), ("极高 (200,250)", canny_4)]:
        canny_3ch = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
        edge_count = cv2.countNonZero(canny_img)  # 统计边缘像素数
        cv2.putText(canny_3ch, f"边缘像素: {edge_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        results.append((name, canny_3ch))

    show_multiple_images(results, cols=2)
    cv2.destroyAllWindows()

    print("   💡 双阈值是 Canny 的精髓——既去噪声又不漏掉真正的弱边缘")


def demo_canny_pipeline():
    """
    演示4: Canny 边缘检测完整流程

    深入展示 Canny 算法的每个步骤，理解背后的原理。
    """
    print("\n--- 📖 演示4: Canny 算法逐步拆解 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 步骤1: 高斯模糊 ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # --- 步骤2: 用 Sobel 计算梯度 ---
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # 梯度大小 = sqrt(Gx² + Gy²)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    # 梯度方向 = atan2(Gy, Gx)，转为角度
    gradient_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    gradient_direction = np.abs(gradient_direction)  # 取绝对值用于显示

    # --- 步骤3: 非极大值抑制 (NMS) ---
    # 对于每个像素，沿梯度方向检查它是否是最强的
    # 如果是 → 保留；如果旁边有更强的 → 抑制（设为0）
    # 这样边缘就从"模糊的一条带"变成"清晰的一条线"
    nms = np.zeros_like(gradient_magnitude)
    mag = gradient_magnitude.astype(np.float32)
    direction = np.arctan2(grad_y, grad_x)

    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            angle = direction[i, j]
            # 将角度量化到4个方向: 0°, 45°, 90°, 135°
            if (-np.pi/8 <= angle < np.pi/8) or (angle >= 7*np.pi/8) or (angle < -7*np.pi/8):
                neighbors = [mag[i, j-1], mag[i, j+1]]  # 水平方向
            elif (np.pi/8 <= angle < 3*np.pi/8) or (-7*np.pi/8 <= angle < -5*np.pi/8):
                neighbors = [mag[i-1, j+1], mag[i+1, j-1]]  # 45°方向
            elif (3*np.pi/8 <= angle < 5*np.pi/8) or (-5*np.pi/8 <= angle < -3*np.pi/8):
                neighbors = [mag[i-1, j], mag[i+1, j]]  # 垂直方向
            else:
                neighbors = [mag[i-1, j-1], mag[i+1, j+1]]  # 135°方向

            if mag[i, j] >= max(neighbors):
                nms[i, j] = mag[i, j]

    nms = np.clip(nms, 0, 255).astype(np.uint8)

    # --- OpenCV 的 Canny 作为参考 ---
    canny_final = cv2.Canny(gray, 50, 150)

    grad_show = cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR)
    nms_show = cv2.cvtColor(nms, cv2.COLOR_GRAY2BGR)
    canny_show = cv2.cvtColor(canny_final, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("1. 原图 (灰度)", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
        ("2. 高斯模糊", cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)),
        ("3. 梯度大小 (Sobel)", grad_show),
        ("4. 非极大值抑制 (NMS)", nms_show),
        ("5. Canny 最终结果", canny_show),
    ], cols=3)
    cv2.destroyAllWindows()

    print("   💡 这就是 Canny 算法的完整流程——每一步都不可或缺")


# =============================================================================
# 本节总结
# =============================================================================
# 边缘检测三种方法对比:
#   Sobel:     计算一阶导数（梯度），简单快速，适合提取特定方向的边缘
#   Laplacian: 计算二阶导数，对所有方向敏感但对噪声极其敏感，需先模糊
#   Canny:     1986年的算法但仍是金标准——输出清晰的二值边缘图
#               核心四步: 模糊→梯度→NMS→双阈值
#
# cv2.Canny(img, threshold1, threshold2) 是最常用的边缘检测函数
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🔧 处理篇 [2/4]: 边缘检测")
    print("="*60)

    demo_sobel()          # 演示1: Sobel 梯度
    demo_laplacian()      # 演示2: Laplacian
    demo_canny()          # 演示3: Canny 边缘检测
    demo_canny_pipeline() # 演示4: Canny 逐步拆解

    print("\n✅ 处理篇 [2/4] 完成！")


if __name__ == "__main__":
    main()

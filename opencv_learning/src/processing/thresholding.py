# =============================================================================
# 🔧 处理篇 [4/4] - thresholding.py
# 阈值化技术：二值化、自适应阈值、Otsu 算法
# =============================================================================
# 学完本模块你应该理解：
#   1. 什么是阈值化——把灰度图变成黑白图
#   2. 全局阈值 vs 自适应阈值
#   3. Otsu 算法——自动选择最佳阈值
#   4. 各种阈值类型及其适用场景
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 阈值化 (Thresholding)
# =============================================================================
# 阈值化 = 把灰度图像变成只有黑白两色的二值图像
#
# 规则很简单:
#   if 像素值 > 阈值:
#       输出 = 255 (白)
#   else:
#       输出 = 0   (黑)
#
# 为什么要做阈值化？
#   1. 把图像简化为"前景(目标)"和"背景"——这是很多算法的基础
#   2. 让轮廓检测、物体计数等操作成为可能
#   3. 数据压缩——每个像素从 256 种可能变成只有 2 种
#
# 阈值化的核心挑战: 选什么阈值？
#   - 太低: 太多东西变成白色（噪声也被当成了目标）
#   - 太高: 真正的目标也被当成了黑色（漏检）
#   - 不均匀光照: 同一阈值在亮处合适，在暗处就不合适
# =============================================================================


def demo_basic_threshold():
    """
    演示1: 基础阈值化

    cv2.threshold() 是最基本的阈值化函数。
    它返回两个值: retval(使用的阈值) 和 dst(输出的二值图像)。

    阈值类型:
        THRESH_BINARY:    src > thresh → 255, 否则 → 0
        THRESH_BINARY_INV: src > thresh → 0,   否则 → 255  (取反)
        THRESH_TRUNC:     src > thresh → thresh, 否则 → 不变
        THRESH_TOZERO:    src > thresh → 不变,   否则 → 0
        THRESH_TOZERO_INV: src > thresh → 0,     否则 → 不变
    """
    print("\n--- 📖 演示1: 基础阈值化 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"   灰度图统计: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")

    # --- 选一个阈值 ---
    thresh_value = 127  # 阈值：灰度值 127（0-255 的中点）

    # --- 不同阈值类型的效果 ---
    # cv2.threshold(src, thresh, maxval, type)
    #   src:     输入图像（灰度图）
    #   thresh:  阈值
    #   maxval:  当条件满足时，输出什么值（通常是 255）
    #   type:    阈值类型
    #   返回值:  (实际使用的阈值, 输出图像)

    ret, binary     = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    ret, binary_inv = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
    ret, trunc      = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_TRUNC)
    ret, tozero     = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_TOZERO)
    ret, tozero_inv = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_TOZERO_INV)

    results = [
        ("原图（灰度）", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
    ]
    type_names = [
        ("BINARY", binary),
        ("BINARY_INV", binary_inv),
        ("TRUNC", trunc),
        ("TOZERO", tozero),
        ("TOZERO_INV", tozero_inv),
    ]
    for name, res in type_names:
        res_3ch = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        cv2.putText(res_3ch, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        results.append((name, res_3ch))

    show_multiple_images(results, cols=3)
    cv2.destroyAllWindows()

    # --- 可视化: 不同阈值的选择 ---
    import matplotlib
    matplotlib.use("TkAgg")  # 使用 Tkinter 后端
    import matplotlib.pyplot as plt

    # 绘制灰度直方图 + 阈值线
    plt.figure(figsize=(10, 4))
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color="gray", alpha=0.7)
    plt.axvline(x=thresh_value, color="r", linestyle="--", linewidth=2,
                label=f"阈值 = {thresh_value}")
    plt.axvline(x=50, color="b", linestyle="--", linewidth=1, label="低阈值=50")
    plt.axvline(x=200, color="g", linestyle="--", linewidth=1, label="高阈值=200")
    plt.xlabel("像素灰度值"), plt.ylabel("像素数量")
    plt.title("灰度直方图与阈值选择"), plt.legend()
    plt.tight_layout()
    plt.show()

    print("   💡 阈值就像一条分界线——线以上是目标(白)，线以下是背景(黑)")


def demo_adaptive_threshold():
    """
    演示2: 自适应阈值

    全局固定阈值的问题：图像不同区域亮度不一样怎么办？
    比如一张扫描文档，左边有阴影，右边光线强——同一个阈值没法同时处理两边。

    自适应阈值 (Adaptive Thresholding) 的解决方案:
        每个像素用自己周围一个小窗口来计算自己的"专属阈值"
        暗的地方阈值自动调低，亮的地方阈值自动调高

    两种计算方法:
        ADAPTIVE_THRESH_MEAN_C:  阈值 = 窗口平均值 - 常数C
        ADAPTIVE_THRESH_GAUSSIAN_C: 阈值 = 窗口加权平均值(Gaussian) - 常数C
    """
    print("\n--- 📖 演示2: 自适应阈值 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 模拟不均匀光照 ---
    # 创建一个从左到右的渐变，模拟"左边暗右边亮"
    h, w = gray.shape
    gradient = np.tile(np.linspace(0.3, 1.5, w), (h, 1)).astype(np.float32)
    uneven = (gray.astype(np.float32) * gradient).clip(0, 255).astype(np.uint8)

    # --- 全局阈值（对比）---
    _, global_thresh = cv2.threshold(uneven, 127, 255, cv2.THRESH_BINARY)

    # --- 自适应阈值 ---
    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    #   blockSize: 邻域窗口大小（必须是奇数）
    #   C:         从平均值中减去的常数（调整灵敏度）

    adaptive_mean = cv2.adaptiveThreshold(
        uneven, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,      # 使用窗口平均值
        cv2.THRESH_BINARY,
        11,   # blockSize: 11x11 窗口
        2     # C: 阈值 = 平均值 - 2
    )

    adaptive_gaussian = cv2.adaptiveThreshold(
        uneven, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 使用窗口高斯加权平均
        cv2.THRESH_BINARY,
        11,
        2
    )

    # --- 可视化对比 ---
    uneven_show = cv2.cvtColor(uneven, cv2.COLOR_GRAY2BGR)
    global_show = cv2.cvtColor(global_thresh, cv2.COLOR_GRAY2BGR)
    amean_show = cv2.cvtColor(adaptive_mean, cv2.COLOR_GRAY2BGR)
    agauss_show = cv2.cvtColor(adaptive_gaussian, cv2.COLOR_GRAY2BGR)

    cv2.putText(global_show, "全局阈值 (失败)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(amean_show, "自适应-MEAN (成功)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(agauss_show, "自适应-GAUSSIAN (成功)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    show_multiple_images([
        ("不均匀光照原图", uneven_show),
        ("全局阈值 (左暗失败)", global_show),
        ("自适应均值 (效果OK!)", amean_show),
        ("自适应高斯 (效果OK!)", agauss_show),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 光照不均匀时，自适应阈值是救星——每个像素有自己的专属阈值")


def demo_otsu():
    """
    演示3: Otsu 大津算法——自动选择最佳阈值

    Otsu 算法会自动寻找"最好的"阈值，不需要手动指定！
    它的原理:
        1. 尝试所有可能的阈值（0到255）
        2. 对每个阈值，计算它把图像分成两类后的"类间方差"
        3. 类间方差最大的那个阈值就是最佳阈值

    什么是"类间方差"？
        方差越大 = 两类（前景和背景）差异越大 = 分得越好
        方差越小 = 两类混在一起 = 分得不好
        Otsu 就是让两类"截然不同"的那个阈值

    使用方法: cv2.threshold(..., cv2.THRESH_OTSU)
    这时候传入的阈值参数会被忽略，Otsu 会自动计算
    """
    print("\n--- 📖 演示3: Otsu 自动阈值 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Otsu 自动阈值 ---
    # 注意: THRESH_BINARY + THRESH_OTSU 用 | (按位或) 组合
    # ret 是 Otsu 自动计算出的最佳阈值
    ret_otsu, otsu_thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU  # 组合两个 flag
    )
    print(f"   Otsu 自动选择的阈值: {ret_otsu:.1f}")

    # --- 对比几个固定阈值 ---
    thresholds = [64, ret_otsu, 190]  # 低、Otsu最优、高
    comparisons = []
    for t in thresholds:
        _, result = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        res_3ch = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        label = f"{'Otsu推荐' if abs(t - ret_otsu) < 1 else ('太低' if t < ret_otsu else '太高')} = {t:.0f}"
        cv2.putText(res_3ch, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        comparisons.append((label, res_3ch))

    # 加噪声后 Otsu 的表现
    noisy = gray.copy().astype(np.float32)
    noise = np.random.normal(0, 10, gray.shape).astype(np.float32)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    ret_noisy, otsu_noisy = cv2.threshold(
        noisy, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    print(f"   加噪后 Otsu 选择的阈值: {ret_noisy:.1f}")

    gray_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    otsu_show = cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2BGR)
    noisy_show = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    otsu_noisy_show = cv2.cvtColor(otsu_noisy, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("原图", gray_show),
        ("Otsu 自动二值化", otsu_show),
        ("加噪图", noisy_show),
        ("加噪后 Otsu", otsu_noisy_show),
    ], cols=2)
    cv2.destroyAllWindows()

    print(f"   💡 Otsu 自动找到最佳阈值 {ret_otsu:.0f}——不需要手动调参！")


def demo_threshold_pipeline():
    """
    演示4: 阈值化实战——文档扫描预处理

    综合运用阈值化技术，模拟一个"手机拍文档→提取文字区域"的流程。
    """
    print("\n--- 📖 演示4: 阈值化实战 — 文档/文字提取 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 步骤1: 高斯模糊去噪 ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 步骤2: 自适应阈值提取文字效果 ---
    # 对于文档图像，自适应高斯阈值效果通常最好
    binary_adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,   # 大窗口 → 每个字内部不会中空
        3     # 常数 C：值越大越不敏感
    )

    # --- 步骤3: 形态学去噪 ---
    # 开运算去掉小的白色噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)

    # --- 步骤4: 形态学闭运算连接断开的笔画 ---
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    gray_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    adapt_show = cv2.cvtColor(binary_adaptive, cv2.COLOR_GRAY2BGR)
    cleaned_show = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    closed_show = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("1. 原图", gray_show),
        ("2. 自适应阈值", adapt_show),
        ("3. 去噪(开运算)", cleaned_show),
        ("4. 连接笔画(闭运算)", closed_show),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 文档扫描的典型流程: 模糊→自适应阈值→开运算去噪→闭运算连笔画")


# =============================================================================
# 本节总结
# =============================================================================
# 阈值化方法:
#   全局阈值:  cv2.threshold(img, thresh, maxval, type)
#              + THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC 等
#   自适应阈值: cv2.adaptiveThreshold(img, maxval, method, type, blockSize, C)
#              + 光照不均匀时的救星
#   Otsu:      cv2.threshold(img, 0, 255, THRESH_BINARY | THRESH_OTSU)
#              + 自动找最佳阈值
#
# 常见流程: 模糊去噪 → 阈值化 → 形态学后处理
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🔧 处理篇 [4/4]: 阈值化技术")
    print("="*60)

    demo_basic_threshold()       # 演示1: 基础阈值
    demo_adaptive_threshold()    # 演示2: 自适应阈值
    demo_otsu()                  # 演示3: Otsu 算法
    demo_threshold_pipeline()    # 演示4: 文档扫描实战

    print("\n✅ 处理篇 [4/4] 完成！")


if __name__ == "__main__":
    main()

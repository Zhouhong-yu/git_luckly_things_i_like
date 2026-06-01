# =============================================================================
# 🧱 基础篇 [3/3] - color_spaces.py
# 颜色空间转换：RGB/BGR、HSV、灰度、LAB 等
# =============================================================================
# 学完本模块你应该理解：
#   1. 什么是颜色空间（Color Space）
#   2. 常见颜色空间（BGR, 灰度, HSV, LAB）的特点和用途
#   3. 如何在颜色空间之间转换
#   4. 通道的分离与合并
#   5. 为什么颜色空间转换对图像处理如此重要
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 什么是颜色空间？
# =============================================================================
# 颜色空间 (Color Space) 也叫颜色模型，是用数学方式表示颜色的方法。
#
# 想象一下：你要向别人描述"红色"，你可以说：
#   - RGB 方式: "R=255, G=0, B=0"（红绿蓝三原色的比例）
#   - HSV 方式: "色调=0°, 饱和度=100%, 亮度=100%"（像调色板一样）
#   - 灰度方式: "亮度=76"（只有一个亮度值）
# 它们描述的是同一种颜色，只是"语言"不同——这就是颜色空间的意义。
#
# 常用颜色空间及应用：
# ┌──────────┬──────────────────────────────────┬─────────────────────┐
# │ 颜色空间  │ 通道含义                          │ 典型应用             │
# ├──────────┼──────────────────────────────────┼─────────────────────┤
# │ BGR      │ Blue 蓝, Green 绿, Red 红         │ OpenCV 默认格式      │
# │ RGB      │ Red 红, Green 绿, Blue 蓝         │ 显示、网页           │
# │ 灰度     │ 亮度 (0黑~255白)                  │ 简化计算、边缘检测   │
# │ HSV      │ Hue色调, Saturation饱和度, Value明度│ 颜色检测、抠图      │
# │ LAB      │ L亮度, A绿红通道, B蓝黄通道         │ 色彩校正、肤色检测   │
# │ YCrCb    │ Y亮度, Cr红色差, Cb蓝色差          │ 肤色检测、视频压缩   │
# └──────────┴──────────────────────────────────┴─────────────────────┘
# =============================================================================


def demo_bgr_to_gray():
    """
    演示1: BGR → 灰度转换

    彩色图转换为灰度图是最常用的操作之一。
    灰度图只有亮度信息，处理起来比彩色图快 3 倍（因为通道数从 3 变成 1）。

    转换公式（加权平均）:
        Gray = 0.114*B + 0.587*G + 0.299*R
        绿色权重最高，因为人眼对绿色最敏感
        红色次之，蓝色最低
    """
    print("\n--- 📖 演示1: BGR → 灰度转换 ---")
    img = load_image("sample.jpg")

    # --- 方法1: cv2.cvtColor() ---
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # COLOR_BGR2GRAY 是 OpenCV 定义的常量，表示"BGR 转 灰度"

    # --- 方法2: 手动计算（理解原理）---
    # 用 numpy 的矩阵运算实现加权平均
    b = img[:, :, 0].astype(np.float32)  # 蓝色通道，转为浮点数避免溢出
    g = img[:, :, 1].astype(np.float32)  # 绿色通道
    r = img[:, :, 2].astype(np.float32)  # 红色通道
    gray_manual = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

    # --- 方法3: 直接读为灰度图 ---
    # cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    print(f"   彩色图 shape: {img.shape}")        # (高, 宽, 3)
    print(f"   灰度图 shape: {gray1.shape}")      # (高, 宽) — 没有通道维度
    print(f"   灰度图像素值范围: [{gray1.min()}, {gray1.max()}]")

    # 展示对比效果
    # 注意：灰度图是2维的，直接拼接需要先转成3通道
    gray_3ch = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)  # 转回3通道以便拼接
    gray_manual_3ch = cv2.cvtColor(gray_manual, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("原图 (BGR)", img),
        ("cv2.cvtColor 灰度", gray_3ch),
        ("手动计算 灰度", gray_manual_3ch),
    ])
    cv2.destroyAllWindows()

    print("   💡 人眼对绿色最敏感 → 绿色权重最高 (0.587)")


def demo_bgr_to_hsv():
    """
    演示2: BGR → HSV 转换

    HSV 是图像处理中非常重要的颜色空间，因为它把"颜色"和"亮度"分开了。

    HSV 三个通道:
        H (Hue 色调):        0-180 (OpenCV 中缩放到 180)
                             表示颜色在色环上的位置:
                               0=红, 30=黄, 60=绿, 90=青, 120=蓝, 150=品红
                             注意 OpenCV 把 360° 映射到 0-180
        S (Saturation 饱和度): 0-255
                             表示颜色的鲜艳程度:
                               0=灰色（无颜色）, 255=最鲜艳
        V (Value 明度):       0-255
                             表示颜色的亮度:
                               0=黑色, 255=最亮

    为什么 HSV 很强大？
        在 BGR 中，"红色"可以是 (0,0,200) 深红、(0,50,255) 亮红、(50,0,220) 偏紫的红
        但在 HSV 中，所有红色都在 H ≈ 0 附近！只需设定 H 的范围就能选出所有红色物体。
    """
    print("\n--- 📖 演示2: BGR → HSV 转换 ---")
    img = load_image("sample.jpg")

    # --- 转换到 HSV ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 分离 HSV 三个通道 ---
    # cv2.split() 把多通道图像拆成多个单通道图像
    h, s, v = cv2.split(hsv)
    print(f"   H (色调) 通道: 范围 [{h.min()}, {h.max()}] mean={h.mean():.1f}")
    print(f"   S (饱和度) 通道: 范围 [{s.min()}, {s.max()}] mean={s.mean():.1f}")
    print(f"   V (明度) 通道: 范围 [{v.min()}, {v.max()}] mean={v.mean():.1f}")

    # 将单通道转成三通道以便可视化（用伪彩色显示）
    h_show = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
    s_show = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    v_show = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    # 在展示图上标注通道名
    cv2.putText(h_show, "H (Hue 色调)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(s_show, "S (Saturation 饱和度)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(v_show, "V (Value 明度)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    show_multiple_images([
        ("原图 (BGR)", img),
        ("H 色调通道", h_show),
        ("S 饱和度通道", s_show),
        ("V 明度通道", v_show),
    ])
    cv2.destroyAllWindows()

    print("   💡 HSV 把颜色和亮度分开 → 提取特定颜色物体超级方便")


def demo_color_detection():
    """
    演示3: 基于 HSV 的颜色检测（实战应用）

    这是 HSV 空间最经典的应用：从图像中提取特定颜色的物体。
    原理很简单：
        1. 转换到 HSV 空间
        2. 设定目标颜色的 H、S、V 范围
        3. 用 cv2.inRange() 创建掩模（mask）——范围内的像素=255(白)，范围外的=0(黑)
        4. 用掩模提取原图中的对应区域
    """
    print("\n--- 📖 演示3: 颜色检测（HSV 实战）---")
    img = load_image("sample.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 定义蓝色的 HSV 范围 ---
    # 蓝色在 HSV 色环上大约在 H=100~130 左右
    # 需要两个阈值: 下界(lower)和上界(upper)
    # 范围内的像素会被选中

    # 蓝色的范围（需要根据实际图像调整）
    lower_blue = np.array([100, 50, 50])    # H=100(蓝), S≥50(不要太灰), V≥50(不要太暗)
    upper_blue = np.array([130, 255, 255])  # H=130, S≤255, V≤255

    # 红色的范围（红色在 H 的 0 附近和 180 附近，因为色环是环形的！）
    # 所以我们分两段处理
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # --- cv2.inRange() 创建掩模 ---
    # 这个函数对每个像素检查: lower ≤ 像素值 ≤ upper
    # 如果满足条件 → 输出 255（白色）
    # 如果不满足   → 输出 0（黑色）
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 红色需要合并两段
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # 合并两个掩模

    # --- 用掩模提取原图中的颜色区域 ---
    # cv2.bitwise_and() 对每个像素做"按位与"运算
    # 掩模中白色的地方 (255) → 保留原图像素
    # 掩模中黑色的地方 (0)   → 变成黑色
    blue_region = cv2.bitwise_and(img, img, mask=mask_blue)
    red_region = cv2.bitwise_and(img, img, mask=mask_red)

    mask_blue_3ch = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)
    mask_red_3ch = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("原图", img),
        ("蓝色掩模", mask_blue_3ch),
        ("提取的蓝色区域", blue_region),
        ("红色掩模", mask_red_3ch),
        ("提取的红色区域", red_region),
    ], cols=3)
    cv2.destroyAllWindows()

    print("   💡 cv2.inRange() + bitwise_and() = 颜色检测的核心套路")


def demo_channel_split_merge():
    """
    演示4: 通道分离与合并

    cv2.split()   — 把多通道图像拆成多个单通道
    cv2.merge()   — 把多个单通道合成为一个多通道图像

    理解通道操作是理解 OpenCV 图像处理的关键。
    """
    print("\n--- 📖 演示4: 通道分离与合并 ---")
    img = load_image("sample.jpg")

    # --- 分离通道 ---
    # cv2.split() 返回一个包含三个数组的元组
    # b, g, r 都是二维数组（单通道），尺寸和原图一样
    b, g, r = cv2.split(img)

    print(f"   B 通道: shape={b.shape}, dtype={b.dtype}, mean={b.mean():.1f}")
    print(f"   G 通道: shape={g.shape}, dtype={g.dtype}, mean={g.mean():.1f}")
    print(f"   R 通道: shape={r.shape}, dtype={r.dtype}, mean={r.mean():.1f}")

    # --- 单独查看每个通道 ---
    # 单通道图像看起来是"灰度"的
    # 亮的地方表示该通道的值高（该颜色在该区域很强）

    # --- 合并通道 ---
    # cv2.merge() 把多个单通道数组合并
    merged = cv2.merge([b, g, r])  # 按 BGR 顺序合并

    # --- 有趣的操作: 通道重排 ---
    # 把通道顺序从 BGR 变成 RGB（OpenCV → matplotlib 的转换）
    rgb = cv2.merge([r, g, b])  # 把 R 放到第一通道
    # 这个图像在 OpenCV 窗口中看起来颜色会不对（因为 OpenCV 默认 BGR 显示）
    # 但我们不展示了，重要的是理解原理

    # --- 零化某个通道 ---
    # 把蓝色通道全部设为0
    b_zero = img.copy()
    b_zero[:, :, 0] = 0  # 通道0 = 蓝色

    # 把绿色通道全部设为0
    g_zero = img.copy()
    g_zero[:, :, 1] = 0  # 通道1 = 绿色

    # 把红色通道全部设为0
    r_zero = img.copy()
    r_zero[:, :, 2] = 0  # 通道2 = 红色

    b_show = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    g_show = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    r_show = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("B 通道 (亮度=蓝色强度)", b_show),
        ("G 通道 (亮度=绿色强度)", g_show),
        ("R 通道 (亮度=红色强度)", r_show),
        ("蓝色=0", b_zero),
        ("绿色=0", g_zero),
        ("红色=0", r_zero),
    ])
    cv2.destroyAllWindows()

    print("   💡 通道操作 = numpy 数组的第3维索引操作，简单但强大")


def demo_color_transfer():
    """
    演示5: 色彩空间的高级应用

    展示 LAB 和 YCrCb 颜色空间，以及它们在实际中的应用。
    """
    print("\n--- 📖 演示5: 其他颜色空间 (LAB, YCrCb) ---")
    img = load_image("sample.jpg")

    # --- LAB 颜色空间 ---
    # L: 亮度 (Lightness), 0=黑 100=白（OpenCV 中缩放到 0-255）
    # A: 绿色到红色 (-128到127，OpenCV 中偏移到 0-255）
    # B: 蓝色到黄色
    # LAB 的空间感知均匀——数值变化 = 人眼感知的变化
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # --- YCrCb 颜色空间 ---
    # Y:  亮度
    # Cr: 红色色差分量
    # Cb: 蓝色色差分量
    # 广泛应用于视频压缩（MPEG、JPEG 都用它）和肤色检测
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

    l_show = cv2.cvtColor(l_ch, cv2.COLOR_GRAY2BGR)
    a_show = cv2.cvtColor(a_ch, cv2.COLOR_GRAY2BGR)
    b_lab_show = cv2.cvtColor(b_ch, cv2.COLOR_GRAY2BGR)
    cr_show = cv2.cvtColor(cr_ch, cv2.COLOR_GRAY2BGR)
    cb_show = cv2.cvtColor(cb_ch, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("LAB L (亮度)", l_show),
        ("LAB A (绿→红)", a_show),
        ("LAB B (蓝→黄)", b_lab_show),
        ("YCrCb Cr (红色差)", cr_show),
        ("YCrCb Cb (蓝色差)", cb_show),
    ], cols=3)
    cv2.destroyAllWindows()

    print("   💡 LAB 感知均匀 → 适合色彩校正；YCrCb → 肤色检测 + 视频压缩")


# =============================================================================
# 本节总结
# =============================================================================
# 常用转换:
#   BGR → 灰度:   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   BGR → HSV:    cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#   BGR → LAB:    cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#   BGR → RGB:    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # matplotlib 必备
#
# 通道操作:
#   分离: b, g, r = cv2.split(img)  → 三个单通道数组
#   合并: merged = cv2.merge([b, g, r])
#
# 颜色检测流程:
#   1. BGR → HSV
#   2. mask = cv2.inRange(hsv, lower, upper)  → 创建掩模
#   3. result = cv2.bitwise_and(img, img, mask=mask)  → 提取目标区域
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🧱 基础篇 [3/3]: 颜色空间转换")
    print("="*60)

    demo_bgr_to_gray()          # 演示1: 灰度转换
    demo_bgr_to_hsv()           # 演示2: HSV 转换
    demo_color_detection()      # 演示3: 颜色检测
    demo_channel_split_merge()  # 演示4: 通道分离合并
    demo_color_transfer()       # 演示5: 其他颜色空间

    print("\n✅ 基础篇 [3/3] 完成！")


if __name__ == "__main__":
    main()

# =============================================================================
# 🔍 分析篇 [3/3] - feature_detection.py
# 特征检测与匹配：角点检测、SIFT、ORB、特征匹配
# =============================================================================
# 学完本模块你应该理解：
#   1. 什么是"图像特征"——图像中有辨识度的点
#   2. Harris 角点检测的原理
#   3. SIFT (尺度不变特征变换) 的思想
#   4. ORB (快速特征检测) —— 免费的 SIFT 替代品
#   5. 特征匹配——找到两张图中相同的关键点
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 什么是图像特征？
# =============================================================================
# 图像特征 (Feature) = 图像中"与众不同"的点，通常包括：
#   - 角点 (Corner): 两个边缘交汇的地方（如桌角、墙转角）
#   - 斑点 (Blob):   一块颜色/亮度与众不同的区域
#
# 好的特征应该具备:
#   1. 可重复性: 同一物体从不同角度拍，特征点应该依然能被检测到
#   2. 独特性:   不同特征点之间的描述应该很不相同（方便匹配）
#   3. 局部性:   特征只占图像一小块区域（部分遮挡也不影响其他特征）
#
# 特征检测的典型流程:
#   1. 检测: 找到图像中的关键点 (Keypoint)
#   2. 描述: 为每个关键点生成一个"描述子" (Descriptor)
#   3. 匹配: 在不同图像中找到描述子相似的关键点——它们可能是同一个物体
# =============================================================================


def demo_harris_corner():
    """
    演示1: Harris 角点检测

    Harris 是最经典的角点检测算法（1988年），虽然古老但概念清晰。

    核心思想: 用一个窗口在图像上滑动，观察窗口内像素的变化
        - 平坦区域: 窗口向任何方向移动，像素变化都很小
        - 边缘:     窗口沿边缘方向移动变化小，垂直方向变化大
        - 角点:     窗口向任何方向移动，像素变化都很大！

    数学上通过计算"结构张量"的特征值来判断:
        - λ1≈λ2≈0: 平坦区域
        - λ1≫λ2≈0: 边缘
        - λ1≈λ2≫0: 角点
    """
    print("\n--- 📖 演示1: Harris 角点检测 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f32 = np.float32(gray)  # Harris 需要 float32 输入

    # --- cv2.cornerHarris() ---
    # cv2.cornerHarris(src, blockSize, ksize, k)
    #   blockSize: 计算梯度的邻域大小
    #   ksize:     Sobel 核的大小
    #   k:         Harris 检测器的自由参数（经验值 0.04-0.06）
    harris = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)

    # harris 是一个和原图同样大小的数组，角点处值较大
    # 膨胀一下让角点更明显
    harris_dilated = cv2.dilate(harris, None)

    # 在原图上标记角点
    img_harris = img.copy()
    # 角点处的值 > 最大值的 1%，阈值可根据需要调整
    threshold = 0.01 * harris.max()
    img_harris[harris > threshold] = [0, 0, 255]  # 红色标记

    # --- cv2.goodFeaturesToTrack() ---
    # 这是 Harris + Shi-Tomasi 的改进版，直接返回"最好"的N个角点
    # 比手动设置阈值更方便
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=50, qualityLevel=0.01, minDistance=10
    )
    # maxCorners:   最多返回多少个角点
    # qualityLevel: 质量阈值（0.01 = 取最佳的前1%）
    # minDistance:  角点之间的最小距离（避免太密集）

    img_gftt = img.copy()
    if corners is not None:
        corners = np.int32(corners)  # 转为整数坐标
        for corner in corners:
            x, y = corner.ravel()  # ravel() 展平数组
            cv2.circle(img_gftt, (x, y), 5, (0, 255, 0), -1)

    print(f"   goodFeaturesToTrack 检测到 {len(corners) if corners is not None else 0} 个角点")

    show_multiple_images([
        ("原图", img),
        ("Harris 响应图", cv2.cvtColor((harris * 10).clip(0, 255).astype(np.uint8),
                                      cv2.COLOR_GRAY2BGR)),
        ("Harris 角点 (红色)", img_harris),
        ("goodFeaturesToTrack (绿色)", img_gftt),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 角点 = 窗口往任何方向移动像素都剧烈变化的位置")


def demo_sift():
    """
    演示2: SIFT —— 尺度不变特征变换

    SIFT (Scale-Invariant Feature Transform) 是计算机视觉史上最重要的算法之一，
    由 David Lowe 在 1999 年提出，2004 年完善。

    SIFT 的优势:
        1. 尺度不变: 不管物体在图中是大是小，都能检测到
        2. 旋转不变: 不管物体怎么旋转，描述子都不变
        3. 光照部分不变: 对亮度变化有一定鲁棒性
        4. 描述子信息丰富: 每个特征有一个 128 维的向量描述

    ⚠️ SIFT 有专利保护，在 opencv-python 中可能在 cv2.SIFT_create()
       如果不可用，会自动回退到 ORB
    """
    print("\n--- 📖 演示2: SIFT 特征检测 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 创建 SIFT 检测器 ---
    try:
        sift = cv2.SIFT_create()
        sift_available = True
        print("   ✅ 使用 SIFT 检测器")
    except AttributeError:
        # 如果 opencv-python 没有 SIFT（可能需要 opencv-contrib-python）
        print("   ⚠️ SIFT 不可用，使用 ORB 代替")
        print("     如需 SIFT，请安装: pip install opencv-contrib-python")
        sift_available = False
        return

    # --- 检测关键点并计算描述子 ---
    # sift.detectAndCompute(img, mask) 一步完成检测和描述
    #   返回值: (关键点列表, 描述子数组)
    #   描述子形状: (关键点数, 128) —— 每个关键点一个128维向量
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    print(f"   检测到 {len(keypoints)} 个 SIFT 关键点")
    if descriptors is not None:
        print(f"   描述子形状: {descriptors.shape}")  # (N, 128)

    # --- 绘制关键点 ---
    # cv2.drawKeypoints() 专门用来画关键点
    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS:
    #   画出关键点的大小(圆圈)和方向(半径线)
    img_sift = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )

    show_multiple_images([
        ("原图", img),
        (f"SIFT 关键点 ({len(keypoints)}个)", img_sift),
    ])
    cv2.destroyAllWindows()

    print("   💡 SIFT = 尺度不变 + 旋转不变 + 128维描述子 = 特征匹配的黄金标准")


def demo_orb():
    """
    演示3: ORB —— 快速高效的特征检测

    ORB (Oriented FAST and Rotated BRIEF) 是 SIFT 的免费替代品，
    由 OpenCV 实验室在 2011 年提出。

    ORB = FAST (角点检测) + BRIEF (二值描述子) + 方向 + 尺度
        FAST:  非常快的角点检测器（比 Harris 快一个数量级）
        BRIEF: 用二值串(0和1)做描述子，比浮点向量匹配快很多

    对比:
        SIFT:  高精度，慢，有专利 → 学术研究、离线处理
        ORB:   略低精度，非常快，免费 → 实时应用、SLAM、手机APP
    """
    print("\n--- 📖 演示3: ORB 特征检测 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 创建 ORB 检测器 ---
    # cv2.ORB_create(nfeatures=..., scaleFactor=..., nlevels=...)
    #   nfeatures:   最多保留多少个特征点
    #   scaleFactor: 图像金字塔的缩放因子（尺度不变性）
    #   nlevels:     金字塔层数
    orb = cv2.ORB_create(nfeatures=100)

    # --- 检测和描述 ---
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    print(f"   检测到 {len(keypoints)} 个 ORB 关键点")
    if descriptors is not None:
        print(f"   描述子形状: {descriptors.shape}")  # (N, 32) —— 比SIFT的128少很多
        print(f"   描述子类型: {descriptors.dtype}")   # uint8 —— 二值描述子

    # 绘制
    img_orb = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    show_multiple_images([
        ("原图", img),
        (f"ORB 关键点 ({len(keypoints)}个)", img_orb),
    ])
    cv2.destroyAllWindows()

    print("   💡 ORB = FAST(快速角点) + BRIEF(二值描述子) = 免费+SIFT级速度")


def demo_feature_matching():
    """
    演示4: 特征匹配——找两张图中相同的特征点

    这是 SIFT/ORB 最重要的应用：给定两张图像，找到它们之间的对应关系。
    这是 3D 重建、全景拼接、物体跟踪的核心步骤。

    匹配方法:
        - 暴力匹配 (Brute-Force): 对每个特征，遍历另一张图的所有特征找最相似的
        - FLANN: 基于 KD 树的快速近似最近邻搜索（适合大量特征）

    匹配好坏过滤:
        - Lowe's ratio test: 最近邻距离 / 次近邻距离 < 阈值 → 认为是好的匹配
          原理: 好的匹配应该显著优于其他所有匹配
    """
    print("\n--- 📖 演示4: 特征匹配 ---")
    img1 = load_image("sample.jpg")

    # 创建"第二张图"——对原图做一些变换
    # 在实际应用中，这可能是从不同角度拍的同一场景
    h, w = img1.shape[:2]
    # 旋转 + 缩放 + 裁剪模拟"不同视角"
    img2 = cv2.resize(img1, (int(w * 0.8), int(h * 0.8)))
    # 旋转15度
    import math
    center = (img2.shape[1] // 2, img2.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 15, 1.0)
    img2 = cv2.warpAffine(img2, rot_mat, (img2.shape[1], img2.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # --- 使用 ORB 进行特征检测和匹配 ---
    orb = cv2.ORB_create(nfeatures=100)

    # 检测两张图的特征
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        print("   ⚠️ 特征点不足，跳过匹配")
        return

    # --- 暴力匹配器 ---
    # cv2.BFMatcher(normType, crossCheck)
    #   normType: 距离度量方式
    #     NORM_HAMMING: 汉明距离（适用于 ORB 的二值描述子）
    #     NORM_L2:      欧氏距离（适用于 SIFT 的浮点描述子）
    #   crossCheck=True: 双向验证（A匹配到B，且B也匹配到A）
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # knnMatch: 对每个查询特征，找 k 个最佳匹配
    # 然后用 Lowe's ratio test 过滤
    matches = bf.knnMatch(des1, des2, k=2)

    # --- Lowe's ratio test ---
    # 最近匹配的距离 / 次近匹配的距离 < 阈值(通常0.75)
    good_matches = []
    for match in matches:
        if len(match) >= 2:
            m, n = match[0], match[1]
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print(f"   总匹配数: {len(matches)}")
    print(f"   Lowe's ratio test 后: {len(good_matches)} 个好的匹配")

    # --- 绘制匹配结果 ---
    # cv2.drawMatches() 将两张图并排显示，连线表示匹配
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches[:30], None,  # 最多显示30个匹配
        matchColor=(0, 255, 0),       # 匹配线颜色
        singlePointColor=(255, 0, 0), # 单个点的颜色
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    show_image(f"特征匹配 (好的匹配: {len(good_matches)})", img_matches)
    cv2.destroyAllWindows()

    print("   💡 Lowe's ratio test = 好的匹配不仅是最近，还要显著优于其他所有匹配")


# =============================================================================
# 本节总结
# =============================================================================
# 角点检测:
#   cv2.cornerHarris()        — Harris 经典角点检测
#   cv2.goodFeaturesToTrack() — 改进版，直接返回最好的N个角点
#
# 特征检测与描述:
#   SIFT: cv2.SIFT_create()   — 尺度不变 + 旋转不变，精度高但慢（有专利）
#   ORB:  cv2.ORB_create()    — 快速 + 免费，实时应用首选
#
# 特征匹配:
#   BFMatcher() + knnMatch()  — 暴力匹配 + Lowe's ratio test
#   drawMatches()             — 可视化匹配结果
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🔍 分析篇 [3/3]: 特征检测与匹配")
    print("="*60)

    demo_harris_corner()     # 演示1: Harris 角点
    demo_sift()              # 演示2: SIFT
    demo_orb()               # 演示3: ORB
    demo_feature_matching()  # 演示4: 特征匹配

    print("\n✅ 分析篇 [3/3] 完成！")


if __name__ == "__main__":
    main()

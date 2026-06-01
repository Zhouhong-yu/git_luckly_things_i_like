# =============================================================================
# 🏆 应用篇 [3/3] - image_stitching.py
# 图像拼接（全景图合成）
# =============================================================================
# 学完本模块你应该理解：
#   1. 图像拼接的完整流程
#   2. 特征检测 + 匹配 = 找到两张图的重叠区域
#   3. 单应性矩阵 (Homography) = 描述透视变换的 3×3 矩阵
#   4. RANSAC 算法 = 在大量匹配中排除错误匹配
#   5. 图像融合 = 无缝拼接两张图
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 图像拼接
# =============================================================================
# 图像拼接 (Image Stitching) = 把多张有重叠区域的照片合成为一张全景图。
#
# 你手机上的"全景模式"就是用的这个技术！
#
# 拼接流程:
# ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
# │ 1. 特征  │ → │ 2. 特征  │ → │ 3. 计算   │ → │ 4. 扭曲  │ → │ 5. 融合  │
# │   检测   │    │   匹配   │    │  单应性   │    │   变换   │    │   拼接   │
# └──────────┘    └──────────┘    └───────────┘    └──────────┘    └──────────┘
#
# 步骤1: 用 SIFT/ORB 检测两张图的关键点和描述子
# 步骤2: 匹配描述子——找到两张图"看的是同一个东西"的关键点对
# 步骤3: 用匹配点计算单应性矩阵 H（3×3 透视变换矩阵）
# 步骤4: 用 H 把第二张图"扭曲"到第一张图的视角
# 步骤5: 把两张图"融合"在一起，处理重叠区域的过渡
#
# 关键数学: 单应性矩阵 (Homography)
#   [x']   [h11 h12 h13] [x]
#   [y'] = [h21 h22 h23] [y]
#   [1 ]   [h31 h32 h33] [1]
#   它描述了一个平面到另一个平面的透视映射
#   最少需要 4 对匹配点来计算（8个自由度）
# =============================================================================


def demo_homography():
    """
    演示1: 单应性矩阵——图像拼接的基础

    在拼接之前，先理解什么是单应性矩阵。
    它能够把一张图的"视角"变换成另一张图的视角。

    单应性假设: 两张图拍摄的是同一个平面（或者拍摄者位置没变，只是旋转了镜头）
    """
    print("\n--- 📖 演示1: 理解单应性矩阵 ---")
    img = load_image("sample.jpg")
    h, w = img.shape[:2]

    # --- 定义源图像的四个角 ---
    # 假设我们想把一张"拍歪了"的图变成正面的
    src_pts = np.float32([
        [100, 100],          # 左上
        [w - 200, 80],       # 右上
        [w - 150, h - 180],  # 右下
        [50, h - 70],        # 左下
    ])

    # --- 目标：把源点映射到一个端正的矩形 ---
    dst_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

    # --- 计算单应性矩阵 ---
    # cv2.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold)
    #   method: 计算方法
    #     0: 常规方法（需要至少4对点）
    #     cv2.RANSAC: 使用 RANSAC 排除异常点（推荐!）
    #     cv2.LMEDS:  使用最小中值法
    #   ransacReprojThreshold: RANSAC 的重投影误差阈值
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    print(f"   单应性矩阵 H (3×3):")
    print(f"   {H[0]}")
    print(f"   {H[1]}")
    print(f"   {H[2]}")
    print(f"   矩阵秩: {np.linalg.matrix_rank(H)}")

    # --- 应用单应性变换 ---
    warped = cv2.warpPerspective(img, H, (w, h))

    # --- 可视化 ---
    img_with_pts = img.copy()
    for i, pt in enumerate(src_pts.astype(int)):
        cv2.circle(img_with_pts, tuple(pt), 10, (0, 0, 255), -1)
        # 用箭头标注顺序
        labels = ["左上", "右上", "右下", "左下"]
        cv2.putText(img_with_pts, labels[i], tuple(pt + [15, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.polylines(img_with_pts, [src_pts.astype(int)], True, (0, 255, 0), 2)

    show_multiple_images([
        ("原图 + 选中四边形", img_with_pts),
        ("单应性变换后 (透视校正)", warped),
    ])
    cv2.destroyAllWindows()

    print("   💡 单应性 = 把一张图的平面'投影'到另一张图的平面上")


def demo_manual_stitch():
    """
    演示2: 手动实现图像拼接（逐步拆解）

    不使用 OpenCV 的 Stitcher API，而是手动实现拼接的每个步骤。
    这样可以深入理解拼接的原理。
    """
    print("\n--- 📖 演示2: 手动图像拼接流程 ---")
    img = load_image("sample.jpg")
    h, w = img.shape[:2]

    # --- 模拟"两张有重叠区域的照片"---
    # 把原图分成左右两半，右半张模拟"第二张照片"
    # 两张图之间有 100 像素的重叠区域
    overlap = 150  # 重叠区域宽度

    img_left = img[:, :w//2 + overlap]  # 左半 + 重叠
    img_right = img[:, w//2 - overlap:]  # 重叠 + 右半

    # 对右图做一点轻微旋转，模拟"手持拍摄的角度变化"
    center = (img_right.shape[1] // 2, img_right.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 5, 1.0)
    img_right = cv2.warpAffine(img_right, rot_mat,
                               (img_right.shape[1], img_right.shape[0]))

    print(f"   左图尺寸: {img_left.shape[1]}x{img_left.shape[0]}")
    print(f"   右图尺寸: {img_right.shape[1]}x{img_right.shape[0]}")
    print(f"   重叠区域: {overlap} 像素")

    # --- 步骤1: 检测特征 ---
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 使用 SIFT (如果有的话，否则用 ORB)
    try:
        detector = cv2.SIFT_create(nfeatures=200)
    except (AttributeError, cv2.error):
        detector = cv2.ORB_create(nfeatures=200)

    kp_left, des_left = detector.detectAndCompute(gray_left, None)
    kp_right, des_right = detector.detectAndCompute(gray_right, None)

    print(f"   左图特征点: {len(kp_left)}, 右图特征点: {len(kp_right)}")

    if des_left is None or des_right is None:
        print("   ❌ 特征检测失败，跳过")
        return

    # --- 步骤2: 特征匹配 ---
    # 使用暴力匹配器
    if hasattr(detector, 'descriptorSize'):
        # SIFT → 用 L2 距离
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        # ORB → 用汉明距离
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(des_left, des_right, k=2)

    # Lowe's ratio test: 好的匹配应该是"明显的"，而不是模糊的
    good_matches = []
    for match in matches:
        if len(match) >= 2:
            m, n = match[0], match[1]
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print(f"   Lowe's test 后好的匹配: {len(good_matches)}")

    if len(good_matches) < 4:
        print("   ❌ 好的匹配点太少（需要至少4对），无法计算单应性")
        return

    # --- 步骤3: 计算单应性矩阵 ---
    src_pts = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

    # 使用 RANSAC 排除错误匹配
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("   ❌ 无法计算单应性矩阵")
        return

    inliers = mask.ravel().tolist().count(1)
    print(f"   RANSAC 内点(正确匹配): {inliers}/{len(good_matches)}")
    print(f"   单应性矩阵:\n{H}")

    # --- 步骤4: 扭曲右图并拼接 ---
    # 计算拼接后画布的大小
    # 把右图的四个角投影到左图的坐标系中
    h_r, w_r = img_right.shape[:2]
    corners = np.float32([
        [0, 0], [w_r, 0], [w_r, h_r], [0, h_r]
    ]).reshape(-1, 1, 2)
    # cv2.perspectiveTransform() 对点集应用透视变换
    transformed_corners = cv2.perspectiveTransform(corners, H)

    # 计算合并后的画布范围
    all_corners = np.concatenate([
        np.float32([[0, 0], [img_left.shape[1], 0],
                     [img_left.shape[1], img_left.shape[0]],
                     [0, img_left.shape[0]]]).reshape(-1, 1, 2),
        transformed_corners
    ])

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 平移矩阵：把整个画布移到正坐标区域
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float32)

    # 把右图变形到左图的坐标系中
    warped_right = cv2.warpPerspective(
        img_right,
        translation @ H,  # 先单应性变换，再平移到正坐标
        (x_max - x_min, y_max - y_min)
    )

    # 把左图放到画布上
    canvas = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    canvas[-y_min:-y_min + img_left.shape[0],
           -x_min:-x_min + img_left.shape[1]] = img_left

    # --- 步骤5: 图像融合 ---
    # 简单方法: 对于重叠区域，取两图的平均值
    mask_left = np.zeros(canvas.shape[:2], dtype=np.float32)
    mask_left[-y_min:-y_min + img_left.shape[0],
              -x_min:-x_min + img_left.shape[1]] = 1

    mask_right = (warped_right.sum(axis=2) > 0).astype(np.float32)

    # 重叠区域: 两张图的 mask 都为1的地方
    overlap_mask = mask_left * mask_right

    # 简单融合: 在重叠区域取平均
    result = canvas.copy()
    for c in range(3):
        overlap_area = overlap_mask > 0
        result[:, :, c][overlap_area] = (
            canvas[:, :, c][overlap_area].astype(np.float32) * 0.5 +
            warped_right[:, :, c][overlap_area].astype(np.float32) * 0.5
        ).astype(np.uint8)

    # 把右图非重叠部分加到结果中
    non_overlap = (mask_right > 0) & (overlap_mask == 0)
    result[non_overlap] = warped_right[non_overlap]

    # --- 步骤6: 裁剪黑边 ---
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w_crop, h_crop = cv2.boundingRect(coords)
        result_cropped = result[y:y+h_crop, x:x+w_crop]
    else:
        result_cropped = result

    # --- 显示结果 ---
    matches_img = cv2.drawMatches(
        img_left, kp_left, img_right, kp_right,
        good_matches[:30], None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    show_multiple_images([
        ("左图", img_left),
        ("右图", img_right),
    ])
    cv2.destroyAllWindows()

    show_image(f"特征匹配 ({len(good_matches)} 个好的匹配, {inliers} 内点)",
               cv2.resize(matches_img, (1200, 400)))
    cv2.destroyAllWindows()

    show_image(f"拼接结果 (手动实现)", result_cropped)
    cv2.destroyAllWindows()

    print("   💡 这就是图像拼接的全流程——你手机的全景模式也是这样工作的！")


def demo_stitcher_api():
    """
    演示3: OpenCV Stitcher API——一键拼接

    OpenCV 提供了高级的 Stitcher 类，封装了所有步骤。

    使用很简单:
        1. 创建 Stitcher 对象
        2. 传入图像列表
        3. 自动完成拼接
    """
    print("\n--- 📖 演示3: OpenCV Stitcher API ---")
    img = load_image("sample.jpg")
    h, w = img.shape[:2]

    # 创建两张"有重叠的照片"
    overlap = 150
    img_left = img[:, :w//2 + overlap]
    img_right = img[:, w//2 - overlap:]

    # 给右图加一点旋转，模拟真实拍摄条件
    center = (img_right.shape[1] // 2, img_right.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 3, 1.0)
    img_right = cv2.warpAffine(img_right, rot_mat,
                               (img_right.shape[1], img_right.shape[0]))

    # --- 创建 Stitcher ---
    # cv2.Stitcher_create(mode)
    #   mode: cv2.Stitcher_PANORAMA (=0, 全景) 或 cv2.Stitcher_SCANS (=1, 扫描文档)
    try:
        # OpenCV 4.x
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except (AttributeError, NameError):
        try:
            # OpenCV 4.x (某些版本的常量名可能不同)
            stitcher = cv2.Stitcher_create(0)  # 0 = PANORAMA
        except AttributeError:
            # OpenCV 3.x 的兼容写法
            stitcher = cv2.createStitcher(False)
            if stitcher is None:
                print("   ❌ 无法创建 Stitcher，跳过")
                return

    # --- 拼接 ---
    # stitcher.stitch(images) 返回 (status, result)
    #   status: 0=OK, 1=需要更多图片, 2=单应性估计失败, 3=相机参数调整失败
    status, stitched = stitcher.stitch([img_left, img_right])

    # --- 显示结果 ---
    show_multiple_images([
        ("左图", img_left),
        ("右图 (旋转3°)", img_right),
    ])
    cv2.destroyAllWindows()

    if status == 0:  # cv2.Stitcher_OK — 拼接成功
        # 裁剪黑边
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w_c, h_c = cv2.boundingRect(coords)
            stitched = stitched[y:y+h_c, x:x+w_c]

        show_image("OpenCV Stitcher API 拼接结果 ✅", stitched)
        cv2.destroyAllWindows()
        print("   ✅ 拼接成功！")
    else:
        status_msgs = {
            1: "需要更多图片",
            2: "单应性矩阵估计失败",
            3: "相机参数调整失败",
        }
        msg = status_msgs.get(status, f"未知错误 (status={status})")
        print(f"   ❌ 拼接失败: {msg}")


# =============================================================================
# 本节总结
# =============================================================================
# 图像拼接流程:
#   1. 特征检测:  SIFT/ORB → keypoints + descriptors
#   2. 特征匹配:  BFMatcher + knnMatch + Lowe's ratio test
#   3. 单应性矩阵: cv2.findHomography(src, dst, cv2.RANSAC)
#   4. 透视变换:  cv2.perspectiveTransform() + cv2.warpPerspective()
#   5. 融合:      重叠区域取平均 / 多频段融合 / 图割优化
#
# 一键拼接: cv2.Stitcher_create().stitch([img1, img2, ...])
#
# 关键数学:
#   单应性矩阵 H (3×3): 把一张图的像素映射到另一张图的坐标系
#   RANSAC:           排除错误匹配——拼接质量的保证
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🏆 应用篇 [3/3]: 图像拼接（全景图）")
    print("="*60)

    demo_homography()      # 演示1: 理解单应性
    demo_manual_stitch()   # 演示2: 手动实现拼接
    demo_stitcher_api()    # 演示3: Stitcher API

    print("\n✅ 应用篇 [3/3] 完成！")


if __name__ == "__main__":
    main()

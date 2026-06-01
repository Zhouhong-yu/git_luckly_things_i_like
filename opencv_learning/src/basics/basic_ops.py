# =============================================================================
# 🧱 基础篇 [2/3] - basic_ops.py
# 基本图像操作：裁剪、缩放、旋转、翻转、平移
# =============================================================================
# 学完本模块你应该理解：
#   1. 如何裁剪图像（ROI = Region of Interest = 感兴趣区域）
#   2. 缩放图像的多种插值方法
#   3. 用仿射变换实现旋转和平移
#   4. 图像的几何变换背后的数学原理
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 图像的几何变换
# =============================================================================
# 图像的基本操作分为两类:
#
# 1. 像素级操作（改变像素值）:
#    - 裁剪 (crop): 取图像的一部分
#    - 亮度调整: 所有像素值加/减一个常数
#
# 2. 几何变换（改变像素位置）:
#    - 缩放 (resize): 改变图像尺寸
#    - 旋转 (rotate): 绕某点旋转
#    - 翻转 (flip): 镜像翻转
#    - 平移 (translate): 移动图像位置
#
# 几何变换的数学基础: 仿射变换矩阵 (Affine Transformation Matrix)
#   它是一个 2×3 的矩阵，描述了如何把原图的坐标映射到新图:
#   [x']   [a b c] [x]
#   [y'] = [d e f] [y]
#                       其中 (x,y)是原坐标，(x',y')是新坐标
# =============================================================================


def demo_crop():
    """
    演示1: 图像裁剪 (Crop / ROI)

    裁剪就是取图像的一个矩形子区域。
    这是最简单的操作，因为图像是 numpy 数组，直接用切片即可。

    应用场景:
        - 截取人脸区域
        - 去除不需要的边框
        - 从大图中提取目标物体
    """
    print("\n--- 📖 演示1: 图像裁剪 (ROI) ---")
    img = load_image("sample.jpg")
    h, w = img.shape[:2]

    # --- 数组切片语法回顾 ---
    # img[起始行:结束行, 起始列:结束列]
    # Python 切片是"左闭右开"区间: [start, end)
    # 即包含 start，不包含 end

    # --- 裁剪中心区域 ---
    # 取图像正中间 50% 的区域
    crop_center = img[h//4 : 3*h//4, w//4 : 3*w//4]
    print(f"   原图尺寸: {w}x{h}")
    print(f"   裁剪后尺寸: {crop_center.shape[1]}x{crop_center.shape[0]}")

    # --- 裁剪其他区域 ---
    crop_top_left = img[0:h//2, 0:w//2]     # 左上角 1/4
    crop_bot_right = img[h//2:h, w//2:w]     # 右下角 1/4

    show_multiple_images([
        ("原图", img),
        ("中心裁剪", crop_center),
        ("左上裁剪", crop_top_left),
        ("右下裁剪", crop_bot_right),
    ])
    cv2.destroyAllWindows()

    print("   💡 裁剪 = numpy 切片，零计算开销（只是"视图"，不复制数据）")


def demo_resize():
    """
    演示2: 图像缩放 (Resize)

    cv2.resize() 改变图像尺寸。
    放大时需要在原始像素之间"插值"（猜测缺失的像素值），这就是插值算法的作用。

    插值方法对比:
        - INTER_NEAREST: 最近邻——取最近的像素值，最快但锯齿严重
        - INTER_LINEAR:  双线性——取周围4个像素的加权平均，速度和效果均衡（默认）
        - INTER_CUBIC:   双三次——取周围16个像素，效果最好但较慢
        - INTER_AREA:    区域重采样——缩小时效果最好，防止摩尔纹
    """
    print("\n--- 📖 演示2: 图像缩放 ---")
    img = load_image("sample.jpg")
    h, w = img.shape[:2]
    print(f"   原图尺寸: {w}x{h}")

    # --- 方法1: 指定目标尺寸（宽, 高）---
    # 注意 cv2.resize() 的参数顺序是 (宽, 高)，和 shape 的 (高, 宽) 相反！
    resized_exact = cv2.resize(img, (320, 240))
    print(f"   缩放到指定尺寸 320x240: shape={resized_exact.shape}")

    # --- 方法2: 按比例缩放 ---
    # 指定 fx（水平缩放因子）和 fy（垂直缩放因子）
    # 当 dsize=None 时，根据 fx, fy 计算新尺寸
    # 新宽度 = 原宽度 × fx
    scaled_half = cv2.resize(img, None, fx=0.5, fy=0.5)          # 缩小一半
    scaled_double = cv2.resize(img, None, fx=2.0, fy=2.0)         # 放大两倍
    print(f"   缩小50%: shape={scaled_half.shape}")
    print(f"   放大200%: shape={scaled_double.shape}")

    # --- 方法3: 对比不同插值方法 ---
    # 先缩小再放大，让插值方法的效果差异更明显
    small = cv2.resize(img, (80, 60))  # 先缩到很小

    # 然后用不同的插值方法放大回来
    near  = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)  # 马赛克效果
    linear = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)  # 稍微模糊
    cubic  = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)   # 最平滑

    show_multiple_images([
        ("最近邻 INTER_NEAREST", near),
        ("双线性 INTER_LINEAR (默认)", linear),
        ("双三次 INTER_CUBIC", cubic),
    ])
    cv2.destroyAllWindows()

    print("   💡 缩小用 INTER_AREA，放大用 INTER_CUBIC，默认 INTER_LINEAR 够用")


def demo_flip():
    """
    演示3: 图像翻转 (Flip)

    cv2.flip() 对图像进行镜像翻转。

    翻转方向:
        - flipCode > 0: 水平翻转（左右镜像）
        - flipCode = 0: 垂直翻转（上下颠倒）
        - flipCode < 0: 同时水平和垂直翻转（相当于旋转180°）
    """
    print("\n--- 📖 演示3: 图像翻转 ---")
    img = load_image("sample.jpg")

    flip_h = cv2.flip(img, 1)   # 水平翻转：左边变右边
    flip_v = cv2.flip(img, 0)   # 垂直翻转：上边变下边
    flip_b = cv2.flip(img, -1)  # 两个方向都翻转

    show_multiple_images([
        ("原图", img),
        ("水平翻转 (flip=1)", flip_h),
        ("垂直翻转 (flip=0)", flip_v),
        ("双方向翻转 (flip=-1)", flip_b),
    ])
    cv2.destroyAllWindows()


def demo_rotate():
    """
    演示4: 图像旋转 (Rotate)

    旋转是"仿射变换"的一种。核心概念：

    仿射变换矩阵: 2×3 的矩阵，用来描述"如何把原图像素映射到新图"
    旋转矩阵:     cv2.getRotationMatrix2D(旋转中心, 角度, 缩放因子)
                 返回一个 2×3 的矩阵

    应用场景:
        - 校正倾斜的文档/照片
        - 数据增强（训练 AI 模型时扩充数据集）
    """
    print("\n--- 📖 演示4: 图像旋转 ---")
    img = load_image("sample.jpg")
    h, w = img.shape[:2]

    # --- 基础旋转: cv2.rotate() ---
    # 只支持 90° 的倍数，速度最快
    rot_90  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)         # 顺时针90°
    rot_180 = cv2.rotate(img, cv2.ROTATE_180)                   # 旋转180°
    rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)   # 逆时针90°

    # --- 任意角度旋转: cv2.getRotationMatrix2D() + cv2.warpAffine() ---
    # 步骤1: 创建旋转矩阵
    # cv2.getRotationMatrix2D(旋转中心(x,y), 旋转角度(逆时针为正), 缩放因子)
    center = (w // 2, h // 2)  # 旋转中心 = 图像中心
    angle = 45                 # 旋转角度（逆时针）
    scale = 1.0                # 缩放因子（1.0 = 保持原大小）
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # rotation_matrix 是一个 2×3 的 numpy 数组
    print(f"   旋转矩阵 (2×3):\n{rotation_matrix}")

    # 步骤2: 应用仿射变换
    # cv2.warpAffine(原图, 变换矩阵, 输出尺寸)
    # 它会根据变换矩阵对每个像素重新计算位置
    rot_45 = cv2.warpAffine(img, rotation_matrix, (w, h))

    show_multiple_images([
        ("90°旋转", rot_90),
        ("180°旋转", rot_180),
        ("270°旋转", rot_270),
        ("45°旋转", rot_45),
    ])
    cv2.destroyAllWindows()

    print("   💡 90°倍数用 cv2.rotate()，任意角度用 getRotationMatrix2D() + warpAffine()")


def demo_translate():
    """
    演示5: 图像平移 (Translate)

    平移也是仿射变换的一种：每个像素在 x 和 y 方向移动固定距离。

    平移矩阵:
        [1  0  tx]     x' = x + tx
        [0  1  ty]     y' = y + ty
    """
    print("\n--- 📖 演示5: 图像平移 ---")
    img = load_image("sample.jpg")
    h, w = img.shape[:2]

    # --- 手动创建平移矩阵 ---
    # np.float32 指定数据类型为 32 位浮点数（OpenCV 的矩阵运算需要这种精度）
    tx, ty = 100, 50  # x 方向平移 100 像素，y 方向平移 50 像素
    translation_matrix = np.float32([
        [1, 0, tx],  # x' = 1*x + 0*y + tx
        [0, 1, ty]   # y' = 0*x + 1*y + ty
    ])

    # --- 应用平移 ---
    translated = cv2.warpAffine(img, translation_matrix, (w, h))

    # --- 更复杂的例子: 平移 + 把图像"镶"在一个更大的画布中 ---
    # 创建一个比原图大的画布
    canvas_w, canvas_h = w + 200, h + 200
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 200  # 灰色画布

    # 把原图放到画布中央
    # 这可以理解为"把原图平移到画布中心"
    center_matrix = np.float32([
        [1, 0, 100],  # 右移100
        [0, 1, 100]   # 下移100
    ])
    # 注意: warpAffine 中指定的输出尺寸大于原图，多余区域会露出背景（这里是灰色）
    # warpAffine 的 borderValue 参数可以指定背景颜色
    centered = cv2.warpAffine(img, center_matrix, (canvas_w, canvas_h),
                               borderValue=(200, 200, 200))

    show_multiple_images([
        ("原图", img),
        ("平移 (100, 50)", translated),
        ("居中放置", centered),
    ])
    cv2.destroyAllWindows()

    print("   💡 平移矩阵 = [[1,0,tx],[0,1,ty]]，用 warpAffine() 应用")


def demo_perspective():
    """
    演示6: 透视变换 (Perspective Transform)

    透视变换比仿射变换更强大——仿射变换保持"平行线仍然平行"，
    透视变换则可以把任意四边形映射到另一个四边形。
    就像你把一张纸倾斜，原来平行的线在照片中会汇聚到一点。

    应用场景:
        - 扫描文档校正（把拍歪的纸张"摆正"）
        - 车牌矫正
        - 增强现实（AR）
    """
    print("\n--- 📖 演示6: 透视变换（仿射 vs 透视）---")
    img = load_image("sample.jpg")
    h, w = img.shape[:2]

    # --- 定义原图中的四个点（找一张"倾斜"的图片区域）---
    # 这四个点构成一个梯形（模拟被拍歪的矩形）
    margin = 50  # 边距
    src_points = np.float32([
        [margin, margin],              # 左上角
        [w - margin, margin + 100],    # 右上角（比左上角低，模拟倾斜）
        [w - margin, h - margin],      # 右下角
        [margin, h - margin - 100],    # 左下角（比右下角高，模拟倾斜）
    ])

    # --- 定义目标矩形（我们要把它"拉正"成什么样子）---
    # 目标是一个端正的矩形
    dst_points = np.float32([
        [0, 0],          # 左上
        [w - 1, 0],      # 右上
        [w - 1, h - 1],  # 右下
        [0, h - 1],      # 左下
    ])

    # --- 计算透视变换矩阵 ---
    # cv2.getPerspectiveTransform(原图四点, 目标四点)
    # 返回一个 3×3 的透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    print(f"   透视变换矩阵 (3×3):\n{perspective_matrix}")

    # --- 应用透视变换 ---
    # cv2.warpPerspective() 和 warpAffine() 类似，但用的是 3×3 矩阵
    corrected = cv2.warpPerspective(img, perspective_matrix, (w, h))

    # --- 在原图上标记我们选中的四个点 ---
    img_with_pts = img.copy()
    for pt in src_points.astype(int):
        cv2.circle(img_with_pts, tuple(pt), 8, (0, 0, 255), -1)
    cv2.polylines(img_with_pts, [src_points.astype(int)], True, (0, 255, 0), 2)

    show_multiple_images([
        ("原图 + 选中区域", img_with_pts),
        ("透视校正后", corrected),
    ])
    cv2.destroyAllWindows()

    print("   💡 透视变换 = 把任意四边形"拉正"，文档扫描的核心技术")


# =============================================================================
# 本节总结
# =============================================================================
# 裁剪:    img[y1:y2, x1:x2] — numpy 切片
# 缩放:    cv2.resize(img, (w, h), interpolation=...)
# 翻转:    cv2.flip(img, flipCode)
# 旋转:    cv2.getRotationMatrix2D() + cv2.warpAffine()
# 平移:    自定义 2×3 矩阵 + cv2.warpAffine()
# 透视:    cv2.getPerspectiveTransform() + cv2.warpPerspective()
#
# 核心数学：仿射变换矩阵 2×3，透视变换矩阵 3×3
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🧱 基础篇 [2/3]: 基本图像操作")
    print("="*60)

    demo_crop()         # 演示1: 裁剪
    demo_resize()       # 演示2: 缩放
    demo_flip()         # 演示3: 翻转
    demo_rotate()       # 演示4: 旋转
    demo_translate()    # 演示5: 平移
    demo_perspective()  # 演示6: 透视

    print("\n✅ 基础篇 [2/3] 完成！")


if __name__ == "__main__":
    main()

# =============================================================================
# 🔍 分析篇 [1/3] - contours.py
# 轮廓分析：查找轮廓、绘制轮廓、形状识别、面积/周长计算
# =============================================================================
# 学完本模块你应该理解：
#   1. 什么是轮廓——图像中相连的边界点组成的曲线
#   2. 如何查找和绘制轮廓
#   3. 轮廓的几何属性：面积、周长、外接矩形、最小外接圆
#   4. 如何用轮廓做形状识别
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, show_image, show_multiple_images


# =============================================================================
# 核心概念: 轮廓 (Contour)
# =============================================================================
# 轮廓 = 图像中物体边界上的连续点连成的曲线。
#
# 想象你把一个物体的边缘用笔画出来——那条线就是轮廓。
#
# 轮廓 vs 边缘:
#   边缘 (Edge):    图像中像素值剧烈变化的地方（不一定是闭合的）
#   轮廓 (Contour): 闭合的边界曲线，描述一个完整物体的形状
#
# 轮廓的应用:
#   - 物体检测与计数（图中有几个圆？几个方块？）
#   - 形状识别（这是圆形还是三角形？）
#   - 尺寸测量（这个零件多大？）
#   - 手势识别（手指的轮廓特征）
#
# OpenCV 中查找轮廓的流程:
#   1. 灰度化
#   2. 二值化（阈值化或 Canny 边缘检测）
#   3. cv2.findContours() 查找轮廓
#   4. cv2.drawContours() 绘制轮廓
# =============================================================================


def demo_find_and_draw():
    """
    演示1: 查找和绘制轮廓

    这是轮廓分析的基础：找到图中的所有轮廓，并画出来。
    """
    print("\n--- 📖 演示1: 查找和绘制轮廓 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 步骤1: 预处理——得到二值图 ---
    # 轮廓查找需要在二值图上进行
    # 方法A: 阈值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 方法B: Canny 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # --- 步骤2: 查找轮廓 ---
    # cv2.findContours(image, mode, method)
    #   image:  二值图像（注意：findContours 会修改原图，所以最好传 .copy()）
    #   mode:   轮廓检索模式
    #     RETR_EXTERNAL:  只检索最外层轮廓
    #     RETR_LIST:      检索所有轮廓，不建立层级关系
    #     RETR_TREE:      检索所有轮廓，建立完整的层级关系（谁在谁里面）
    #     RETR_CCOMP:     两级层级：外层和内层
    #   method: 轮廓近似方法
    #     CHAIN_APPROX_NONE:      存储轮廓上所有点（内存大但精度高）
    #     CHAIN_APPROX_SIMPLE:    压缩水平/垂直/斜线段，只存端点（推荐!）
    #     CHAIN_APPROX_TC89_L1:   Teh-Chin 链近似算法
    #
    # 返回值（OpenCV 3+ 是2个值，OpenCV 4+ 是2个值）:
    #   contours: 轮廓列表，每个轮廓是一个 numpy 数组，形状 (N, 1, 2)
    #              N 是这个轮廓上点的数量，(1, 2) 是每个点的 (x, y) 坐标
    #   hierarchy: 层级信息，形状 (1, N, 4)
    #              [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"   找到 {len(contours)} 个轮廓")

    # 打印前几个轮廓的信息
    for i, cnt in enumerate(contours[:5]):
        print(f"   轮廓[{i}]: {len(cnt)} 个点, 面积={cv2.contourArea(cnt):.0f}")

    # --- 步骤3: 绘制轮廓 ---
    # cv2.drawContours(image, contours, contourIdx, color, thickness)
    #   contourIdx: -1 表示画所有轮廓，也可以指定画第几个

    # 在原图上绘制
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    # 不同颜色标记不同的轮廓
    img_colored = img.copy()
    for i, cnt in enumerate(contours):
        # 给每个轮廓一个随机颜色
        color = (
            (i * 50 + 50) % 255,
            (i * 80 + 100) % 255,
            (i * 120 + 150) % 255
        )
        cv2.drawContours(img_colored, contours, i, color, 3)

    # 在黑色背景上只画轮廓（纯净的轮廓图）
    blank = np.zeros_like(img)
    cv2.drawContours(blank, contours, -1, (0, 255, 0), 2)

    show_multiple_images([
        ("二值图 (阈值化)", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
        ("Canny 边缘", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)),
        ("轮廓 (绿色)", img_with_contours),
        ("彩色轮廓", img_colored),
        ("纯轮廓图", blank),
    ], cols=3)
    cv2.destroyAllWindows()

    print(f"   💡 共找到 {len(contours)} 个轮廓，使用 CHAIN_APPROX_SIMPLE 压缩存储")


def demo_contour_features():
    """
    演示2: 轮廓的几何属性

    每个轮廓都自带"身份证"——面积、周长、外接形状等几何特征。
    这些特征可以用来识别和分类物体。
    """
    print("\n--- 📖 演示2: 轮廓的几何属性 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积排序，取最大的几个轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    img_features = img.copy()

    for i, cnt in enumerate(contours):
        if len(cnt) < 5:  # 有些特征需要至少5个点
            continue

        # --- 1. 面积 (Area) ---
        # cv2.contourArea() 计算轮廓包围的面积
        area = cv2.contourArea(cnt)
        # 也可以直接用 Green 公式: 0.5 * sum(x_i*y_{i+1} - x_{i+1}*y_i)
        # 注意: 对于有孔洞的轮廓，考虑使用 cv2.contourArea(cnt, oriented=True)

        # --- 2. 周长 (Perimeter / Arc Length) ---
        # cv2.arcLength(contour, closed)
        #   closed=True:  轮廓是闭合曲线
        #   closed=False: 轮廓是开放曲线
        perimeter = cv2.arcLength(cnt, True)

        # --- 3. 外接矩形 (Bounding Rectangle) ---
        # 正外接矩形（和坐标轴对齐的）
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_features, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- 4. 最小外接旋转矩形 ---
        # 和坐标轴可以不对齐的旋转矩形——面积更小，更贴合物体
        rect = cv2.minAreaRect(cnt)
        # rect 格式: ((center_x, center_y), (width, height), angle)
        box = cv2.boxPoints(rect)  # 获取矩形四个顶点
        box = np.int32(box)        # 转为整数坐标（OpenCV 新版可能要用 np.intp）
        cv2.drawContours(img_features, [box], 0, (255, 0, 0), 2)

        # --- 5. 最小外接圆 ---
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        center = (int(cx), int(cy))
        cv2.circle(img_features, center, int(radius), (0, 0, 255), 2)

        # --- 6. 轮廓矩 (Moments) ---
        # 矩是形状的统计特征，可以计算质心、方向等
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx_m = int(M["m10"] / M["m00"])  # 质心 x
            cy_m = int(M["m01"] / M["m00"])  # 质心 y
            cv2.circle(img_features, (cx_m, cy_m), 5, (255, 255, 0), -1)

        # --- 打印信息 ---
        print(f"\n   轮廓[{i}]:")
        print(f"     面积: {area:.0f} 像素²")
        print(f"     周长: {perimeter:.1f} 像素")
        print(f"     正外接矩形: ({x},{y}) {w}x{h}")
        print(f"     最小外接圆半径: {radius:.1f}")
        if M["m00"] != 0:
            print(f"     质心: ({cx_m}, {cy_m})")

    show_multiple_images([
        ("原图", img),
        ("几何特征标注", img_features),
    ])
    cv2.destroyAllWindows()

    # 标注图例
    print("\n   颜色图例: 🟢正外接矩形  🔵最小旋转矩形  🔴最小外接圆  🟡质心")


def demo_shape_matching():
    """
    演示3: 形状匹配与识别

    用轮廓的几何特征来判断这是什么形状。

    常用方法:
        - 轮廓近似 (cv2.approxPolyDP): 把轮廓近似为多边形，数顶点数
        - Hu 矩 (cv2.HuMoments): 七个不随旋转/缩放/平移变化的特征值
        - 形状匹配 (cv2.matchShapes): 比较两个形状的相似度（基于 Hu 矩）
    """
    print("\n--- 📖 演示3: 形状识别 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    img_shapes = img.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # 忽略太小的轮廓
            continue

        # --- 轮廓近似 ---
        # cv2.approxPolyDP(curve, epsilon, closed)
        #   epsilon: 近似精度——原轮廓到近似多边形的最大距离
        #            值越大，近似的边越少（越粗糙）
        #   通常设 epsilon = 周长 × 某个比例 (如 0.02)
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # --- 根据顶点数判断形状 ---
        vertices = len(approx)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # 计算外接矩形宽高比
        aspect_ratio = float(w) / h if h > 0 else 0

        # 判断逻辑
        if vertices == 3:
            shape_name = "三角形"
            color = (0, 255, 0)
        elif vertices == 4:
            # 正方形 vs 矩形
            if 0.9 <= aspect_ratio <= 1.1:
                shape_name = "正方形"
                color = (255, 0, 0)
            else:
                shape_name = "矩形"
                color = (0, 0, 255)
        elif vertices == 5:
            shape_name = "五边形"
            color = (255, 255, 0)
        elif vertices >= 8:
            # 圆形 → 轮廓近似后有很多顶点
            # 进一步用面积-周长比验证
            # 对于完美的圆: area = pi*r², perimeter = 2*pi*r
            # 面积 / 周长² = pi*r² / (4*pi²*r²) = 1/(4*pi)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.7 <= circularity <= 1.3:
                shape_name = "圆形"
                color = (255, 0, 255)
            else:
                shape_name = f"{vertices}边形"
                color = (128, 128, 128)
        else:
            shape_name = f"{vertices}边形"
            color = (128, 128, 128)

        # --- 绘制识别结果 ---
        cv2.drawContours(img_shapes, [approx], 0, color, 2)
        cv2.putText(img_shapes, f"{shape_name} ({area:.0f})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    show_multiple_images([
        ("原图", img),
        ("形状识别结果", img_shapes),
    ])
    cv2.destroyAllWindows()

    print("   💡 轮廓近似(approxPolyDP)是形状识别的关键——数顶点就能判断形状！")


def demo_contour_hierarchy():
    """
    演示4: 轮廓层级关系

    当一个物体包含孔洞（比如甜甜圈），或物体中套着另一个物体时，
    轮廓之间存在父子关系。
    """
    print("\n--- 📖 演示4: 轮廓层级关系 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # RETR_TREE: 建立完整层级关系
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # hierarchy 的形状是 (1, N, 4)
    # hierarchy[0][i] = [Next, Previous, First_Child, Parent]
    #   Next:        同一层级的下一个轮廓的索引
    #   Previous:    同一层级的上一个轮廓的索引
    #   First_Child: 第一个子轮廓的索引
    #   Parent:      父轮廓的索引
    #   -1 表示"没有"

    print(f"   总轮廓数: {len(contours)}")
    print(f"   hierarchy 形状: {hierarchy.shape}")

    # 分析层级
    outer_count = 0  # 最外层轮廓
    inner_count = 0  # 内层轮廓（有父轮廓的）

    for i in range(len(contours)):
        parent = hierarchy[0][i][3]
        if parent == -1:
            outer_count += 1
        else:
            inner_count += 1

    print(f"   最外层轮廓: {outer_count} 个")
    print(f"   内层轮廓(孔洞): {inner_count} 个")

    # 可视化层级
    img_hierarchy = img.copy()
    for i, cnt in enumerate(contours):
        parent = hierarchy[0][i][3]
        if parent == -1:
            # 外层轮廓用绿色
            cv2.drawContours(img_hierarchy, [cnt], 0, (0, 255, 0), 2)
            cv2.putText(img_hierarchy, f"outer {i}", tuple(cnt[0][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # 内层轮廓用红色
            cv2.drawContours(img_hierarchy, [cnt], 0, (0, 0, 255), 2)
            cv2.putText(img_hierarchy, f"inner {i}→parent{parent}",
                        tuple(cnt[0][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    show_image("轮廓层级: 绿色=外层, 红色=内层(孔洞)", img_hierarchy)
    cv2.destroyAllWindows()

    print("   💡 hierarchy[0][i] = [next, prev, child, parent] —— 树形结构描述嵌套关系")


# =============================================================================
# 本节总结
# =============================================================================
# 轮廓查找流程:
#   灰度化 → 二值化 → cv2.findContours() → cv2.drawContours()
#
# 轮廓属性:
#   面积:   cv2.contourArea(cnt)
#   周长:   cv2.arcLength(cnt, True)
#   外接矩形: cv2.boundingRect(cnt)  / cv2.minAreaRect(cnt)
#   外接圆:  cv2.minEnclosingCircle(cnt)
#   质心:   cv2.moments(cnt) → M["m10"]/M["m00"], M["m01"]/M["m00"]
#
# 形状识别:
#   approx = cv2.approxPolyDP(cnt, epsilon, True) → 数顶点数
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🔍 分析篇 [1/3]: 轮廓分析")
    print("="*60)

    demo_find_and_draw()       # 演示1: 查找和绘制轮廓
    demo_contour_features()    # 演示2: 几何属性
    demo_shape_matching()      # 演示3: 形状识别
    demo_contour_hierarchy()   # 演示4: 层级关系

    print("\n✅ 分析篇 [1/3] 完成！")


if __name__ == "__main__":
    main()

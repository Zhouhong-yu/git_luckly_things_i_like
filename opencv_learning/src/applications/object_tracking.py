# =============================================================================
# 🏆 应用篇 [2/3] - object_tracking.py
# 目标跟踪：光流法、Meanshift、Camshift、Tracker API
# =============================================================================
# 学完本模块你应该理解：
#   1. 光流法 (Optical Flow) 的原理——追踪像素的运动
#   2. Lucas-Kanade 稀疏光流
#   3. Meanshift 和 Camshift 跟踪算法
#   4. OpenCV 的 Tracker API（更高级的跟踪器）
# =============================================================================

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import (
    load_image, create_sample_image, show_image, show_multiple_images
)


# =============================================================================
# 核心概念: 目标跟踪
# =============================================================================
# 目标跟踪 (Object Tracking) 回答: "目标在下一帧移动到了哪里？"
#
# 目标跟踪 vs 目标检测:
#   检测 (Detection): 在每一帧独立地找目标（不看历史）
#   跟踪 (Tracking):  利用前一帧的信息预测当前帧目标的位置（看历史）
#
# 常见跟踪方法:
#
# 1. 光流法 (Optical Flow):
#    原理: 假设相邻帧中同一物体的像素亮度不变，
#         通过计算像素的"运动向量"来跟踪
#    代表: Lucas-Kanade (稀疏), Farneback (稠密)
#
# 2. Meanshift / Camshift:
#    原理: 用颜色直方图描述目标，在每帧中搜索颜色分布最相似的区域
#          Meanshift: 爬山算法，沿概率密度梯度方向迭代移动
#          Camshift:  Meanshift + 自适应窗口大小
#
# 3. 现代 Tracker (KCF, CSRT, MOSSE 等):
#    原理: 更复杂的数学模型（相关滤波、深度学习等）
#    使用: OpenCV 的 cv2.TrackerXXX API
# =============================================================================


def demo_optical_flow():
    """
    演示1: Lucas-Kanade 光流法

    Lucas-Kanade 是最经典的稀疏光流算法。
    它假设在一个小窗口内，所有像素的运动方向相同。

    应用场景:
        - 视频稳定
        - 运动检测
        - 车辆/行人跟踪
        - 结构振动分析
    """
    print("\n--- 📖 演示1: Lucas-Kanade 光流法 ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 因为没有真实的视频帧，我们模拟一个"运动"
    # 把原图平移10个像素当作"下一帧"
    h, w = gray.shape
    tx, ty = 10, 5  # x 方向移10px, y 方向移5px

    # 创建"下一帧"——对平移后的区域做仿射变换
    trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    gray_next = cv2.warpAffine(gray, trans_matrix, (w, h))

    # --- 步骤1: 在第一帧中找一些好的角点作为跟踪目标 ---
    # 跟踪需要"好"的特征点——角点是最佳选择
    feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10)
    prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    if prev_points is None:
        print("   ⚠️ 未找到足够的特征点")
        return

    print(f"   跟踪 {len(prev_points)} 个特征点")

    # --- 步骤2: 用光流法计算这些点在下一帧中的位置 ---
    # cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, ...)
    #   Pyr 表示使用了图像金字塔（多尺度处理）
    #   LK 表示 Lucas-Kanade 算法
    #
    #   winSize:  搜索窗口大小（越大越鲁棒但越慢）
    #   maxLevel: 图像金字塔层数
    #   criteria: 迭代停止条件
    lk_params = dict(
        winSize=(15, 15),      # 搜索窗口大小
        maxLevel=2,            # 金字塔层数
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        gray, gray_next, prev_points, None, **lk_params
    )

    # status[i]=1 表示第 i 个点成功跟踪，=0 表示失败
    # error[i] 是跟踪误差

    good_prev = prev_points[status == 1]
    good_next = next_points[status == 1]
    good_errors = error[status == 1]

    print(f"   成功跟踪: {len(good_prev)} 个点")
    if len(good_errors) > 0:
        print(f"   平均跟踪误差: {good_errors.mean():.3f}")

    # --- 可视化光流 ---
    # 在两个帧上都绘制特征点，并用箭头连接
    # 把两个灰度帧转成彩色便于区分
    frame1_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame2_color = cv2.cvtColor(gray_next, cv2.COLOR_GRAY2BGR)
    flow_viz = np.hstack([frame1_color, frame2_color])

    # 在拼接图上绘制
    for i, (prev_pt, next_pt) in enumerate(zip(good_prev[:30], good_next[:30])):
        x1, y1 = prev_pt.ravel()
        x2, y2 = next_pt.ravel()
        # 第一帧中的点
        cv2.circle(flow_viz, (int(x1), int(y1)), 3, (0, 255, 0), -1)
        # 第二帧中的点（偏移 w 个像素因为拼接在右边）
        cv2.circle(flow_viz, (int(x2) + w, int(y2)), 3, (0, 0, 255), -1)
        # 连线
        cv2.line(flow_viz, (int(x1), int(y1)),
                 (int(x2) + w, int(y2)), (255, 255, 0), 1)

    show_image(f"光流跟踪 (绿=起始点, 红=终点, 线=运动轨迹)", flow_viz)
    cv2.destroyAllWindows()

    print(f"   💡 预测的运动: dx≈{tx}, dy≈{ty}，每对红-绿点之间的距离反映了运动量")


def demo_dense_optical_flow():
    """
    演示2: 稠密光流 (Dense Optical Flow)

    稠密光流计算图像中每个像素的运动向量，而非仅特征点。
    用颜色编码运动方向和大小。

    Gunnar Farneback 算法:
        - 用多项式近似每个像素的邻域
        - 估计多项式之间的位移
        - 输出一个 (h, w, 2) 的流场
    """
    print("\n--- 📖 演示2: 稠密光流 (Farneback) ---")
    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 模拟运动
    h, w = gray.shape
    trans_matrix = np.float32([[1, 0, 8], [0, 1, 3]])
    gray_next = cv2.warpAffine(gray, trans_matrix, (w, h))

    # --- 计算稠密光流 ---
    # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    flow = cv2.calcOpticalFlowFarneback(
        gray, gray_next, None,
        pyr_scale=0.5,  # 金字塔缩放因子
        levels=3,        # 金字塔层数
        winsize=15,      # 窗口大小
        iterations=3,    # 迭代次数
        poly_n=5,        # 多项式展开的邻域大小
        poly_sigma=1.2,  # 高斯标准差
        flags=0
    )
    # flow 的形状是 (h, w, 2):
    #   flow[y, x, 0] = 该像素在 x 方向的运动（dx）
    #   flow[y, x, 1] = 该像素在 y 方向的运动（dy）

    print(f"   稠密光流 shape: {flow.shape}")
    print(f"   平均 x 方向运动: {flow[:,:,0].mean():.2f} px (实际 tx=8)")
    print(f"   平均 y 方向运动: {flow[:,:,1].mean():.2f} px (实际 ty=3)")

    # --- 可视化稠密光流 ---
    # 用 HSV 颜色编码: H=运动方向, S=1(最大饱和度), V=运动大小
    # 这是一个非常有名的可视化方法

    # 计算每个像素的运动大小和方向
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 创建 HSV 图像
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # H: 方向映射到 0-180
    hsv[..., 1] = 255                       # S: 最大饱和度
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # V: 运动大小

    # HSV → BGR 用于显示
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    show_multiple_images([
        ("前一帧", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
        ("后一帧", cv2.cvtColor(gray_next, cv2.COLOR_GRAY2BGR)),
        ("稠密光流 (颜色=方向, 亮度=速度)", flow_rgb),
    ])
    cv2.destroyAllWindows()

    print("   💡 颜色编码: 不同颜色=不同运动方向，亮度=运动速度")


def demo_meanshift_camshift():
    """
    演示3: Meanshift 和 Camshift 跟踪

    Meanshift = 基于颜色直方图的爬山式搜索
    Camshift  = Meanshift + 自适应窗口大小调整

    工作原理:
        1. 在初始帧选择目标（比如一个红色球）
        2. 计算目标的颜色直方图（HSV 空间中的 H 通道）
        3. 在后续帧中，用反向投影找到颜色最相似的区域
        4. Meanshift 迭代移动窗口到概率密度的"顶峰"
    """
    print("\n--- 📖 演示3: Meanshift / Camshift 跟踪原理 ---")
    img = load_image("sample.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 模拟: 选择一个区域作为跟踪目标 ---
    # 在实际应用中，第一帧由用户手动选择或检测器自动选择
    # 这里我们选图像中心的一个矩形区域
    h, w = img.shape[:2]
    track_window = (w//2 - 50, h//2 - 50, 100, 100)  # (x, y, w, h)
    x, y, tw, th = track_window

    # --- 计算目标区域的颜色直方图 ---
    roi = hsv[y:y+th, x:x+tw]
    # 只使用 Hue(色调) 通道，因为它对光照变化最鲁棒
    roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # --- Meanshift 迭代 ---
    # 终止条件: 迭代10次 或 移动距离小于1像素
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # 反向投影
    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # cv2.meanShift(probImage, window, criteria)
    #   probImage: 反向投影图（概率图）
    #   window:    初始搜索窗口 (x, y, w, h)
    #   criteria:  停止条件
    #   返回: (新窗口, 迭代次数)
    ret, new_window = cv2.meanShift(back_proj, track_window, term_crit)

    # --- Camshift ---
    # cv2.CamShift() 和 meanShift 类似，但会自适应调整窗口大小
    #   返回: (旋转矩形, 新窗口)
    ret_cam, cam_window = cv2.CamShift(back_proj, track_window, term_crit)

    # --- 可视化 ---
    img_mean = img.copy()
    x_m, y_m, w_m, h_m = new_window
    cv2.rectangle(img_mean, (x_m, y_m), (x_m + w_m, y_m + h_m), (0, 255, 0), 2)
    cv2.putText(img_mean, "Meanshift", (x_m, y_m - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    img_cam = img.copy()
    # Camshift 返回的是旋转矩形
    box = cv2.boxPoints(ret_cam)
    box = np.int32(box)
    cv2.drawContours(img_cam, [box], 0, (255, 0, 0), 2)
    cv2.putText(img_cam, "Camshift (自适应大小)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    img_init = img.copy()
    cv2.rectangle(img_init, (x, y), (x + tw, y + th), (255, 255, 0), 2)

    back_proj_show = cv2.cvtColor(back_proj, cv2.COLOR_GRAY2BGR)

    show_multiple_images([
        ("初始选择区域", img_init),
        ("反向投影 (概率图)", back_proj_show),
        ("Meanshift 结果", img_mean),
        ("Camshift 结果", img_cam),
    ], cols=2)
    cv2.destroyAllWindows()

    print("   💡 Meanshift = 爬山找概率密度峰值; Camshift = Meanshift + 自适应窗口")


def demo_tracker_api():
    """
    演示4: OpenCV Tracker API

    OpenCV 提供了多个高级跟踪器，使用统一的接口。
    这些跟踪器比 Meanshift 更准确、更鲁棒。

    常用跟踪器:
        - CSRT:  准确率高，速度较慢（推荐用于高精度需求）
        - KCF:   速度快，准确率中等（经典方法）
        - MOSSE: 非常快，但准确率较低（适合嵌入式设备）
        - MIL:   多实例学习，对部分遮挡鲁棒

    使用流程:
        1. tracker = cv2.TrackerXXX_create()
        2. tracker.init(frame, bbox)     — 初始化，选定目标
        3. ok, bbox = tracker.update(frame) — 每帧更新位置
    """
    print("\n--- 📖 演示4: OpenCV 高级 Tracker API ---")

    # 列出可用的跟踪器
    available_trackers = []

    # OpenCV 不同版本的跟踪器 API 略有不同
    # 这里尝试加载各种可用的跟踪器
    tracker_creators = {
        "CSRT":  lambda: cv2.TrackerCSRT_create(),
        "KCF":   lambda: cv2.TrackerKCF_create(),
        "MOSSE": lambda: cv2.legacy.TrackerMOSSE_create() if hasattr(cv2, 'legacy') else None,
        "MIL":   lambda: cv2.TrackerMIL_create(),
    }

    for name, creator in tracker_creators.items():
        try:
            tracker = creator()
            if tracker is not None:
                available_trackers.append((name, tracker))
                print(f"   ✅ {name} 跟踪器可用")
        except (AttributeError, Exception):
            print(f"   ⚠️ {name} 跟踪器不可用")

    if not available_trackers:
        print("   ⚠️ 没有可用的跟踪器")
        print("   💡 升级 opencv-python 或 opencv-contrib-python 到最新版本来获取更多跟踪器")
        return

    # --- 模拟跟踪 ---
    # 在真实场景中，这里是逐帧的视频流
    img = load_image("sample.jpg")
    h, w = img.shape[:2]

    # 选择跟踪目标——图像中心的一个矩形
    bbox = (w//2 - 50, h//2 - 50, 100, 100)  # (x, y, w, h)

    # --- 初始化跟踪器 ---
    name, tracker = available_trackers[0]
    try:
        tracker.init(img, bbox)
        print(f"\n   使用 {name} 跟踪器")
        print(f"   初始目标位置: ({bbox[0]}, {bbox[1]}) {bbox[2]}x{bbox[3]}")

        # --- 模拟下一帧 ---
        # 平移目标位置模拟运动
        img_next = cv2.warpAffine(
            img,
            np.float32([[1, 0, 10], [0, 1, 5]]),
            (w, h)
        )

        # 更新跟踪
        ok, new_bbox = tracker.update(img_next)
        if ok:
            print(f"   更新后位置: ({new_bbox[0]:.0f}, {new_bbox[1]:.0f}) "
                  f"{new_bbox[2]:.0f}x{new_bbox[3]:.0f}")
            print(f"   预测位移: dx={new_bbox[0]-bbox[0]:.0f}, dy={new_bbox[1]-bbox[1]:.0f}")

            # 可视化
            img_tracked = img_next.copy()
            x, y, bw, bh = [int(v) for v in new_bbox]
            cv2.rectangle(img_tracked, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(img_tracked, f"Tracked ({name})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 原始位置
            cv2.rectangle(img_tracked, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 0, 255), 1)

            show_image(f"{name} 跟踪结果 (绿=新位置, 红=原位置)", img_tracked)
            cv2.destroyAllWindows()
        else:
            print("   ⚠️ 跟踪失败（目标可能移出了画面）")
    except Exception as e:
        print(f"   ❌ 跟踪出错: {e}")


# =============================================================================
# 本节总结
# =============================================================================
# 光流法:
#   稀疏: cv2.calcOpticalFlowPyrLK()     — 跟踪特征点，快速
#   稠密: cv2.calcOpticalFlowFarneback() — 每个像素的运动，信息丰富
#
# Meanshift/Camshift:
#   基于颜色直方图的概率密度梯度搜索
#   Meanshift:  cv2.meanShift()
#   Camshift:   cv2.CamShift() (自适应窗口大小)
#
# 高级 Tracker API:
#   tracker = cv2.TrackerXXX_create()
#   tracker.init(frame, bbox)
#   ok, bbox = tracker.update(next_frame)
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🏆 应用篇 [2/3]: 目标跟踪")
    print("="*60)

    demo_optical_flow()        # 演示1: LK 光流法
    demo_dense_optical_flow()  # 演示2: 稠密光流
    demo_meanshift_camshift()  # 演示3: Meanshift/Camshift
    demo_tracker_api()         # 演示4: Tracker API

    print("\n✅ 应用篇 [2/3] 完成！")


if __name__ == "__main__":
    main()

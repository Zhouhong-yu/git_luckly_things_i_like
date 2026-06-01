# =============================================================================
# 🏆 应用篇 [1/3] - face_detection.py
# 人脸检测：Haar Cascade 级联分类器 + DNN 深度学习方法
# =============================================================================
# 学完本模块你应该理解：
#   1. Haar Cascade 人脸检测的原理
#   2. 如何使用 OpenCV 预训练的人脸检测模型
#   3. 检测参数对结果的影响
#   4. DNN 人脸检测（更准确）
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
# 核心概念: 人脸检测
# =============================================================================
# 人脸检测 (Face Detection) 回答："图像中哪里有人脸？"
#
# 它不同于人脸识别 (Face Recognition) — 识别需要回答"这是谁的脸？"
#
# Haar Cascade (Viola-Jones 算法，2001年):
#   原理:
#     1. 用"Haar-like 特征"描述人脸（眼睛区域比脸颊暗、鼻梁比两侧亮等）
#     2. 用"积分图"快速计算特征（O(1)时间计算任意矩形区域的像素和）
#     3. 用 AdaBoost 从大量特征中选出最有区分力的特征
#     4. 用"级联分类器"逐级过滤——像漏斗一样，大部分非人脸窗口被快速排除
#
#   优点: 非常快，适合实时检测
#   缺点: 对侧脸、遮挡、光照变化敏感
#
# DNN 方法:
#   使用深度神经网络（如 SSD、YOLO 中的面部检测模型）
#   更准确，但需要更多计算资源
# =============================================================================

# --- 全局: 加载 Haar Cascade 分类器 ---
# OpenCV 自带了一些预训练的 Haar Cascade 模型
# 它们以 XML 文件的形式存储在 OpenCV 的安装目录中
def load_cascade():
    """
    加载 OpenCV 自带的人脸检测 Haar Cascade 模型。

    返回:
        cv2.CascadeClassifier 对象，如果加载失败则返回 None
    """
    # cv2.data.haarcascades 指向 OpenCV 的 Haar Cascade XML 文件目录
    # 这些是 OpenCV 安装时自带的预训练模型
    try:
        cascade_dir = cv2.data.haarcascades
    except AttributeError:
        # 某些 OpenCV 构建版本可能没有 haarcascades 属性
        # 尝试常见的安装路径
        possible_dirs = [
            os.path.join(os.path.dirname(cv2.__file__), "data"),
            "/usr/share/opencv4/haarcascades",
            "/usr/local/share/opencv4/haarcascades",
        ]
        cascade_dir = ""
        for d in possible_dirs:
            if os.path.isdir(d):
                cascade_dir = d + "/"
                break

    cascade_path = cascade_dir + "haarcascade_frontalface_default.xml"

    # 也可以尝试其他模型：
    # haarcascade_frontalface_alt.xml       — 更准确但更慢
    # haarcascade_frontalface_alt2.xml      — 最准确
    # haarcascade_frontalface_alt_tree.xml  — 树形结构
    # haarcascade_profileface.xml           — 侧脸检测

    if not os.path.exists(cascade_path):
        print(f"   ⚠️ 未找到 Haar Cascade 文件: {cascade_path}")
        return None

    # cv2.CascadeClassifier() 加载预训练的级联分类器
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade


def demo_haar_face_detection():
    """
    演示1: Haar Cascade 人脸检测

    这是最经典、最简单的人脸检测方法。
    虽然准确率不如深度学习方法，但速度极快，适合实时应用。
    """
    print("\n--- 📖 演示1: Haar Cascade 人脸检测 ---")
    face_cascade = load_cascade()
    if face_cascade is None:
        print("   ❌ 无法加载人脸检测模型，跳过")
        return

    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 人脸检测 ---
    # face_cascade.detectMultiScale(image, scaleFactor, minNeighbors, minSize)
    #   scaleFactor: 每次将检测窗口缩小多少倍（1.1=每次缩小10%）
    #                值越小→检测越细致→越慢但越不容易漏检
    #                值越大→检测越粗糙→越快但可能漏检小脸
    #   minNeighbors: 一个区域要被判定为人脸，至少需要多少个相邻检测框
    #                 值越大→要求越严格→误检越少但可能漏检
    #   minSize:     人脸的最小尺寸（低于此的不检测）
    #   maxSize:     人脸的最大尺寸（高于此的不检测）
    #   返回值: 矩形列表 [(x, y, w, h), ...]
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"   检测到 {len(faces)} 张人脸")

    # --- 绘制检测结果 ---
    img_faces = img.copy()
    for i, (x, y, w, h) in enumerate(faces):
        # 画矩形框
        cv2.rectangle(img_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 标号
        cv2.putText(img_faces, f"Face {i+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 在人脸区域做简单的模糊（隐私保护风格）
        face_roi = img[y:y+h, x:x+w]
        if face_roi.size > 0:
            blurred_face = cv2.GaussianBlur(face_roi, (31, 31), 30)

    print(f"   检测到的人脸位置:")
    for i, (x, y, w, h) in enumerate(faces):
        print(f"     人脸{i+1}: x={x}, y={y}, 宽={w}, 高={h}")

    # --- 对比不同参数 ---
    # 更严格的参数
    faces_strict = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=7, minSize=(50, 50)
    )

    # 更宽松的参数
    faces_loose = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
    )

    img_strict = img.copy()
    for (x, y, w, h) in faces_strict:
        cv2.rectangle(img_strict, (x, y), (x + w, y + h), (0, 0, 255), 2)

    img_loose = img.copy()
    for (x, y, w, h) in faces_loose:
        cv2.rectangle(img_loose, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(img_strict, f"严格: {len(faces_strict)}个", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_loose, f"宽松: {len(faces_loose)}个", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_faces, f"标准: {len(faces)}个", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    show_multiple_images([
        ("标准参数 (green)", img_faces),
        ("严格参数 (red)", img_strict),
        ("宽松参数 (blue)", img_loose),
    ])
    cv2.destroyAllWindows()

    print("   💡 minNeighbors 越大→越严格→误检少; scaleFactor 越小→越细致→越慢")


def demo_face_with_eyes():
    """
    演示2: 同时检测人脸和眼睛

    OpenCV 也提供了眼睛检测的 Haar Cascade。
    人脸检测 + 眼睛检测 = 更高的人脸确认率。
    """
    print("\n--- 📖 演示2: 人脸 + 眼睛检测 ---")
    face_cascade = load_cascade()
    if face_cascade is None:
        print("   ❌ 无法加载人脸检测模型，跳过")
        return

    # 加载眼睛检测器
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    if not os.path.exists(eye_cascade_path):
        print("   ⚠️ 未找到眼睛检测模型")
        return
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    img = load_image("sample.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 先检测人脸 ---
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    print(f"   检测到 {len(faces)} 张人脸")

    img_result = img.copy()

    for (x, y, w, h) in faces:
        # 画人脸框
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- 在人脸区域内检测眼睛 ---
        # 只在人脸上半部分找眼睛（眼睛不可能在脸部下半部分）
        face_roi_gray = gray[y:y + h//2, x:x + w]
        face_roi_color = img_result[y:y + h//2, x:x + w]

        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 3, minSize=(15, 15))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    show_image(f"人脸(绿框) + 眼睛(蓝框) 检测 ({len(faces)}个)", img_result)
    cv2.destroyAllWindows()

    print("   💡 人脸+眼睛双重验证 → 更高的人脸确认率，降低误检")


def demo_dnn_face_detection():
    """
    演示3: DNN 深度学习人脸检测

    OpenCV 从 3.3 版本开始支持加载 Caffe/TensorFlow 等框架训练的模型。
    对于人脸检测，可以使用预训练的 SSD 或 YOLO 模型。

    这里使用 OpenCV 自带的 DNN 人脸检测模型（如果需要下载的话）。
    如果模型不可用，将使用 Haar Cascade 作为回退方案。

    DNN vs Haar:
        DNN: 更准确（特别是侧脸、部分遮挡），但需要更多计算
        Haar: 非常快，但准确率较低
    """
    print("\n--- 📖 演示3: DNN 深度学习人脸检测 ---")

    # DNN 人脸检测需要模型文件
    # 常用模型: "opencv_face_detector_uint8.pb" (TensorFlow)
    #          "res10_300x300_ssd_iter_140000_fp16.caffemodel" (Caffe)
    # 这里检查是否可用
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data",
        "opencv_face_detector_uint8.pb"
    )
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data",
        "opencv_face_detector.pbtxt"
    )

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("   ⚠️ DNN 模型文件未找到，说明如何使用 Haar 进行类似的区域检测")
        print("   💡 生产环境中建议使用 DNN: 下载预训练模型文件到 data/ 目录")
        print("     常用: OpenCV Face Detector (SSD), MTCNN, RetinaFace")
        print("     使用方式: cv2.dnn.readNetFromTensorflow(model, config)")
        print("              net.forward() 获得检测结果")
        return

    # 如果模型存在，执行 DNN 检测
    try:
        net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        img = load_image("sample.jpg")
        h, w = img.shape[:2]

        # 创建 blob（神经网络需要的输入格式）
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300), [104, 117, 123], False, False
        )
        net.setInput(blob)
        detections = net.forward()

        img_result = img.copy()
        count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_result, f"{confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"   DNN 检测到 {count} 张人脸")
        show_image(f"DNN 人脸检测 ({count}个)", img_result)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"   ❌ DNN 检测失败: {e}")


def demo_real_time_simulation():
    """
    演示4: 模拟实时人脸检测流程

    展示一个完整的人脸检测处理流水线：
    1. 读取图像
    2. 灰度化
    3. 直方图均衡化（改善光照）
    4. 人脸检测
    5. 后处理（绘制、模糊等）
    """
    print("\n--- 📖 演示4: 完整人脸检测流水线 ---")
    face_cascade = load_cascade()
    if face_cascade is None:
        print("   ❌ 无法加载人脸检测模型，跳过")
        return

    img = load_image("sample.jpg")

    # --- 步骤1: 灰度化 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step1_show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- 步骤2: 直方图均衡化（改善光照）---
    equalized = cv2.equalizeHist(gray)
    step2_show = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    # --- 步骤3: 高斯模糊降噪 ---
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    step3_show = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # --- 步骤4: 人脸检测 ---
    faces = face_cascade.detectMultiScale(blurred, 1.1, 5, minSize=(30, 30))

    # --- 步骤5: 绘制结果 ---
    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 在人脸上方绘制一个半透明的标签
        overlay = result.copy()
        cv2.rectangle(overlay, (x, y - 25), (x + w, y), (0, 255, 0), -1)
        result = cv2.addWeighted(result, 1, overlay, 0.5, 0)
        cv2.putText(result, "FACE", (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(step1_show, "Step 1: 灰度化", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(step2_show, "Step 2: 直方图均衡化", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(step3_show, "Step 3: 高斯模糊", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    show_multiple_images([
        ("0. 原图", img),
        ("1. 灰度化", step1_show),
        ("2. 直方图均衡化", step2_show),
        ("3. 高斯模糊", step3_show),
    ], cols=2)
    cv2.destroyAllWindows()

    show_image(f"4. 最终结果: 检测到 {len(faces)} 张人脸", result)
    cv2.destroyAllWindows()

    print("   💡 直方图均衡化 → 改善光照 → 提高检测率（暗处的人脸也能检测到）")


# =============================================================================
# 本节总结
# =============================================================================
# Haar Cascade 人脸检测流程:
#   1. 加载模型: cv2.CascadeClassifier(xml_path)
#   2. 灰度化:   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   3. 检测:     cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
#   4. 绘制:     cv2.rectangle() 画框
#
# 提升检测率技巧:
#   - 直方图均衡化改善光照
#   - 调整 scaleFactor 和 minNeighbors
#   - 组合多个 Cascade（人脸+眼睛+嘴巴）
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  🏆 应用篇 [1/3]: 人脸检测")
    print("="*60)

    demo_haar_face_detection()      # 演示1: Haar Cascade
    demo_face_with_eyes()           # 演示2: 人脸+眼睛
    demo_dnn_face_detection()       # 演示3: DNN 方法
    demo_real_time_simulation()     # 演示4: 完整流水线

    print("\n✅ 应用篇 [1/3] 完成！")


if __name__ == "__main__":
    main()

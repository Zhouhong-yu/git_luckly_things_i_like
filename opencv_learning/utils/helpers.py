# =============================================================================
# 🛠 helpers.py - 工具函数库
# =============================================================================
# 本文件提供一些常用的辅助函数，供其他模块调用。
# 这样做的好处是：避免在多个文件中重复写相同的代码（DRY 原则：Don't Repeat Yourself）
# =============================================================================

# Python 3.7-3.8 兼容：让所有类型注解不被求值，避免 list[tuple] 语法报错
from __future__ import annotations

# Windows 控制台编码修复：让 emoji 和中文能正常输出
import sys as _sys
if _sys.platform == 'win32' and hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2                # OpenCV 库：import cv2 是约定俗成的写法
import numpy as np        # NumPy 库：import numpy as np 也是约定俗成的写法
import os                 # 操作系统接口：处理文件路径


# =============================================================================
# 1. 路径工具函数
# =============================================================================

def get_data_path(filename: str = "sample.jpg") -> str:
    """
    获取 data 目录下文件的完整路径。

    OpenCV 读取图片时需要完整路径，这个函数帮我们自动拼接路径。

    参数:
        filename: 图片文件名，默认为 "sample.jpg"

    返回:
        str: 文件的完整路径，例如 "D:/zhy/ai/git/opencv_learning/data/sample.jpg"

    原理:
        os.path.dirname(os.path.abspath(__file__)) 获取当前文件(helpers.py)所在目录
        再往上两级就到了项目根目录，然后拼接 data/ 和文件名
    """
    # __file__ 是当前文件的路径，例如 ".../utils/helpers.py"
    # os.path.abspath() 转为绝对路径
    # os.path.dirname() 取目录部分，即 ".../utils"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # os.path.dirname() 再取一次，即 ".../opencv_learning"（项目根目录）
    project_root = os.path.dirname(current_dir)

    # os.path.join() 把多个路径片段拼接成一个完整路径
    # 它会自动使用当前操作系统的路径分隔符（Windows 是 \，Linux 是 /）
    return os.path.join(project_root, "data", filename)


def get_output_path(filename: str) -> str:
    """
    获取 outputs 目录下文件的完整路径。
    用于保存处理后的图片结果。

    参数:
        filename: 要保存的文件名

    返回:
        str: 输出文件的完整路径
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    # 如果 outputs 目录不存在就创建它
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 表示目录已存在也不报错
    return os.path.join(output_dir, filename)


# =============================================================================
# 2. 图像生成工具 —— 当 data/ 目录下没有测试图片时，自动生成测试图
# =============================================================================

def create_sample_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    生成一张彩色的测试图片。
    这张图片包含多种几何图形和颜色渐变，适合用来测试各种图像处理算法。

    参数:
        width:  图片宽度（像素），默认 640
        height: 图片高度（像素），默认 480

    返回:
        np.ndarray: 一个 [height, width, 3] 的 BGR 彩色图像（数据类型 uint8）

    理解 numpy 数组：
        np.ndarray 是 NumPy 的核心数据结构——多维数组。
        OpenCV 中的图像本质上就是一个 np.ndarray：
        - 彩色图是三维数组: [行(高), 列(宽), 通道(3)]
        - 灰度图是二维数组: [行(高), 列(宽)]
        - 每个元素的值范围是 0-255（uint8 类型）
    """
    # --- 创建白色背景 ---
    # np.ones() 创建一个所有元素都是 1 的数组
    # 形状 (height, width, 3) 表示高×宽×3通道
    # dtype=np.uint8 指定数据类型为无符号8位整数（0-255）
    # 乘以 255 后，所有像素值变成 255，即白色（BGR 中 (255,255,255) = 白色）
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # --- 绘制彩色矩形 ---
    # cv2.rectangle(图像, 左上角坐标, 右下角坐标, BGR颜色, 线宽)
    # 坐标格式是 (x, y)，其中 x 是列号（横向），y 是行号（纵向）
    # 这和我们平时 "先行后列" 的直觉相反，需要注意！
    # 线宽 -1 表示填充整个矩形
    cv2.rectangle(image, (50, 50), (200, 180), (255, 0, 0), -1)    # 纯蓝色矩形 BGR=(255,0,0)
    cv2.rectangle(image, (220, 50), (370, 180), (0, 255, 0), -1)   # 纯绿色矩形 BGR=(0,255,0)
    cv2.rectangle(image, (390, 50), (540, 180), (0, 0, 255), -1)   # 纯红色矩形 BGR=(0,0,255)

    # --- 绘制圆形 ---
    # cv2.circle(图像, 圆心坐标, 半径, BGR颜色, 线宽)
    cv2.circle(image, (125, 280), 80, (255, 255, 0), -1)   # 青色填充圆 BGR=(255,255,0)
    cv2.circle(image, (320, 280), 80, (0, 255, 255), 3)    # 黄色空心圆（线宽3）
    cv2.circle(image, (515, 280), 80, (255, 0, 255), -1)   # 品红色填充圆

    # --- 绘制三角形 ---
    # cv2.polylines() 可以绘制多边形
    # 三角形需要三个顶点坐标，放在一个 numpy 数组中
    # 注意顶点坐标格式: (x, y)
    triangle_pts = np.array([[100, 380], [200, 460], [20, 460]], dtype=np.int32)
    # reshape(-1, 1, 2) 把数组变成 OpenCV 需要的形状：[点数, 1, 2]
    cv2.fillPoly(image, [triangle_pts.reshape(-1, 1, 2)], (0, 165, 255))  # 橙色三角形

    # --- 绘制文字 ---
    # cv2.putText(图像, 文字, 起始位置, 字体, 字号, 颜色, 线宽)
    cv2.putText(image, "OpenCV Learning", (130, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # --- 绘制渐变色条 ---
    # 这里展示如何直接操作 numpy 数组来创建渐变效果
    for i in range(256):
        # image[y1:y2, x1:x2] 这种切片写法可以访问图像的一个矩形区域
        # 这里在底部画一条从黑色渐变到彩色的色条
        color_val = i  # 0 到 255 的渐变值
        image[440:480, i * 2 + 40:i * 2 + 42] = (color_val, 128, 255 - color_val)

    return image


# =============================================================================
# 3. 图像显示工具
# =============================================================================

def show_image(title: str, image: np.ndarray, wait: bool = True):
    """
    在窗口中显示图像。
    这是一个封装函数，让显示图像的操作更简洁。

    参数:
        title: 窗口标题
        image: 要显示的图像（numpy 数组）
        wait:  True = 等待用户按键后关闭窗口
              False = 立即返回（用于连续显示多张图时）

    OpenCV 窗口操作原理:
        cv2.imshow() 创建或更新一个窗口
        cv2.waitKey() 等待键盘输入，这是必须的——没有它窗口不会刷新
        参数是等待的毫秒数: 0 = 无限等待, >0 = 等待指定毫秒
    """
    cv2.imshow(title, image)
    if wait:
        print(f"   📺 显示窗口: '{title}' — 按任意键关闭窗口继续...")
        cv2.waitKey(0)           # 等待用户按键
        cv2.destroyWindow(title) # 关闭当前窗口


def show_multiple_images(images: list[tuple[str, np.ndarray]], cols: int = 2):
    """
    在一个窗口中并排显示多张图片。
    用于对比不同处理效果——比如原图和滤波后的图放在一起看。

    参数:
        images: 列表，每个元素是 (标题, 图像) 的元组
        cols:   每行放几张图

    原理:
        用 np.hstack() 或 np.vstack() 把多张图片拼成一张大图，然后显示。
        如果图片大小不一致，会先调整到相同大小。
        如果图片通道数不一致（比如灰度图和彩色图混在一起），灰度图会先转成彩色。
    """
    if len(images) == 0:
        return

    # --- 获取第一张图片的尺寸作为统一尺寸 ---
    h, w = images[0][1].shape[:2]  # .shape 返回 (高度, 宽度) 或 (高度, 宽度, 通道)

    # --- 处理每张图片 ---
    processed = []
    for title, img in images:
        # 统一尺寸
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        # 如果是灰度图（2维），转为3通道彩色以便拼接
        if len(img.shape) == 2:
            # cv2.cvtColor() 用于颜色空间转换
            # COLOR_GRAY2BGR 将灰度图转成 BGR 三通道（但看起来还是灰色）
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 在图片上添加标题文字
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        processed.append(img)

    # --- 按行列拼接图片 ---
    rows = []
    for i in range(0, len(processed), cols):
        batch = processed[i:i + cols]
        # 如果这一行图片数不够 cols，用空白图片补齐
        while len(batch) < cols:
            blank = np.zeros_like(processed[0])
            batch.append(blank)
        # np.hstack() 水平拼接（horizontal stack）
        # 把同一行的多张图片左右拼在一起
        rows.append(np.hstack(batch))

    # np.vstack() 垂直拼接（vertical stack）
    # 把所有行上下拼在一起
    collage = np.vstack(rows)

    show_image("对比展示", collage)


# =============================================================================
# 4. 计时上下文管理器
# =============================================================================

class Timer:
    """
    计时器——用于测量代码块的执行时间。

    使用方式:
        with Timer("描述标签"):
            # 这里放你要计时的代码
            ...

    原理:
        __enter__ 在进入 with 块时执行，记录开始时间
        __exit__  在离开 with 块时执行，计算耗时并打印

    Python 的 with 语句 + 上下文管理器 = 优雅的资源管理
    """
    def __init__(self, label: str = "操作"):
        self.label = label

    def __enter__(self):
        # cv2.getTickCount() 返回 CPU 的时钟周期数
        # 它是 OpenCV 提供的高精度计时方式
        self.start = cv2.getTickCount()
        return self

    def __exit__(self, *args):
        end = cv2.getTickCount()
        # cv2.getTickFrequency() 返回 CPU 时钟频率（Hz，每秒的周期数）
        # (end - start) / frequency = 秒数
        elapsed = (end - self.start) / cv2.getTickFrequency()
        print(f"   ⏱  [{self.label}] 耗时: {elapsed*1000:.2f} ms")


# =============================================================================
# 5. 图片加载函数（带自动回退）
# =============================================================================

def load_image(filename: str = "sample.jpg") -> np.ndarray:
    """
    尝试从 data 目录加载图片。
    如果图片不存在，则自动生成一张测试图片。

    这是一个健壮的设计模式——"优雅降级"（Graceful Degradation）：
    即使依赖的外部资源不存在，程序也能正常运行。

    参数:
        filename: 期望加载的图片文件名

    返回:
        np.ndarray: 加载或生成的图片
    """
    path = get_data_path(filename)
    if os.path.exists(path):
        # 文件存在，使用 cv2.imread() 读取
        img = cv2.imread(path)
        if img is not None:
            print(f"✅ 成功加载图片: {path}")
            return img

    # 文件不存在或读取失败，生成测试图片并自动保存
    print(f"⚠️  未找到图片 {path}，自动生成测试图片")
    img = create_sample_image()
    # 自动保存到 data/ 目录，下次运行就不需要生成了
    try:
        data_dir = os.path.dirname(path)
        os.makedirs(data_dir, exist_ok=True)
        cv2.imwrite(path, img)
        print(f"   💾 已自动保存到: {path}")
    except Exception:
        pass  # 保存失败也不影响程序运行
    return img

# =============================================================================
# 🧱 基础篇 [1/3] - io_display.py
# 图像的读取、显示与保存（I/O = Input/Output）
# =============================================================================
# 这是 OpenCV 学习的起点。
# 学完本模块你应该理解：
#   1. 图像在计算机中是如何存储的（numpy 数组）
#   2. 如何用 OpenCV 读写图像文件
#   3. 如何创建窗口显示图像
#   4. BGR vs RGB 颜色顺序的区别
# =============================================================================

import cv2                     # OpenCV 库，所有 cv2.xxx 函数都来自这里
import numpy as np             # NumPy 库，OpenCV 的图像就是用 numpy 数组表示的
import os
import sys

# 把项目根目录加入 sys.path，这样就能 import utils 了
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.helpers import load_image, get_output_path, show_image


# =============================================================================
# 核心概念 1: 图像在 OpenCV 中是什么？
# =============================================================================
# OpenCV 中的图像就是一个 NumPy 多维数组 (ndarray):
#
#   - 彩色图像: shape = (高度, 宽度, 3)
#       第三个维度是 3 个"通道" (channel) —— B、G、R
#       ⚠️ 注意顺序是 BGR，不是我们常见的 RGB！
#       OpenCV 使用 BGR 是历史原因（早期摄像头厂商多用 BGR 格式）
#
#   - 灰度图像: shape = (高度, 宽度)
#       只有亮度信息，没有颜色，每个像素 0=黑, 255=白
#
#   - 像素值的数据类型通常是 uint8（无符号 8 位整数，范围 0-255）
#   - 图像的"原点"在左上角，x 向右增长，y 向下增长
#
#   (0,0) ──── x (列) ────→
#     │
#     │   坐标 (x, y) 即 (列号, 行号)
#     │   OpenCV 中绝大多数函数用 (x,y) 顺序
#     │   但 numpy 数组索引是 [行, 列] = [y, x]
#     y
#    (行)
#     ↓
# =============================================================================


def demo_image_creation():
    """
    演示1: 理解图像的本质——numpy 数组

    我们手动创建一个 numpy 数组，然后把它当作图像显示。
    这样你能最直观地理解"图像就是数组"这个概念。
    """
    print("\n--- 📖 演示1: 手动创建图像 ---")

    # --- 创建一个 300x400 的纯色图像 ---
    # np.zeros() 创建一个全 0 的三维数组
    # 形状 (300, 400, 3): 300行高, 400列宽, 3个通道(BGR)
    # dtype=np.uint8: 每个元素占1字节，范围0-255
    height, width = 300, 400
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # --- 理解坐标系统 ---
    # numpy 数组索引顺序是 [行(高), 列(宽), 通道]
    # img[行, 列] 访问的是第"行"第"列"的像素
    # img[行, 列, 通道] 访问特定通道的值：0=蓝色, 1=绿色, 2=红色

    # 把整个图像设成蓝色（B=255, G=0, R=0）
    img[:, :] = (255, 0, 0)    # BGR 顺序！蓝色分量255，其余为0
    # 上面这行的含义: img[所有行, 所有列] = BGR蓝色

    # --- 在图像上画一条红色线 ---
    # img[y=150, 全部列] = (0, 0, 255) → B=0, G=0, R=255 → 红色
    img[140:160, :] = (0, 0, 255)  # 行 140到160（高20像素的横条）

    # --- 在图像上画一条绿色竖线 ---
    # img[全部行, x=200] = (0, 255, 0) → B=0, G=255, R=0 → 绿色
    img[:, 190:210] = (0, 255, 0)  # 列 190到210（宽20像素的竖条）

    # --- 检查图像数组的关键属性 ---
    print(f"   图像形状 (shape):  {img.shape}")   # (300, 400, 3) = 高×宽×通道
    print(f"   图像维度 (ndim):   {img.ndim}")     # 3 = 三维数组
    print(f"   数据类型 (dtype):  {img.dtype}")    # uint8 = 0到255的整数
    print(f"   总像素数 (size):   {img.size}")     # 300*400*3 = 360000
    print(f"   单个像素值示例 img[10,10]: {img[10, 10]}")  # 某个像素的BGR三个值

    show_image("手动创建的 numpy 数组图像", img)
    cv2.destroyAllWindows()   # 关闭所有 OpenCV 窗口

    print("   💡 理解: 图像 = numpy 数组，每个元素 = 一个像素的一个通道值")


def demo_image_read():
    """
    演示2: 读取图像文件

    cv2.imread() 是最基本的图像读取函数。
    它会读取图片文件，解码后返回一个 numpy 数组。
    """
    print("\n--- 📖 演示2: 读取图像文件 ---")

    # --- 方法1: cv2.imread(路径, 模式) ---
    # 使用 load_image() 辅助函数（如果没有图片会自动生成测试图）
    img = load_image("sample.jpg")

    # --- 打印图像信息 ---
    # 这些属性对于调试和理解图像数据非常重要
    print(f"   图像尺寸 (宽×高): {img.shape[1]} x {img.shape[0]}")
    print(f"   通道数: {img.shape[2] if len(img.shape) > 2 else 1}")
    # 计算图像占用内存大小: 宽×高×通道×每像素字节数
    memory_kb = img.size / 1024
    print(f"   内存占用: {memory_kb:.1f} KB")

    # --- cv2.imread() 的模式参数 ---
    # cv2.IMREAD_COLOR (默认值=1):  读为彩色图 (BGR三通道)
    # cv2.IMREAD_GRAYSCALE (=0):   读为灰度图 (单通道)
    # cv2.IMREAD_UNCHANGED (=-1):  保持原样 (包括 Alpha 透明通道)
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample.jpg")
    if os.path.exists(path):
        img_color = cv2.imread(path, cv2.IMREAD_COLOR)      # 彩色模式
        img_gray  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 灰度模式

        print(f"\n   彩色图 shape: {img_color.shape}")  # (高, 宽, 3)
        print(f"   灰度图 shape: {img_gray.shape}")     # (高, 宽) — 只有2维!

    show_image("原始图像 (BGR彩色)", img)
    cv2.destroyAllWindows()


def demo_image_display():
    """
    演示3: 显示图像的多种方式

    学习如何创建窗口、设置窗口属性、在窗口中显示图像。
    """
    print("\n--- 📖 演示3: 显示图像 ---")

    img = load_image("sample.jpg")

    # --- 方式1: cv2.imshow() + cv2.waitKey() ---
    # 这是最基本的显示方式
    # cv2.imshow("窗口标题", 图像数组)  → 创建窗口并显示图像
    # cv2.waitKey(毫秒)                  → 等待键盘输入
    #   参数=0: 无限等待，直到用户按任意键
    #   参数>0: 等待指定毫秒数后自动关闭
    #   返回值: 用户按下的按键的 ASCII 码（没有按键时返回 -1）
    cv2.imshow("基本显示", img)
    print("   按任意键关闭窗口...")
    key = cv2.waitKey(0)  # 等待按键，返回按键的 ASCII 码
    print(f"   你按下的键的 ASCII 码: {key}")
    cv2.destroyAllWindows()

    # --- 方式2: 创建可调整大小的窗口 ---
    # cv2.WINDOW_NORMAL 允许用户用鼠标拖动窗口边缘来改变大小
    cv2.namedWindow("可调整窗口", cv2.WINDOW_NORMAL)
    # 设置窗口初始大小
    cv2.resizeWindow("可调整窗口", 800, 600)
    cv2.imshow("可调整窗口", img)
    print("   可以拖动窗口边缘调整大小，按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 方式3: 使用 matplotlib 显示（适合在 Jupyter 中使用）---
    # OpenCV 使用 BGR 颜色顺序，matplotlib 使用 RGB 顺序
    # 所以需要转换：BGR → RGB
    try:
        import matplotlib.pyplot as plt
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title("用 Matplotlib 显示（已转换 BGR→RGB）")
        plt.axis("off")  # 不显示坐标轴
        plt.show()
    except ImportError:
        print("   ⚠️ matplotlib 未安装，跳过此演示")


def demo_image_save():
    """
    演示4: 保存图像

    cv2.imwrite() 将图像保存为文件。
    支持的格式：JPG, PNG, BMP, TIFF 等（由文件扩展名决定）
    """
    print("\n--- 📖 演示4: 保存图像 ---")

    img = load_image("sample.jpg")

    # --- 保存为不同格式 ---
    # JPG: 有损压缩，文件小，不支持透明
    jpg_path = get_output_path("output_demo.jpg")
    success = cv2.imwrite(jpg_path, img)
    print(f"   保存 JPG: {'✅ 成功' if success else '❌ 失败'} → {jpg_path}")

    # PNG: 无损压缩，文件较大，支持透明
    png_path = get_output_path("output_demo.png")
    success = cv2.imwrite(png_path, img)
    print(f"   保存 PNG: {'✅ 成功' if success else '❌ 失败'} → {png_path}")

    # --- JPG 质量控制 ---
    # imwrite() 的第三个参数可以控制压缩质量
    # 对于 JPG: cv2.IMWRITE_JPEG_QUALITY 取值范围 0-100
    # 值越大质量越高，文件也越大
    for quality in [10, 50, 95]:
        q_path = get_output_path(f"output_quality_{quality}.jpg")
        # [cv2.IMWRITE_JPEG_QUALITY, quality] 是 OpenCV 传参的固定写法
        cv2.imwrite(q_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # os.path.getsize() 获取文件大小（字节）
        size_kb = os.path.getsize(q_path) / 1024
        print(f"   质量={quality:2d}: 文件大小={size_kb:.1f} KB")


def demo_pixel_access():
    """
    演示5: 像素的访问与修改

    学习如何读写图像中某个特定像素的值。
    这是理解图像处理的最底层操作。
    """
    print("\n--- 📖 演示5: 像素访问与修改 ---")

    img = load_image("sample.jpg").copy()  # .copy() 创建副本，以免修改原图
    h, w = img.shape[:2]  # 获取高度和宽度

    # --- 读取单个像素 ---
    # img[y, x] 返回该位置 BGR 三个通道的值
    center_y, center_x = h // 2, w // 2  # 图像中心点
    pixel_bgr = img[center_y, center_x]
    print(f"   图像中心点 ({center_x}, {center_y}) 的 BGR 值: {pixel_bgr}")
    print(f"      B (蓝色) = {pixel_bgr[0]}")
    print(f"      G (绿色) = {pixel_bgr[1]}")
    print(f"      R (红色) = {pixel_bgr[2]}")

    # --- 修改一个矩形区域 ---
    # 在图像左上角画一个 100x100 的红色方块
    img[0:100, 0:100] = (0, 0, 255)  # BGR = (0, 0, 255) = 红色
    # 解释: img[行0到100, 列0到100] 选中左上角100×100的区域
    #       赋值为 (B=0, G=0, R=255) 即纯红色

    # --- 修改一个圆形区域 ---
    # 用布尔索引实现——这是 numpy 的高级用法
    # 创建一个和图像同样大小的掩模（mask）
    mask = np.zeros((h, w), dtype=np.uint8)
    # 在掩模上画一个白色圆形
    cv2.circle(mask, (w // 2, h // 2), 80, 255, -1)
    # mask > 0 返回一个布尔数组，True 的地方就是圆形内部
    img[mask > 0] = (255, 0, 0)  # 圆形区域设为蓝色

    show_image("像素修改演示（红方块+蓝圆）", img)
    cv2.destroyAllWindows()

    print("   💡 像素操作 = numpy 数组索引操作，非常灵活！")


# =============================================================================
# 本节总结
# =============================================================================
# 关键概念回顾:
#   图像 = numpy 数组 (height, width, channels)
#   cv2.imread()  读取图像
#   cv2.imshow()  显示图像
#   cv2.imwrite() 保存图像
#   cv2.waitKey() 等待键盘输入（显示图像的必须步骤）
#   BGR 顺序 ≠ RGB 顺序（用 matplotlib 显示时别忘了转换！）
# =============================================================================

def main():
    """
    本模块的主函数——按顺序运行所有演示。
    """
    print("\n" + "="*60)
    print("  🧱 基础篇 [1/3]: 图像读取、显示与保存")
    print("="*60)

    demo_image_creation()   # 演示1: 创建图像
    demo_image_read()       # 演示2: 读取图像
    demo_image_display()    # 演示3: 显示图像
    demo_image_save()       # 演示4: 保存图像
    demo_pixel_access()     # 演示5: 像素操作

    print("\n✅ 基础篇 [1/3] 完成！")


if __name__ == "__main__":
    main()

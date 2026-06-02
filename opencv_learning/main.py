# =============================================================================
# 🚀 main.py - Python + OpenCV 学习项目主入口
# =============================================================================
# 本文件是整个学习项目的入口程序。
# 运行后可以看到一个菜单，输入编号即可进入对应的学习模块。
# 每个模块都可以独立运行，也通过本菜单统一管理。
# =============================================================================

# ---------- 导入 Python 内置模块 ----------
import sys          # 系统相关功能：获取 Python 版本、退出程序等
import os           # 操作系统接口：处理文件路径

# Windows 控制台编码修复（必须在任何 print 之前执行）
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------- 将 src 目录添加到 Python 的模块搜索路径 ----------
# sys.path 是一个列表，存放着 Python 搜索模块时所有会去查找的目录
# os.path.dirname(__file__) 获取当前文件(main.py)所在的目录
# os.path.abspath() 将其转为绝对路径（例如 "D:/zhy/ai/git/opencv_learning"）
project_root = os.path.abspath(os.path.dirname(__file__))
# 把项目根目录加入搜索路径，这样我们就可以 import src.xxx 了
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def print_banner():
    """
    打印程序启动横幅。
    """
    banner = r"""
╔══════════════════════════════════════════════════════════╗
║     🎯  Python + OpenCV 计算机视觉学习项目              ║
║     从零开始，逐行注释，实战驱动                         ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """
    打印学习模块菜单。
    每个模块对应一个编号，用户输入编号即可运行对应模块。
    """
    menu = """
┌────────────────────────────────────────────────────────┐
│                   📚 学习模块菜单                        │
├──────┬─────────────────────────────────────────────────┤
│ 编号 │ 模块名称                  所属类别               │
├──────┼─────────────────────────────────────────────────┤
│  1   │ 图像读写与显示            🧱 基础篇              │
│  2   │ 基本图像操作               🧱 基础篇              │
│  3   │ 颜色空间转换               🧱 基础篇              │
│  4   │ 图像滤波                   🔧 处理篇              │
│  5   │ 边缘检测                   🔧 处理篇              │
│  6   │ 形态学操作                 🔧 处理篇              │
│  7   │ 阈值化技术                 🔧 处理篇              │
│  8   │ 轮廓分析                   🔍 分析篇              │
│  9   │ 直方图分析                 🔍 分析篇              │
│ 10   │ 特征检测与匹配             🔍 分析篇              │
│ 11   │ 人脸检测                   🏆 应用篇              │
│ 12   │ 目标跟踪                   🏆 应用篇              │
│ 13   │ 图像拼接（全景图）          🏆 应用篇              │
│  0   │ 退出程序                                          │
├──────┼─────────────────────────────────────────────────┤
│  a   │ 按顺序运行全部模块（学习路线）                    │
└──────┴─────────────────────────────────────────────────┘
    """
    print(menu)


def run_module(module_id):                              
    """
    根据用户输入的编号，运行对应的学习模块。

    参数:
        module_id (str): 用户输入的模块编号（字符串类型，因为 input() 返回字符串）

    每个模块都是一个 .py 文件，我们通过动态导入来运行它。
    每个模块文件里都有一个 main() 函数作为入口。
    """
    # --- 模块映射表 ---
    # 字典 (dict) 是 Python 中最常用的数据结构之一
    # 这里 key 是编号，value 是 (模块名, 导入路径, 描述)
    modules = {
        "1":  ("图像读写与显示",   "src.basics.io_display",        "🧱 基础篇"),
        "2":  ("基本图像操作",     "src.basics.basic_ops",         "🧱 基础篇"),
        "3":  ("颜色空间转换",     "src.basics.color_spaces",      "🧱 基础篇"),
        "4":  ("图像滤波",         "src.processing.filters",       "🔧 处理篇"),
        "5":  ("边缘检测",         "src.processing.edge_detection","🔧 处理篇"),
        "6":  ("形态学操作",       "src.processing.morphological", "🔧 处理篇"),
        "7":  ("阈值化技术",       "src.processing.thresholding",  "🔧 处理篇"),
        "8":  ("轮廓分析",         "src.analysis.contours",        "🔍 分析篇"),
        "9":  ("直方图分析",       "src.analysis.histograms",      "🔍 分析篇"),
        "10": ("特征检测与匹配",   "src.analysis.feature_detection","🔍 分析篇"),
        "11": ("人脸检测",         "src.applications.face_detection","🏆 应用篇"),
        "12": ("目标跟踪",         "src.applications.object_tracking","🏆 应用篇"),
        "13": ("图像拼接",         "src.applications.image_stitching","🏆 应用篇"),
    }

    # --- 检查用户输入 ---
    if module_id not in modules:
        print(f"❌ 无效的编号: {module_id}，请输入 0-13 或 a")
        return

    # --- 获取模块信息 ---
    name, import_path, category = modules[module_id]

    # --- 打印分隔线和模块标题 ---
    print(f"\n{'='*60}")
    print(f"  ▶ 正在运行: {category} - {name}")
    print(f"{'='*60}\n")

    # --- 动态导入模块 ---
    try:
        # importlib.import_module() 可以根据字符串路径动态导入模块
        # 比如 importlib.import_module("src.basics.io_display")
        # 效果等同于 import src.basics.io_display
        import importlib
        module = importlib.import_module(import_path)

        # 每个模块文件都定义了 main() 函数，我们直接调用它
        if hasattr(module, "main"):
            module.main()
        else:
            print(f"⚠️  模块 {name} 没有 main() 函数，请直接查看源代码学习")
    except ImportError as e:
        # import 失败时的错误处理
        print(f"❌ 导入模块失败: {e}")
        print(f"   请确保文件路径正确: {import_path}.py")
    except Exception as e:
        # 其他任何运行时错误的处理
        print(f"❌ 运行模块时出错: {type(e).__name__}: {e}")


def run_all_modules():
    """
    按顺序依次运行所有 13 个模块。
    这是推荐的学习路线，从基础到应用循序渐进。
    """
    print("\n🚀 开始按顺序运行全部模块...\n")
    # 遍历编号 1 到 13
    for i in range(1, 14):
        module_id = str(i)  # 将整数转为字符串，因为 run_module 的参数是字符串
        print(f"\n{'#'*60}")
        print(f"  📖 第 {i}/13 个模块")
        print(f"{'#'*60}")
        run_module(module_id)
        # 每个模块运行完后，等待用户按回车继续
        # 这样用户可以逐个查看输出结果
        input(f"\n按 Enter 键继续下一个模块...")

    print("\n✅ 全部模块运行完毕！祝你学习愉快！")


def main():
    """
    程序的主函数。
    这是整个程序的入口——当你运行 python main.py 时，这个函数首先被执行。

    函数的工作流程：
    1. 打印欢迎横幅
    2. 进入无限循环，等待用户输入
    3. 根据用户输入执行对应的模块
    4. 用户输入 0 时退出循环
    """
    # --- 第一步：打印欢迎横幅 ---
    print_banner()

    # --- 第二步：进入主循环 ---
    # while True 创建了一个无限循环
    # 程序会一直运行，直到用户选择退出（输入 0）
    while True:
        # 打印菜单
        print_menu()

        # --- 获取用户输入 ---
        # input() 函数会暂停程序，等待用户在终端输入并按下回车
        # strip() 方法去掉输入首尾的空白字符（空格、换行等）
        choice = input("请输入模块编号 (0-13, a=全部): ").strip()

        # --- 根据用户输入做不同的事情 ---
        if choice == "0":
            # 用户选择退出
            print("\n👋 再见！祝你学习愉快，编码顺利！")
            break  # break 语句跳出 while 循环

        elif choice.lower() == "a":
            # 用户选择按顺序运行全部模块
            # .lower() 方法将字符串转为小写，这样输入 A 或 a 都可以
            run_all_modules()

        else:
            # 用户输入了一个模块编号
            run_module(choice)

        # 每个操作后打印一个空行，让输出更清晰
        print()


# =============================================================================
# 这段代码是 Python 的常见写法：
# if __name__ == "__main__" 的意思是：
#   "如果这个文件是被直接运行的（而不是被其他文件 import 导入的）"
# 当 python main.py 时，__name__ 变量的值就是 "__main__"，条件成立，执行 main()
# 当 import main 时，__name__ 的值是 "main"，条件不成立，不执行 main()
# 这样既可以把 main.py 当作主程序运行，也可以被其他代码导入而不自动执行
# =============================================================================
if __name__ == "__main__":
    main()

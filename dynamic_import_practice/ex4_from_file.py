"""
练习4（挑战）：从文件路径加载模块
任务：动态加载一个不在 plugins 文件夹里的 .py 文件

假设桌面上有一个 mystery.py，你不知道它里面有什么，
但你知道它有一个 run() 函数。写代码加载并运行它。

提示：用 importlib.util.spec_from_file_location 和 module_from_spec
"""
import importlib.util
import sys

# 这里填你要加载的 .py 文件路径
file_path = input("请输入 .py 文件的完整路径：")

# 1. 根据文件路径创建 spec（规格说明）
spec = importlib.util.spec_from_file_location("mystery", file_path)

# 2. 根据 spec 创建一个空的模块
?

# 3. 把模块注册到 sys.modules（系统的模块表），这样模块内部的 import 才能正常工作
?

# 4. 真正执行模块里的代码
?

# 5. 调用模块的 run()
?

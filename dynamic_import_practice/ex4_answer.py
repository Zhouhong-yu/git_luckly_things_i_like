"""
练习4 答案：从文件路径加载模块
"""
import importlib.util
import sys

file_path = input("请输入 .py 文件的完整路径：")

# 1. 创建 spec
spec = importlib.util.spec_from_file_location("mystery", file_path)

# 2. 创建空模块
module = importlib.util.module_from_spec(spec)

# 3. 注册到系统模块表
sys.modules["mystery"] = module

# 4. 执行模块代码
spec.loader.exec_module(module)

# 5. 调用 run()
module.run()

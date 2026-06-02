"""
练习1 答案：最基础的动态导入
"""
import importlib

module_name = input("请输入模块名（hello / goodbye / counter）：")

# "plugins.hello" = "plugins." + "hello"
full_name = "plugins." + module_name
mod = importlib.import_module(full_name)

mod.run()
print("描述：", mod.description())

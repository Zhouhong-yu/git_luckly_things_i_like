"""
练习2：自动发现所有插件
任务：不靠用户输入，自动找出 plugins 文件夹下所有插件，列出名字和描述

提示：
  - 先用 importlib.import_module("plugins") 加载包
  - 包.__path__ 是插件文件夹的路径
  - pkgutil.iter_modules(路径) 可以遍历文件夹里的所有 .py 文件
"""
import importlib
import pkgutil

# 1. 加载 plugins 包
?

# 2. 遍历包里的所有模块
for 遍历结果 in pkgutil.iter_modules(?):
    # 遍历结果.name 就是模块名，比如 "hello"
    module_name = ?
    full_name = "plugins." + module_name

    # 3. 动态加载
    mod = ?

    # 4. 打印名字和描述
    ?

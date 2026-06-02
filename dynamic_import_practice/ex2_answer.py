"""
练习2 答案：自动发现所有插件
"""
import importlib
import pkgutil

# 1. 加载包
package = importlib.import_module("plugins")

# 2. 遍历包里所有 .py 文件
for finder, name, is_pkg in pkgutil.iter_modules(package.__path__):
    if is_pkg:
        continue  # 跳过子文件夹

    full_name = "plugins." + name
    mod = importlib.import_module(full_name)

    print(f"[{name}] {mod.description()}")

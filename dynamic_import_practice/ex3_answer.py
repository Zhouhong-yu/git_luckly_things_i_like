"""
练习3 答案：插件菜单
"""
import importlib
import pkgutil

# 1. 发现并加载所有插件
plugins = []

package = importlib.import_module("plugins")
for finder, name, is_pkg in pkgutil.iter_modules(package.__path__):
    if is_pkg:
        continue
    mod = importlib.import_module("plugins." + name)
    plugins.append(mod)

# 2. 循环显示菜单
while True:
    print("\n=== 可用插件 ===")
    for i, mod in enumerate(plugins):
        # mod.__name__ 形如 "plugins.hello"，取最后一段
        short_name = mod.__name__.split(".")[-1]
        print(f"  {i + 1}. {short_name} - {mod.description()}")
    print("===")

    choice = input("请选择（输入编号，q 退出）：")

    if choice == "q":
        print("拜拜！")
        break

    # 3. 检查输入是否合法
    if not choice.isdigit():
        print("请输入数字！")
        continue

    idx = int(choice) - 1
    if idx < 0 or idx >= len(plugins):
        print("没有这个编号！")
        continue

    # 4. 运行选中的插件
    plugins[idx].run()

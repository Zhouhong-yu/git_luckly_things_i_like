"""
练习3：做一个简单的插件菜单
任务：结合练习1和2，让用户从菜单里选插件来运行

效果：
  === 可用插件 ===
  1. hello  - 一个会打招呼的插件
  2. goodbye - 一个会说再见的插件
  3. counter - 一个会计数的插件
  ===
  请选择（输入编号，q 退出）：

提示：
  - 先用练习2的方法，把发现到的插件存到一个列表里
  - 显示菜单
  - 用户选几号，就运行列表里第几个插件的 run()
  - 选 q 就退出
"""
import importlib
import pkgutil

# 1. 发现并加载所有插件，存到列表里
plugins = []  # 用来放加载好的模块

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
        print(f"  {i + 1}. {mod.__name__} - {mod.description()}")
    print("===")

    choice = input("请选择（输入编号，q 退出）：")

    if choice == "q":
        print("拜拜！")
        break

    # 3. 把输入转成数字，注意检查输入是否合法
    ?

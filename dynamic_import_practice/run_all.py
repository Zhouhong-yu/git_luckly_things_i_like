"""
一键运行所有练习的答案
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(__file__))

exercises = [
    ("ex1_answer.py", "练习1：基础动态导入"),
    ("ex2_answer.py", "练习2：自动发现插件"),
    ("ex3_answer.py", "练习3：插件菜单"),
    ("ex4_answer.py", "练习4：从文件路径加载"),
]

for filename, desc in exercises:
    print(f"\n{'=' * 50}")
    print(f"  {desc}")
    print(f"{'=' * 50}")
    print(f"运行: python {filename}")
    print(f"（需要手动交互，请自己跑一下这个文件）")

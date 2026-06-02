# 插件：计数器
count = 0


def run():
    global count
    count += 1
    print(f"我被调用了 {count} 次！")


def description():
    return "一个会计数的插件"

while True:
    try:
        # 获取用户输入的浮点数
        num = float(input("请输入一个浮点数: "))
        # 计算 (输入 - 1) * 25
        result = (num - 1) * 25
        # 输出结果
        print(f"结果是: {result}")
    except ValueError:
        print("输入错误，请确保输入的是一个有效的数字。")
    except EOFError:
        # 处理Ctrl+D（Unix）或Ctrl+Z（Windows）等退出信号
        print("\n程序已结束")
        break
    except KeyboardInterrupt:
        # 处理Ctrl+C中断
        print("\n程序被用户中断")
        break
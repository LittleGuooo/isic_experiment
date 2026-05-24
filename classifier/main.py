from .trainer import run_test, run_training
from .config import parse_args


def main():
    """
    解析命令行参数，并根据 test_only 选择训练或测试流程
    """
    args = parse_args()

    if args.test_only:
        run_test(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()

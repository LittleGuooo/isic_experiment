from .config import parse_args, validate_args
from .runtime import run_train


if __name__ == "__main__":
    # 1) 解析命令行参数
    args = parse_args()

    # 2) 对参数做一致性检查
    validate_args(args)

    # 3) 进入训练 / 评估 / 推理主流程
    run_train(args)

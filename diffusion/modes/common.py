from .ldm_ae import build_ldm_ae
from .ddpm import build_ddpm
from .cfg import build_cfg
from .cg import build_cg
from .ldm import build_latent_ddpm


def get_modes(args):
    # 根据命令行参数 args.mode 选择不同的扩散模式
    mode = getattr(args, "mode", "ddpm")

    if mode == "ddpm":
        return build_ddpm(args)
    if mode == "cfg":
        return build_cfg(args)
    if mode == "cg":
        return build_cg(args)
    if args.mode == "ldm_ae":
        return build_ldm_ae(args)
    if args.mode == "latent_ddpm":
        return build_latent_ddpm(args)

    # 这里做兜底，避免传入未支持的 mode
    raise ValueError(f"Unsupported mode: {mode}")

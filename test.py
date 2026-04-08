import torch

ckpt = torch.load(
    "experiments/20260406_003627_ddpm_cond_all_all_labels_res128_bs32_seed42/checkpoints/last.pth.tar",
    map_location="cpu",
)
args = ckpt.get("args", {})
sd = ckpt.get("model_state_dict", {})
print("保存时的 use_class_conditioning:", args.get("use_class_conditioning"))
print("保存时的 num_classes:", args.get("num_classes"))
print("保存时的 resolution:", args.get("resolution"))
for k in ["conv_in.weight", "class_embedding.weight"]:
    if k in sd:
        print(f"{k}: {sd[k].shape}")
    else:
        print(f"{k}: 不存在")

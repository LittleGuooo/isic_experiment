# sd_full_finetune_isic.py
# 作用：
#   使用 Hugging Face diffusers 对 Stable Diffusion v1-5 做 Full UNet fine-tuning。
#   这里的 Full fine-tuning 指：训练 UNet 的全部参数；冻结 VAE 和 Text Encoder。
#
# 数据要求：
#   1. ISIC2018_Task3_Training_GroundTruth.csv
#   2. ISIC2018_Task3_Training_Input/
#
# CSV 格式应类似：
#   image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
#   ISIC_0024306,0,1,0,0,0,0,0
#
# 运行示例：
#   accelerate launch --mixed_precision fp16 sd_full_finetune_isic.py ^
#     --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 ^
#     --train_csv dataset/ISIC2018_Task3_Training_GroundTruth.csv ^
#     --image_dir dataset/ISIC2018_Task3_Training_Input ^
#     --output_dir experiments/sd_full_isic_unet ^
#     --resolution 512 ^
#     --train_batch_size 1 ^
#     --gradient_accumulation_steps 4 ^
#     --max_train_steps 3000 ^
#     --learning_rate 1e-5

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

ISIC_CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

ISIC_PROMPTS = {
    "MEL": "a dermoscopic image of melanoma",
    "NV": "a dermoscopic image of melanocytic nevus",
    "BCC": "a dermoscopic image of basal cell carcinoma",
    "AKIEC": "a dermoscopic image of actinic keratosis or intraepithelial carcinoma",
    "BKL": "a dermoscopic image of benign keratosis-like lesion",
    "DF": "a dermoscopic image of dermatofibroma",
    "VASC": "a dermoscopic image of vascular lesion",
}


class ISICTextToImageDataset(Dataset):
    """
    ISIC2018 -> Stable Diffusion text-to-image fine-tuning dataset.

    每个样本返回：
        pixel_values: 图像张量，shape = [3, H, W]，范围 [-1, 1]
        input_ids: prompt token ids，shape = [tokenizer.model_max_length]
    """

    def __init__(self, train_csv, image_dir, tokenizer, resolution=512):
        self.train_csv = Path(train_csv)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                # Stable Diffusion 的 VAE 输入约定为 [-1, 1]
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.samples = []
        with open(self.train_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_id = row["image"]

                label_name = None
                for cls_name in ISIC_CLASS_NAMES:
                    if int(float(row[cls_name])) == 1:
                        label_name = cls_name
                        break

                if label_name is None:
                    continue

                # ISIC 官方图片通常是 .jpg
                image_path = self.image_dir / f"{image_id}.jpg"

                self.samples.append(
                    {
                        "image_path": image_path,
                        "label_name": label_name,
                        "prompt": ISIC_PROMPTS[label_name],
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No training samples found. Please check csv={train_csv} and image_dir={image_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.transform(image)

        tokenized = self.tokenizer(
            item["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Stable Diffusion 预训练模型名称或本地路径。",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="dataset/ISIC2018_Task3_Training_GroundTruth.csv",
        help="ISIC2018 Task3 训练集标签 CSV。",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="dataset/ISIC2018_Task3_Training_Input",
        help="ISIC2018 Task3 训练图片目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/sd_full_isic_unet",
        help="保存微调后 Stable Diffusion pipeline 的目录。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Stable Diffusion v1 系列默认使用 512。",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="单卡 5060 Ti 建议先用 1。",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数。batch_size=1 且累积 4 步，等效 batch size 为 4。",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help="先跑通建议 3000；不是最终最优训练步数。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Full UNet fine-tuning 常用较小学习率。",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="学习率调度器名称，传给 diffusers get_scheduler。",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="如果已正确安装 xformers，可开启显存优化。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=accelerator_mixed_precision_from_env(),
    )

    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # mixed_precision="fp16" 时，冻结模块用 fp16 可以节省显存。
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 1. 加载 tokenizer / text encoder / VAE / UNet / noise scheduler
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # 2. 冻结 VAE 和 Text Encoder，只训练 UNet 全部参数
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # 3. 显存优化：gradient checkpointing
    unet.enable_gradient_checkpointing()

    # 4. 可选显存优化：xFormers memory efficient attention
    if args.enable_xformers:
        unet.enable_xformers_memory_efficient_attention()

    # 5. 冻结模块放到 GPU，并转为 fp16/bf16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # 6. 数据集
    train_dataset = ISICTextToImageDataset(
        train_csv=args.train_csv,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        resolution=args.resolution,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 7. 优化器：Full fine-tuning 直接优化 UNet 全部参数
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 8. 交给 accelerate 管理
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Stable Diffusion Full UNet Fine-tuning",
    )

    global_step = 0

    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # pixel_values: [-1, 1], shape = [B, 3, 512, 512]
                pixel_values = batch["pixel_values"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                )

                input_ids = batch["input_ids"].to(accelerator.device)

                # 9. 图像编码到 latent space
                # Stable Diffusion 的 VAE latent 需要乘 scaling_factor
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    encoder_hidden_states = text_encoder(input_ids)[0]

                # 10. 随机采样噪声和 timestep
                noise = torch.randn_like(latents)

                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()

                # 11. 前向扩散：给 latent 加噪
                noisy_latents = noise_scheduler.add_noise(
                    latents,
                    noise,
                    timesteps,
                )

                # 12. UNet 预测噪声
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # 13. 训练目标：epsilon prediction 的 MSE loss
                loss = F.mse_loss(
                    model_pred.float(),
                    noise.float(),
                    reduction="mean",
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.detach().item():.6f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()

    # 14. 保存完整 Stable Diffusion pipeline
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unwrapped_unet,
            torch_dtype=weight_dtype,
        )

        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


def accelerator_mixed_precision_from_env():
    """
    accelerate launch --mixed_precision fp16 会把 mixed precision 配置交给 Accelerator。
    这里不手动写死 fp16，避免和 accelerate 配置冲突。
    """
    return None


if __name__ == "__main__":
    main()

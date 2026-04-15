# ISIC2018 Diffusion Training Project API 文档

## 1. 程序总体目标

本项目面向 **ISIC 2018 皮肤镜图像数据集（ISIC2018 dermoscopy images）**，目标是使用 **扩散模型（Diffusion Model）** 完成图像生成训练与评估，并支持以下三种模式：

- **DDPM（Denoising Diffusion Probabilistic Model）**：基础扩散训练与采样
- **CFG（Classifier-Free Guidance）**：通过无分类器引导提升条件采样质量
- **CG（Classifier Guidance）**：预留分类器引导接口，与外部分类器集成

项目当前支持的主要能力：

- 读取 ISIC2018 训练集与验证集
- 训练 unconditional / class-conditional 扩散模型
- 保存训练样本、checkpoint、Diffusers pipeline
- 运行 `val_only` 评估模式
- 运行 `infer_only` 生成模式
- 计算生成质量指标：
  - **FID（Fréchet Inception Distance）**
  - **KID（Kernel Inception Distance）**
  - **IPR（Improved Precision / Recall）**

---

## 2. 脚本目录结构

```text
project_root/
├── main.py
├── config.py
├── data.py
├── modeling.py
├── runtime.py
├── metrics.py
├── utils.py
├──modes/
   ├── __init__.py
   ├── common.py
   ├── ddpm.py
   ├── cfg.py
   └── cg.py
```

### 目录角色说明

- `main.py`：程序入口
- `config.py`：命令行参数定义与参数校验
- `data.py`：数据集、图像预处理、DataLoader 构建
- `modeling.py`：UNet、scheduler、采样循环、pipeline 构建
- `runtime.py`：训练、验证、推理主流程
- `metrics.py`：FID / KID / IPR 相关评估逻辑
- `utils.py`：实验目录、JSON/CSV 保存、checkpoint、元数据同步等工具函数
- `modes/ddpm.py`：DDPM 模式逻辑
- `modes/cfg.py`：CFG 模式逻辑
- `modes/cg.py`：CG 模式逻辑
- `modes/common.py`：根据 `args.mode` 分发模式实现
- `modes/__init__.py`：导出 `get_modes`

---

## 3. 实验文件夹目录结构

### 3.1 训练实验目录

由 `setup_experiment_folders(base_dir, exp_name)` 创建。

```text
experiments/
└── <experiment_name>/
    ├── checkpoints/
    │   ├── last.pth.tar
    │   ├── model_best.pth.tar
    │   ├── epoch_XXX.pth.tar
    │   ├── classifier_last.pth.tar
    │   ├── classifier_best.pth.tar
    │   └── cg_classifier_best.pth.tar
    ├── metrics/
    │   ├── epoch_metrics.csv
    │   ├── epoch_metrics.json
    │   └── cg_classifier_history.json
    ├── metadata/
    │   ├── experiment_metadata.json
    │   └── diffusers_pipeline_model_index.json
    ├── samples/
    │   └── epoch_XXX/
    │       └── sample_XXX[_CLASS].png
    ├── fid/
    │   ├── epoch_XXX_train_fid.json
    │   ├── epoch_XXX_val_fid.json
    │   └── epoch_XXX_train_per_class_metrics.json
    ├── fid_generated_images/
    │   └── epoch_XXX_<split>_generated/
    │       └── <CLASS>/
    │           └── fid_sample_XXXXX.png
    └── model_index.json
```

### 3.2 运行时目录（仅验证 / 仅推理）

由 `setup_runtime_run_folders(exp_dir, run_mode, run_name)` 创建。

#### `val_only`

```text
<experiment_dir>/
└── run_vals/
    └── <run_name>/
        ├── metrics/
        ├── generated_images/
        ├── metadata/
        ├── run_config.json
        └── run_summary.json
```

#### `infer_only`

```text
<experiment_dir>/
└── run_infers/
    └── <run_name>/
        ├── metrics/
        ├── generated_images/
        │   └── generated_<LABEL or unconditional>/
        │       └── sample_XXXXX[_LABEL].png
        ├── metadata/
        ├── run_config.json
        └── run_summary.json
```

---

## 4. API 说明

---

## 4.1 `main.py`

### `main.py` 职责
程序入口。负责：

1. 解析命令行参数
2. 执行参数校验
3. 进入训练主流程

### 函数与调用说明

#### 顶层执行逻辑
```python
args = parse_args()
validate_args(args)
run_train(args)
```

- `parse_args()`：来自 `config.py`
- `validate_args(args)`：来自 `config.py`
- `run_train(args)`：来自 `runtime.py`

---

## 4.2 `config.py`

### `parse_args()`
**作用**：定义所有命令行参数并返回 `argparse.Namespace`。

**主要参数分组：**

#### 模式相关
- `--mode`：运行模式，取值 `ddpm / cfg / cg`
- `--cfg_scale`：CFG 采样 guidance scale
- `--cond_drop_prob`：CFG 训练阶段 label dropout 概率
- `--classifier_ckpt_path`：CG 分类器 checkpoint 路径
- `--classifier_guidance_scale`：CG guidance scale
- `--resnet_time_scale_shift`：UNet 中 ResNet block（残差块，ResNet block）的时间嵌入融合方式，取值 `default / scale_shift`

#### CG 分类器训练相关
- `--cg_diffusion_ckpt_path`：CG 模式下，指定一个已训练好的 diffusion checkpoint；若提供，则运行时跳过 diffusion 训练，直接基于该 checkpoint 构建并训练 guidance classifier（引导分类器）
- `--classifier_train_epochs`：CG classifier 训练轮数
- `--classifier_train_lr`：CG classifier 学习率
- `--classifier_train_batch_size`：CG classifier 训练 batch size；默认复用 `train_batch_size`

#### 运行方式相关
- `--resume_from_checkpoint`：恢复训练或验证/推理时使用的 checkpoint
- `--run_mode`：`train / val_only / infer_only`
- `--infer_label`：`infer_only` 时指定生成类别
- `--infer_num_images`：`infer_only` 时生成图像数量

#### 采样相关
- `--use_ddim_sampling`：是否使用 DDIM 采样
- `--ddim_eta`：DDIM 的随机性系数

#### 数据相关
- `--train_gt_csv_path`
- `--val_gt_csv_path`
- `--train_img_dir`
- `--val_img_dir`
- `--data_mode`：`all / single_label`
- `--target_label`：单类别训练时的类别
- `--exclude_train_nv`：若开启，则仅在训练集构建时剔除 NV 类样本；验证集不受影响

#### 条件控制相关
- `--use_class_conditioning`：是否启用类别条件

#### 训练超参数
- `--output_root`
- `--resolution`
- `--train_batch_size`
- `--eval_batch_size`
- `--dataloader_num_workers`
- `--num_epochs`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--adam_beta1`
- `--adam_beta2`
- `--adam_weight_decay`
- `--adam_epsilon`
- `--lr_scheduler`
- `--lr_warmup_steps`
- `--mixed_precision`
- `--use_ema`：训练时启用 EMA 权重；评估 / 推理 / 保存时优先使用 EMA 权重
- `--ema_decay`：EMA 衰减系数

#### 扩散相关
- `--ddpm_num_steps`
- `--ddpm_num_inference_steps`
- `--ddpm_beta_schedule`

#### 评估相关
- `--enable_per_class_metrics`
- `--save_images_epochs`
- `--save_model_epochs`
- `--eval_epochs`
- `--num_fid_samples_train`
- `--num_fid_samples_val / --num_fid_samples_valid`：用于计算验证集 FID 的生成图片数量；二者映射到同一参数 `num_fid_samples_val`
- `--ipr_k`

#### 其他
- `--seed`

---

### `validate_args(args)`
**作用**：对命令行参数进行一致性校验。

**当前校验规则：**

- `run_mode in [val_only, infer_only]` 时必须提供 `resume_from_checkpoint`
- `data_mode == single_label` 时必须提供 `target_label`
- 若开启 `exclude_train_nv`，则 `single_label + target_label in [NV, 1]` 非法，因为训练集会被剔空
- `infer_only` 时 `infer_num_images > 0`
- `mode in [cfg, cg]` 时必须启用 `use_class_conditioning`
- `cfg` 模式下 `cond_drop_prob` 必须在 `[0, 1)` 范围内
- `cg` 模式下，只有 `val_only / infer_only` 才强制要求提供 `classifier_ckpt_path`
- `infer_only + use_class_conditioning=True` 时必须提供 `infer_label`
- `infer_only + use_class_conditioning=False` 时不应提供 `infer_label`

---

## 4.3 `data.py`

### `ISIC2018DDPMDataset`
**作用**：读取 ISIC2018 CSV 标注与图片，输出扩散训练所需样本。

#### `__init__(gt_csv_path, img_dir, transform=None, data_mode="all", target_label=None)`
- 读取 CSV
- 提取类别列
- 通过 one-hot 标注计算 `label_int`
- 支持：
  - `all`：保留全部类别
  - `single_label`：仅保留单个类别

**成员变量：**
- `self.img_dir`
- `self.transform`
- `self.data_mode`
- `self.target_label`
- `self.class_columns`
- `self.df`
- `self.labels`
- `self.selected_label_idx`
- `self.selected_label_name`

#### `__len__()`
返回数据集样本数。

#### `__getitem__(idx)`
返回单个样本：

```python
{
    "input": image_tensor,
    "label": label_int,
    "sample_id": image_id,
}
```

---

### `build_image_transforms(resolution)`
**作用**：构建图像预处理流程。

**流程：**
1. `Resize` 到固定分辨率
2. `ToTensor`
3. `Normalize([0.5], [0.5])` 到 `[-1, 1]` 范围

---

### `normalize_label_to_index_and_name(target_label, class_names)`
**作用**：把类别输入统一转换为 `(类别索引, 类别名)`。

支持：
- 字符串类别名，如 `"MEL"`
- 数字字符串索引，如 `"0"`

返回：
```python
(target_idx, target_name)
```

---

### `build_datasets_and_loaders(args)`
**作用**：统一构建训练集、验证集、DataLoader 和类别统计信息。

**返回字典字段：**
- `train_dataset`
- `val_dataset`
- `train_dataloader`
- `train_eval_loader`
- `val_eval_loader`
- `class_names`
- `num_classes`
- `train_class_distribution`
- `val_class_distribution`

---

## 4.4 `modeling.py`

### `build_model(args, num_classes)`
**作用**：构建 Diffusers 的 `UNet2DModel`。

**关键点：**
- 输入输出通道均为 3
- `sample_size = args.resolution`
- `num_class_embeds`：
  - unconditional：`None`
  - class-conditional DDPM / CG：`num_classes`
  - CFG：`num_classes + 1`（额外留一个 null class）
- `resnet_time_scale_shift`：由 `args.resnet_time_scale_shift` 控制，传入 Diffusers `UNet2DModel`

---

### `build_noise_scheduler(args)`
**作用**：构建训练阶段使用的 `DDPMScheduler`。

**配置来源：**
- `num_train_timesteps = args.ddpm_num_steps`
- `beta_schedule = args.ddpm_beta_schedule`
- `prediction_type = "epsilon"`

---

### `build_sampling_scheduler(noise_scheduler, use_ddim_sampling=False)`
**作用**：根据配置返回采样 scheduler。

返回：
- `DDIMScheduler.from_config(...)`
- 或 `DDPMScheduler.from_config(...)`

---

### `run_sampling_loop(...)`
**作用**：统一实现逐步去噪采样循环。

**输入核心参数：**
- `model`
- `sampling_scheduler`
- `device`
- `resolution`
- `batch_size`
- `num_inference_steps`
- `generator`
- `predict_fn`
- `ddim_eta`
- `return_pil_safe_uint8`

**流程：**
1. 设置推理步数 `set_timesteps`
2. 从高斯噪声初始化 `sample`
3. 对每个时间步调用 `predict_fn(sample, t)`
4. 调用 scheduler 的 `step(...)`
5. 迭代得到最终样本
6. 可选返回 `uint8` 图像张量

---

### `build_save_pipeline(unet, noise_scheduler, use_ddim_sampling)`
**作用**：构建用于 `save_pretrained(...)` 的 Diffusers pipeline。

返回：
- `DDIMPipeline`
- 或 `DDPMPipeline`

---

## 4.5 `ddpm.py`

### `build_ddpm(args)`
**作用**：返回 DDPM 模式操作集合。

返回字典字段：
- `name`
- `build_extra_components`
- `train_step`
- `sample_images`
- `checkpoint_extra_state`
- `load_checkpoint_extra_state`

---

### `build_extra_components(num_classes, device)`
DDPM 不需要额外组件，返回空字典 `{}`。

---

### `prepare_batch_labels(batch, device)`
**作用**：
- 如果开启 `use_class_conditioning`，返回类别标签
- 否则返回 `None`

---

### `train_step(model, noise_scheduler, batch, accelerator, extra_components)`
**作用**：执行一个 DDPM 训练 step。

**流程：**
1. 取出 `clean_images`
2. 可选取出 `class_labels`
3. 采样噪声 `noise`
4. 随机采样扩散步 `timesteps`
5. 构造 `noisy_images`
6. 预测噪声 `noise_pred`
7. 计算 `MSE loss`

返回：
```python
loss, aux
```

其中 `aux` 包含：
- `class_labels`
- `timesteps`

---

### `sample_images(...)`
**作用**：DDPM 模式图像采样。

内部定义 `predict_fn(sample, t)`，然后调用 `run_sampling_loop(...)`。

---

### `checkpoint_extra_state(extra_components)`
返回空字典。

---

### `load_checkpoint_extra_state(checkpoint, extra_components, device)`
无额外恢复逻辑，返回 `None`。

---

## 4.6 `cfg.py`

### `build_cfg(args)`
**作用**：返回 CFG 模式操作集合。

---

### `_get_null_class_idx(num_classes)`
**作用**：返回 CFG 的 null class 索引。

规则：
- null class 索引固定为 `num_classes`

---

### `build_extra_components(num_classes, device)`
返回：
```python
{
    "num_classes": num_classes,
    "null_class_idx": num_classes,
}
```

---

### `_maybe_drop_labels(class_labels, null_class_idx)`
**作用**：训练阶段随机将部分真实标签替换为 null label，模拟无条件分支。

**规则：**
- 替换概率由 `args.cond_drop_prob` 控制
- 输出仍是一个标签张量

---

### `train_step(model, noise_scheduler, batch, accelerator, extra_components)`
**作用**：执行一个 CFG 训练 step。

与 DDPM 的区别：
- 训练时会先调用 `_maybe_drop_labels(...)`
- 传入模型的是 `dropped_labels`，而不一定是真实标签

返回：
```python
loss, aux
```

`aux` 包含：
- `class_labels`
- `dropped_labels`
- `timesteps`

---

### `sample_images(...)`
**作用**：CFG 采样。

**流程：**
1. 构造 `null_labels`
2. 在 `predict_fn` 中分别计算：
   - `eps_uncond`
   - `eps_cond`
3. 使用 CFG 公式：
```python
eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
```
4. 调用 `run_sampling_loop(...)`

---

### `checkpoint_extra_state(extra_components)`
返回：
```python
{
    "cfg_null_class_idx": int(extra_components["null_class_idx"]),
}
```

---

### `load_checkpoint_extra_state(checkpoint, extra_components, device)`
当前无额外恢复逻辑，返回 `None`。

---

## 4.7 `cg.py`

### `ClassifierGuidanceAdapter`
**作用**：把 classifier guidance 所需的
`∇_{x_t} log p(y | x_t, t)`
抽象成统一接口。

#### `__init__(classifier, guidance_scale)`
保存分类器对象和引导强度。

#### `grad_log_prob(x_t, t, class_labels)`
**作用**：计算：
```python
∇_{x_t} log p(y | x_t, t)

```

### `ClassifierWithUNetDownsample`
**作用**：基于已训练 `UNet2DModel` 的下采样路径与中间块构建 CG classifier。

**结构组成：**
- 复用 diffusion UNet 的：
  - `conv_in`
  - `down_blocks`
  - `mid_block`
  - `time_proj`
  - `time_embedding`
- 末端新增：
  - `AttentionPool2d` / `RotAttentionPool2d`
  - `Linear classifier`

**特点：**
- 输入为 noisy image `x_t` 与 `timesteps`
- 输出为分类 `logits`
- 复用的 UNet backbone 默认冻结，仅训练分类头及 attention pooling


### `build_classifier_from_trained_unet(...)`
**作用**：从当前项目构建好的 UNet 包装出 classifier。

**关键点：**
- 若不使用 rotary（旋转位置编码，rotary），会根据 UNet 的 downsample 结构自动推断最后特征图大小
- 返回 `ClassifierWithUNetDownsample`

---

### `build_cg(args)`
**作用**：返回 CG 模式操作集合。

---

### `build_extra_components(num_classes, device)`
**作用**：
1. 检查 `use_class_conditioning`
2. 若 `run_mode == train` 且未提供 `classifier_ckpt_path`，允许先返回空 classifier / adapter，以便先训练 diffusion model
3. 若处于 `val_only / infer_only`，或训练阶段已提供 `classifier_ckpt_path`，则加载 classifier 并构造 `ClassifierGuidanceAdapter`

**返回：**
- 训练阶段且尚未提供 classifier checkpoint：
```python
{
    "classifier": None,
    "cg_adapter": None,
}
```

---

### `train_step(model, noise_scheduler, batch, accelerator, extra_components)`
**作用**：扩散模型本体按条件扩散方式训练。

**注意：**
- 这里只训练 diffusion model
- 没有实现 classifier 的训练逻辑

---

### `sample_images(...)`
**作用**：CG 采样。

**流程：**
1. 正常计算条件扩散模型的 `eps`
2. 若 `cg_adapter is None`，则退化为普通 class-conditional diffusion 采样
3. 若存在 classifier guidance：
   - 计算 `grad = ∇_{x_t} log p(y | x_t, t)`
   - 读取 `alpha_bar_t = sampling_scheduler.alphas_cumprod[t]`
   - 使用当前实现的修正公式：
```python
eps_guided = eps - guidance_scale * sqrt(1 - alpha_bar_t) * grad
```
---

### `checkpoint_extra_state(extra_components)`
返回：
```python
{
    "classifier_ckpt_path": args.classifier_ckpt_path,
    "classifier_guidance_scale": float(args.classifier_guidance_scale),
}
```

---

### `load_checkpoint_extra_state(checkpoint, extra_components, device)`
当前无额外恢复逻辑，返回 `None`。

---

### `_build_classifier_from_your_code(args, num_classes, device)`
**作用**：按当前项目统一配置先构建一个结构一致的 diffusion UNet，再基于该 UNet 构建 classifier，并从 `classifier_ckpt_path` 加载分类器权重。

**实现细节：**
- 先调用 `build_model(args, num_classes)` 构建 backbone UNet
- 再调用 `build_classifier_from_trained_unet(...)`
- 支持从以下格式恢复权重：
  - `checkpoint["classifier_state_dict"]`
  - `checkpoint["model_state_dict"]`
  - 或直接把整个 checkpoint 视作 `state_dict`
- 最终以 `strict=True` 加载，并设为 `eval()`

---

### `_build_noisy_classifier_batch(batch, noise_scheduler, device)`
**作用**：把 clean batch 转成 classifier 训练/验证所需的 noisy batch。
返回：
- `noisy_images`
- `labels`
- `timesteps`


### `evaluate_guidance_classifier(classifier, data_loader, noise_scheduler, device)`
**作用**：在 noisy image `(x_t, t)` 上评估 classifier。

**返回字段：**
- `loss`
- `accuracy`
- `balanced_accuracy`


### `train_guidance_classifier(...)`
**作用**：只训练 CG classifier，不训练 diffusion model。

**流程：**
1. 构建 `AdamW`
2. 对每个 batch 在线构造 noisy image `x_t`
3. 用 `CrossEntropyLoss` 训练分类器
4. 每个 epoch 在验证集上评估
5. 保存：
   - `classifier_last.pth.tar`
   - `classifier_best.pth.tar`

**返回：**
```python
{
    "history": ...,
    "best_val_balanced_accuracy": ...,
    "best_classifier_state_dict": ...,
}
```

---

## 4.8 `common.py`

### `get_modes(args)`
**作用**：根据 `args.mode` 返回对应模式实现。

映射关系：
- `"ddpm"` -> `build_ddpm(args)`
- `"cfg"` -> `build_cfg(args)`
- `"cg"` -> `build_cg(args)`

不支持的模式会抛出 `ValueError`。

---

## 4.9 `__init__.py`

### 模块导出
```python
from .common import get_modes
```

**作用**：对外暴露 `get_modes`。

---

## 4.10 `metrics.py`

### `tensor_to_uint8_for_fid(x)`
把 `[-1, 1]` 范围图像张量转换为 `uint8`，供 FID/KID 使用。

---

### `uint8_tensor_to_pil(x_uint8)`
把 `uint8` 张量转为 `PIL.Image`。

---

### `allocate_samples_by_ratio(count_dict, total_samples)`
**作用**：按真实数据类别比例，为每个类别分配生成样本数。

**特点：**
- 先做 floor 分配
- 再按小数余量补足剩余样本

---

### `collect_real_images_by_class(real_loader, device, class_names, target_counts_by_class)`
**作用**：从真实数据集中按类别收集指定数量的真实图像。

返回：
```python
{
    class_name: uint8_tensor_batch
}
```

---

### `generate_images_by_class_for_metrics(...)`
**作用**：按类别生成评估用 fake images，并保存到磁盘。

返回：
```python
fake_by_class, generated_dir
```

其中：
- `fake_by_class`：按类别组织的 `uint8` 张量
- `generated_dir`：图像保存目录

---

### `concat_class_tensors(class_tensor_dict, class_names, target_counts_by_class, device)`
把按类别分开的张量按既定顺序拼接成一个总张量。

---

### `compute_fid_from_real_and_fake(real_images_uint8, fake_images_uint8, device)`
计算整体 FID。

返回：
- `float`
- 或 `None`

---

### `compute_kid_from_real_and_fake(real_images_uint8, fake_images_uint8, device, subsets=50, subset_size=50)`
计算整体 KID。

返回：
```python
(kid_mean, kid_std)
```

---

### `_build_vgg16_feature_extractor(device)`
构建用于 IPR 的 VGG16 特征提取器。

---

### `_extract_vgg16_features(images_uint8, vgg16, device, batch_size=64)`
从图像中提取 VGG16 特征。

---

### `_compute_pairwise_distances(X, Y=None)`
计算欧氏距离矩阵。

---

### `_distances_to_radii(distances, k)`
对每个样本求其第 `k` 个邻居距离，用作流形半径。

---

### `_build_manifold(features, k)`
用特征构建流形表示：
```python
Manifold(features, radii)
```

---

### `_compute_precision_or_recall(manifold_ref, feats_query)`
根据流形包含关系计算 precision 或 recall。

---

### `compute_manifold_precision_recall(real_images_uint8, fake_images_uint8, device, k=3, vgg_batch_size=64)`
**作用**：计算 IPR 中的 precision / recall。

返回：
```python
(precision, recall)
```

---

### `evaluate_split_with_overall_and_per_class_metrics(...)`
**作用**：对一个数据划分（train / val）执行整体评估与可选逐类评估。

**主要流程：**
1. 按类别分配评估样本数
2. 收集真实图像
3. 生成 fake 图像
4. 计算整体 FID/KID/IPR
5. 保存整体 JSON
6. 可选执行 train split 的 per-class metrics
7. 返回完整评估结果字典

**返回字段：**
- `overall_fid`
- `overall_kid_mean`
- `overall_kid_std`
- `overall_precision`
- `overall_recall`
- `overall_json_path`
- `per_class_json_path`
- `generated_dir`
- `per_class_generated_dir`
- `per_class_metrics`
- `allocated_counts_by_class`
- `per_class_counts_by_class`

---

## 4.11 `utils.py`

### `make_experiment_name(args)`
生成训练实验名。

---

### `setup_experiment_folders(base_dir, exp_name)`
创建训练实验目录结构。

---

### `make_runtime_run_name(args)`
生成 `val_only` / `infer_only` 运行名。

---

### `setup_runtime_run_folders(exp_dir, run_mode, run_name)`
创建运行时目录结构。

---

### `save_json(data, json_path)`
保存 JSON 文件。

---

### `update_epoch_metrics_csv(metrics_csv_path, row_dict)`
向训练指标 CSV 追加一行。

---

### `update_epoch_metrics_json(metrics_json_path, row_dict)`
向训练指标 JSON 追加一项。

---

### `count_labels_from_indices(labels, indices, class_names)`
按给定样本下标统计类别分布。

---

### `format_count_ratio_dict(count_dict)`
把类别计数字典格式化为：
```python
{
    "CLASS": "count (ratio%)"
}
```

---

### `print_class_distribution(title, count_dict)`
把类别分布打印到控制台。

---

### `save_checkpoint(state, is_best, save_dir, filename="last.pth.tar")`
保存 checkpoint，并在 `is_best=True` 时同步更新 `model_best.pth.tar`。

---

### `disable_pipeline_progress_bar(pipeline)`
关闭 Diffusers pipeline 进度条。

---

### `save_diffusers_model_index_copy(exp_dir, metadata_dir)`
把 `model_index.json` 复制到 metadata 目录。

---

### `recover_exp_dir_from_checkpoint(checkpoint_path, checkpoint_data)`
从 checkpoint 内容或 checkpoint 路径中恢复实验目录。

---

### `sync_experiment_metadata_for_resume(experiment_metadata, args, start_epoch, global_step)`
在断点恢复训练时同步实验元数据。

---

## 4.12 `runtime.py`

### `set_seed(seed)`
设置 Python / NumPy / PyTorch 的随机种子。

---

### `run_validation_only(...)`
**作用**：执行仅验证模式。

**流程：**
1. 创建 `val_only` 运行目录
2. 保存本次运行配置
3. 按需评估 train split
4. 按需评估 val split
5. 保存运行摘要

---

### `run_inference_only(...)`
**作用**：执行仅推理模式。

**流程：**
1. 解析 `infer_label`
2. 创建 `infer_only` 运行目录
3. 保存运行配置
4. 构造采样 scheduler
5. 生成指定数量图像
6. 保存图片与运行摘要

---

### `run_train(args)`
**作用**：训练主流程，也是整个项目的核心入口。

**主要阶段：**

#### 1. 初始化
- 设置随机种子
- 如果是恢复训练，加载 checkpoint 并恢复实验目录
- 否则新建实验目录

#### 2. 基础组件构建
- 创建 `Accelerator`
- 构建数据集与 DataLoader
- 打印类别分布
- 构建 mode 操作集合 `mode_ops`
- 构建 UNet 与 noise scheduler
- 构建模式相关额外组件 `extra_components`

#### 3. 仅验证 / 仅推理分支
当 `run_mode in [val_only, infer_only]` 时：
- 加载模型参数
- 恢复模式相关附加状态
- 进入对应运行函数
- 结束训练器

#### 4. 训练组件构建
- 创建 `AdamW`
- 计算步数
- 创建学习率调度器 `get_scheduler(...)`

#### 5. 恢复训练状态
如有 checkpoint：
- 恢复 model
- 恢复 optimizer
- 恢复 lr_scheduler
- 恢复模式附加状态
- 恢复 epoch / global_step / best 指标

#### 6. `accelerator.prepare(...)`
将训练相关对象交给 Accelerate 托管。

#### 7. 训练循环
对每个 epoch：
- `model.train()`
- 遍历 `train_dataloader`
- 调用 `mode_ops["train_step"](...)`
- `backward`
- 梯度裁剪
- `optimizer.step()`
- `lr_scheduler.step()`
- `optimizer.zero_grad()`

#### 8. 主进程附加工作
在 `accelerator.is_main_process` 下按需：
- 保存样本图
- 执行 train/val 评估
- 保存 checkpoint
- 保存 Diffusers pipeline
- 更新 metrics CSV/JSON
- 更新 experiment metadata

#### 9. 结束
调用 `accelerator.end_training()`

---

## 5. 给大模型修改代码时的边界规范

### 5.1 允许修改的内容

仅在明确任务要求下允许修改以下范围：

1. **函数内部实现**
   - 允许修改单个函数内部逻辑
   - 前提是不破坏该函数的输入输出契约

2. **新功能的局部扩展**
   - 可新增辅助函数
   - 可新增少量字段到返回字典
   - 前提是调用方同步更新

3. **评估逻辑扩展**
   - 可在 `metrics.py` 中新增指标
   - 前提是不破坏现有 FID/KID/IPR 流程

4. **模式逻辑扩展**
   - 可在 `ddpm.py / cfg.py / cg.py` 中补充 mode-specific 逻辑
   - 前提是保留 `mode_ops` 统一接口

---

### 5.2 禁止随意修改的内容

以下内容默认禁止改，除非任务明确要求：

1. **文件名与模块边界**
   - 不要重命名现有文件
   - 不要随意合并文件
   - 不要拆分现有主逻辑文件

2. **核心接口签名**
   - 不要随意改函数名
   - 不要随意改参数名
   - 不要随意改返回字典键名
   - 尤其禁止改以下接口契约：
     - `build_datasets_and_loaders(args)`
     - `build_model(args, num_classes)`
     - `build_noise_scheduler(args)`
     - `build_sampling_scheduler(...)`
     - `run_sampling_loop(...)`
     - `get_modes(args)`
     - `mode_ops["train_step"]`
     - `mode_ops["sample_images"]`
     - `evaluate_split_with_overall_and_per_class_metrics(...)`
     - `run_train(args)`

3. **数据字典结构**
   `Dataset.__getitem__()` 的返回格式必须保持：
   ```python
   {
       "input": image_tensor,
       "label": label_int,
       "sample_id": image_id,
   }
   ```

4. **mode_ops 协议**
   `build_ddpm / build_cfg / build_cg` 返回的字典必须至少包含：
   ```python
   {
       "name": ...,
       "build_extra_components": ...,
       "train_step": ...,
       "sample_images": ...,
       "checkpoint_extra_state": ...,
       "load_checkpoint_extra_state": ...,
   }
   ```

5. **实验目录协议**
   不要随意改变：
   - checkpoint 默认保存位置
   - metrics 默认保存位置
   - samples 默认保存位置
   - metadata 默认保存位置

6. **图像张量约定**
   - 训练输入图像默认是 `[-1, 1]` 范围
   - FID/KID 输入是 `uint8`
   - 不要在未同步所有调用方的前提下更改这个约定

---


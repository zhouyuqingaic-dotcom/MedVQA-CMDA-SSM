"""
CM-DA-SSM · BioMedCLIP 本地加载器

用于 Stage 1（MIMIC-CXR）与 Stage 2（医学 VQA）的冻结图文先验提取。

本地资源：
    BioMedCLIP:
        open_clip_config.json
        open_clip_pytorch_model.bin
        tokenizer.json / tokenizer_config.json 等

    BiomedBERT:
        config.json（用于构造 HF 文本编码器的网络结构）

重要说明：
    1. BiomedBERT 本地目录只提供模型结构配置。
    2. 图像编码器和文本编码器的实际参数来自完整 BioMedCLIP checkpoint。
    3. 禁止额外从 Hugging Face 在线下载 BiomedBERT 原始预训练参数。
    4. 与原工程保持相同的函数签名和三个返回值。
"""

import json
import os
from copy import deepcopy

import open_clip
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from transformers import AutoConfig, AutoTokenizer

from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper import (
    FrozenBioMedCLIPFeatureExtractor,
)
from utils.ddp.ddp_utils import ddp_print


def load_biomedclip(biomedclip_path: str, print_rank: int = 0):
    """
    完全从本地加载 BioMedCLIP 图像编码器、文本编码器和 Tokenizer。

    Args:
        biomedclip_path (str):
            BioMedCLIP 的本地模型目录，包含：
                open_clip_config.json
                open_clip_pytorch_model.bin
                tokenizer.json / tokenizer_config.json 等文件

        print_rank (int):
            兼容原项目 DDP 打印接口，仅指定需要输出日志的 Rank。

    Returns:
        tuple:
            (
                biomed_extractor,        # 冻结的 BioMedCLIP 特征提取器
                biomed_transform,        # BioMedCLIP 图像预处理
                custom_biomed_tokenizer  # 文本 -> [B, 256] input_ids
            )
    """

    ddp_print(
        f"\n⏳ [Loader] 正在本地加载 BioMedCLIP (路径: {biomedclip_path})",
        print_rank=print_rank,
    )

    try:
        # ============================================================
        # 1. 确认 BioMedCLIP 配置和完整权重存在
        # ============================================================
        cfg_path = os.path.join(
            biomedclip_path,
            "open_clip_config.json",
        )
        bin_path = os.path.join(
            biomedclip_path,
            "open_clip_pytorch_model.bin",
        )

        for file_path in (cfg_path, bin_path):
            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    f"缺少 BioMedCLIP 本地文件：{file_path}"
                )

        with open(cfg_path, "r", encoding="utf-8") as f:
            clip_cfg = json.load(f)

        # 深拷贝：不要在本地修复时直接修改官方模型配置，
        # 也不要影响之后重新读取的 open_clip_config.json。
        model_cfg = deepcopy(clip_cfg["model_cfg"])
        preprocess_cfg = clip_cfg["preprocess_cfg"]
        custom_model_name = "biomedclip_local"

        # ============================================================
        # 2. BiomedBERT 只读取本地结构配置
        # ============================================================
        # 可通过环境变量覆盖目录，不影响原有 load_biomedclip 接口。
        # 默认目录与当前 CM-DA-SSM 服务器的目录约定一致。
        biomedbert_config_dir = os.path.abspath(
            os.environ.get(
                "BIOMEDBERT_CONFIG_DIR",
                "/home/yuqing/Models/BiomedBERT-config",
            )
        )

        bert_config_file = os.path.join(
            biomedbert_config_dir,
            "config.json",
        )

        if not os.path.isfile(bert_config_file):
            raise FileNotFoundError(
                "缺少 BiomedBERT 的本地结构配置："
                f"{bert_config_file}\n"
                "请先下载 microsoft/BiomedNLP-BiomedBERT-base-"
                "uncased-abstract 的 config.json，"
                "或设置 BIOMEDBERT_CONFIG_DIR。"
            )

        # 预先检查本地 config.json 能否正确解析。
        # local_files_only=True：此处不允许 Hugging Face 网络回退。
        AutoConfig.from_pretrained(
            biomedbert_config_dir,
            local_files_only=True,
        )

        if "text_cfg" not in model_cfg:
            raise KeyError(
                "BioMedCLIP open_clip_config.json 缺少 text_cfg。"
            )

        # 原始配置指向 microsoft/BiomedNLP-... 在线仓库；
        # 将它替换为包含 config.json 的本地目录。
        model_cfg["text_cfg"]["hf_model_name"] = (
            biomedbert_config_dir
        )

        # 不单独加载原始 BiomedBERT 权重！
        # 下面 pretrained=bin_path 会加载完整 BioMedCLIP 权重，
        # 其中已经包含训练好的文本编码器权重。
        model_cfg["text_cfg"]["hf_model_pretrained"] = False

        ddp_print(
            f"    [Loader] BiomedBERT 本地配置："
            f"{bert_config_file}",
            print_rank=print_rank,
        )

        # ============================================================
        # 3. 注册本地 OpenCLIP 模型结构
        # ============================================================
        # _MODEL_CONFIGS 是 OpenCLIP 注册表。
        # 无条件覆盖当前名字，防止同一进程内残留旧配置，
        # 导致模型仍然尝试请求 Hugging Face。
        if custom_model_name.startswith(HF_HUB_PREFIX):
            raise ValueError(
                "本地模型名不能使用 hf-hub: 前缀。"
            )

        _MODEL_CONFIGS[custom_model_name] = model_cfg

        # ============================================================
        # 4. 从完整 BioMedCLIP checkpoint 加载图文编码器
        # ============================================================
        # pretrained_hf=False：
        #   禁止额外加载 Hugging Face 原始 BiomedBERT 权重。
        #
        # pretrained=bin_path：
        #   加载 BioMedCLIP 已训练好的完整图像/文本编码器。
        #
        # 同时仍使用官方提供的图像预处理参数。
        raw_biomed_model, _, biomed_transform = (
            open_clip.create_model_and_transforms(
                model_name=custom_model_name,
                pretrained=bin_path,
                pretrained_hf=False,
                **{
                    f"image_{k}": v
                    for k, v in preprocess_cfg.items()
                },
            )
        )

        # ============================================================
        # 5. 加载 BioMedCLIP 本地配套的 PubMedBERT Tokenizer
        # ============================================================
        # 必须使用 BioMedCLIP 自己的 tokenizer 文件；
        # 不能用 Qwen tokenizer 或随意换一个 BERT tokenizer。
        hf_tokenizer = AutoTokenizer.from_pretrained(
            biomedclip_path,
            local_files_only=True,
        )

        def custom_biomed_tokenizer(texts):
            """
            保持原 Stage 1 / Stage 2 Collator 的接口：
                输入：文本列表
                输出：LongTensor [B, 256]

            Stage 1 的文本应是固定报告生成指令；
            Stage 2 的文本应是当前问题，不能传入标准答案。
            """
            return hf_tokenizer(
                texts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

        # ============================================================
        # 6. 包装为工程原有的冻结特征提取器
        # ============================================================
        biomed_extractor = FrozenBioMedCLIPFeatureExtractor(
            raw_biomed_model
        )

        ddp_print(
            "✅ BioMedCLIP 图像/文本编码器及 Tokenizer 本地加载成功！",
            print_rank=print_rank,
        )

        return (
            biomed_extractor,
            biomed_transform,
            custom_biomed_tokenizer,
        )

    except Exception as e:
        ddp_print(
            f"❌ BioMedCLIP 本地加载失败：{e}",
            print_rank=print_rank,
        )
        raise
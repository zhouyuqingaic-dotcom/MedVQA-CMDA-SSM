"""
Qwen3-VL-8B 量化模型加载器（CM-DA-SSM Stage 1）。

用户当前上传的项目中 Trainer 已引用 Qwen3VLQuantizedLoader，
但 ZIP 内缺少该文件。这里补齐独立加载器，保持 Trainer 之前的
构造参数和 load() -> (model, processor) 接口不变。

注意：
    - 使用 Transformers 自带的 Qwen3VLForConditionalGeneration；
    - 4-bit / NF4 只用于 QLoRA 底座，新增视觉 Adapter 后续单独挂载；
    - 分布式训练每个进程只管理自己的 CUDA 设备；
    - 不在 import 时加载权重，方便进行无 GPU 的配置单元测试。
"""

from __future__ import annotations

from typing import Any

import torch


def _resolve_torch_dtype(value: str | torch.dtype) -> torch.dtype:
    """
    将配置中的 'bfloat16' / 'float16' / 'float32' 转换为 torch.dtype。

    拒绝 silent fallback：BF16 设备与 FP16 设备要求不同，
    随便回退会改变实验的数值环境。
    """
    if isinstance(value, torch.dtype):
        if value in {torch.bfloat16, torch.float16, torch.float32}:
            return value
        raise ValueError(f"不支持的模型精度：{value}")

    lookup = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in lookup:
        raise ValueError(
            f"torch dtype 必须为 BF16/FP16/FP32，实际为 {value!r}"
        )
    return lookup[value]


class Qwen3VLQuantizedLoader:
    """
    构造 Qwen3-VL 底座模型与 AutoProcessor。

    Args:
        model_path: 本地或 Hugging Face Qwen3-VL 模型目录。
        processor_path: Processor 的本地路径，通常与 model_path 相同。
        load_in_4bit: 是否使用 bitsandbytes 4-bit 量化。
        device_map: torchrun 下建议 {'': local_rank}；
            单 GPU 非分布式可用 'auto'。
    """

    def __init__(
        self,
        model_path: str,
        processor_path: str | None = None,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: str | torch.dtype = "bfloat16",
        torch_dtype: str | torch.dtype = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        device_map: Any = "auto",
    ):
        if not model_path:
            raise ValueError("必须提供 Qwen3-VL 模型路径。")

        if bnb_4bit_quant_type not in {"nf4", "fp4"}:
            raise ValueError(
                "bnb_4bit_quant_type 只能是 nf4 或 fp4。"
            )

        self.model_path = model_path
        self.processor_path = processor_path or model_path
        self.load_in_4bit = bool(load_in_4bit)
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bool(bnb_4bit_use_double_quant)
        self.bnb_4bit_compute_dtype = _resolve_torch_dtype(
            bnb_4bit_compute_dtype
        )
        self.torch_dtype = _resolve_torch_dtype(torch_dtype)
        self.attn_implementation = attn_implementation
        self.device_map = device_map

    def load(self):
        """
        加载 Qwen3VLForConditionalGeneration + AutoProcessor。

        真正执行时需要 Transformers 的 Qwen3-VL 支持；
        4-bit 额外需要 CUDA、bitsandbytes 和 accelerate。
        """
        try:
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen3VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                "缺少支持 Qwen3-VL 的 transformers 版本。"
                "请在正式训练环境安装 transformers、accelerate；"
                "4bit 还需要 bitsandbytes。"
            ) from exc

        model_kwargs = {
            "device_map": self.device_map,
            "dtype": self.torch_dtype,
            "attn_implementation": self.attn_implementation,
        }

        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=(
                    self.bnb_4bit_use_double_quant
                ),
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            )

        # 从本地模型权重初始化。不能从 Stage 2 或其他消融组的
        # checkpoint 读取，保证各组 Stage 1 独立训练。
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            self.processor_path,
        )

        return model, processor

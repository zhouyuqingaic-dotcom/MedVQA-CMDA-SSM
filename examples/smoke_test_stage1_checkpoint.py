"""
CM-DA-SSM Stage 1 Checkpoint / 恢复的最小 CPU 验证。

不加载真实 Qwen3-VL-8B，不运行 MIMIC-CXR，主要检查：
    1. A0 只保存 LoRA + architecture.json；
    2. A1/A2/A3/A3-NoRMS 额外保存 visual_adapter.pt；
    3. 模型重建后 strict=True 加载视觉 Adapter；
    4. 其它组（尤其 A3 vs A3-NoRMS）不能混用 checkpoint；
    5. 输出结构符合下一步 Stage 2 Trainer 的继承约定。

运行：
    python examples/smoke_test_stage1_checkpoint.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from config.stage1_train_config_mimic_cxr import TrainConfig
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_visual_adapter import (
    Qwen3VLCMDASSMVisualAdapter,
)
from utils.training.stage1_checkpoint_utils import (
    get_visual_adapter,
    save_stage1_artifacts,
    read_stage1_manifest,
    restore_visual_adapter_strict,
    validate_stage1_artifacts,
)


class MockPeft(nn.Module):
    """模拟 PEFT 的视觉 Adapter 注册层级，不含 8B 模型权重。"""
    def __init__(self, cfg):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(4, 4))
        self.lora_B = nn.Parameter(torch.randn(4, 4))

        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.model = nn.Module()
        vision = nn.Module()

        if cfg.enable_visual_adapter:
            vision.res_adapter = Qwen3VLCMDASSMVisualAdapter(
                hidden_dim=32,
                r=4,
                ssm_d_state=4,
                ssm_backend="torch_reference",
                query_dim=8,
                condition_hidden_dim=16,
                fusion_mode=cfg.fusion_mode,
                lambda_mode=cfg.lambda_mode,
                use_rms_norm=cfg.use_rms_norm,
                lambda_init=cfg.lambda_init,
                zero_init_up=True,
            )

        self.base_model.model.model.visual = vision


def write_fake_lora(directory):
    # 真实训练使用 trainer.save_model() 写出这些文件。
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}', encoding="utf-8"
    )
    (directory / "adapter_model.safetensors").write_bytes(
        b"demo_only_not_real_lora_weights"
    )


def main():
    torch.manual_seed(2048)

    with tempfile.TemporaryDirectory() as tmp:
        groups = ("A0", "A1", "A2", "A3", "A3-NoRMS")

        for group in groups:
            cfg = TrainConfig(
                ablation_id=group,
                output_root=tmp,
            )
            model = MockPeft(cfg)
            folder = Path(cfg.final_weights_dir)
            write_fake_lora(folder)

            save_stage1_artifacts(
                model=model,
                cfg=cfg,
                directory=folder,
                global_step=10,
                train_sample_count=20,
                train_sample_digest="example-only",
                is_final=True,
            )

            validate_stage1_artifacts(
                cfg=cfg,
                directory=folder,
                expected_train_sample_count=20,
                expected_train_sample_digest="example-only",
            )

            manifest = read_stage1_manifest(folder)
            assert manifest["ablation_id"] == group

            restored = MockPeft(cfg)
            restore_visual_adapter_strict(
                restored,
                cfg=cfg,
                directory=folder,
            )

            if cfg.enable_visual_adapter:
                state = get_visual_adapter(model).state_dict()
                restored_state = get_visual_adapter(restored).state_dict()
                assert set(state) == set(restored_state)
                assert all(
                    torch.equal(state[name], restored_state[name])
                    for name in state
                )
            else:
                assert get_visual_adapter(restored) is None

            print(f"{group:9s}: save / validate / strict restore PASS")

    print("Stage 1 checkpoint smoke test PASSED.")


if __name__ == "__main__":
    main()

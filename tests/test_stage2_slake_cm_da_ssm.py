"""
CM-DA-SSM Stage 2 SLAKE CPU 合约测试。

覆盖：
  - 五组消融的 Stage 1 -> Stage 2 权重路径；
  - 当前 A3 的 λ、η、SSM LR；
  - A0/A1 不产生 BioMedCLIP 输入；
  - A2/A3/A3-NoRMS 只将 question 交给 BioMedCLIP；
  - Stage 2 使用 train.json，不混入 validate/test；
  - Visual Adapter 和来源元数据保存；
  - Stage 2 checkpoint 完整性检查；
  - Stage 1 manifest 的验证契约。

本测试不用真实 Qwen3-VL-8B 权重，不调用 GPU Mamba。
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from config.stage1_train_config_mimic_cxr import TrainConfig
from config.slake.stage2_train_config_slake import Stage2TrainConfig
from config.vqa_rad.stage2_train_config_vqa_rad import Stage2TrainConfig as RAD
from config.vqa_med_2019.stage2_train_config_vqa_med_2019 import Stage2TrainConfig as MED19
from config.vqa_med_2021.stage2_train_config_vqa_med_2021 import Stage2TrainConfig as MED21
from training import stage2_trainer_slake as entry
from utils.data_tools.collator.slake.slake_datasets_train_collator import (
    SLAKETrainCollator,
)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(padding_side="right")

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False
    ):
        question = messages[0]["content"][1]["text"]
        start = "QIMAGE:" + question + "|ASSISTANT:"
        if len(messages) == 1:
            return start
        return start + messages[1]["content"][0]["text"] + "|END"

    def __call__(self, *, text, images, return_tensors, padding):
        encoded = [[ord(char) % 200 + 1 for char in item] for item in text]
        maxlen = max(len(ids) for ids in encoded)
        ids = torch.zeros((len(text), maxlen), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sample in enumerate(encoded):
            ids[i, :len(sample)] = torch.tensor(sample)
            mask[i, :len(sample)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.model = nn.Module()
        self.base_model.model.model.visual = nn.Module()
        self.base_model.model.model.visual.res_adapter = nn.Linear(3, 3)


class TestStage2SlakeContract(unittest.TestCase):

    def test_all_ablation_paths_match_own_stage1(self):
        for ablation in ("A0", "A1", "A2", "A3", "A3-NoRMS"):
            s1 = TrainConfig(ablation_id=ablation, seed=2048, run_tag="main")
            for cls in (Stage2TrainConfig, RAD, MED19, MED21):
                s2 = cls(ablation_id=ablation)
                with self.subTest(ablation=ablation, dataset=cls.__name__):
                    self.assertEqual(s2.stage1_weights_dir, s1.final_weights_dir)
                    self.assertEqual(s2.architecture_fingerprint, s1.architecture_fingerprint)
                    self.assertEqual(s2.ssm_learning_rate, 3e-5)

    def test_a3_final_values(self):
        c = Stage2TrainConfig(ablation_id="A3")
        self.assertEqual(c.lambda_init, 0.4)
        self.assertEqual(c.eta, 0.5)
        self.assertEqual(c.stage1_lr_recipe_name, "CMDA-DLR")
        self.assertEqual(entry.stage1_cfg_for(c).ssm_learning_rate, 1e-4)
        self.assertTrue(c.requires_biomedclip)
        self.assertTrue(c.use_rms_norm)

    def test_collator_biomed_gating_for_all_five_ablations(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "image.png"
            Image.new("RGB", (12, 16), (125, 100, 50)).save(img)
            sample = {
                "image_path": str(img),
                "question": "Is there a fracture?",
                "answer": "yes",
            }
            calls = []

            def biomed_tokenizer(texts):
                calls.append(texts)
                return torch.ones((len(texts), 256), dtype=torch.long)

            for ablation in ("A0", "A1", "A2", "A3", "A3-NoRMS"):
                cfg = Stage2TrainConfig(
                    ablation_id=ablation, slake_max_size=128,
                )
                kwargs = {}
                if cfg.requires_biomedclip:
                    kwargs = {
                        "biomed_transform": lambda image: torch.ones(3, 8, 8),
                        "biomed_tokenizer": biomed_tokenizer,
                    }
                collator = SLAKETrainCollator(
                    processor=FakeProcessor(),
                    cfg=cfg,
                    **kwargs,
                )
                result = collator([sample])
                self.assertGreater(result["labels"].ne(-100).sum().item(), 0)
                self.assertEqual(
                    "biomed_image_tensors" in result,
                    cfg.requires_biomedclip,
                )
                self.assertEqual(
                    "biomed_text_tokens" in result,
                    cfg.requires_biomedclip,
                )
            self.assertEqual(
                calls,
                [["Is there a fracture?"]] * 3,
            )

    def test_dataset_is_train_only_and_filters_empty_supervision(self):
        fake_data = [
            {"question": "question 1", "answer": "yes", "image_path": "1.jpg"},
            {"question": "", "answer": "no", "image_path": "2.jpg"},
            {"question": "question 3", "answer": "", "image_path": "3.jpg"},
            {"question": "question 4", "answer": "no", "image_path": "4.jpg"},
        ]

        class FakeDataset:
            def __init__(self, **kwargs):
                self.arguments = kwargs
            def __len__(self):
                return len(fake_data)
            def __getitem__(self, i):
                return fake_data[i]

        with patch.object(entry, "SLAKEDataset", FakeDataset):
            cfg = Stage2TrainConfig(
                ablation_id="A3",
                max_slake_train_samples=1,
            )
            filtered = entry.load_train_data(cfg)
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["question"], "question 1")

    def test_stage1_manifest_check_validates_source_and_step(self):
        c = Stage2TrainConfig(ablation_id="A3")
        stage1_manifest = c.stage1_manifest()
        stage1_manifest.update(
            {
                "global_step": 5000,
                "train_sample_count": 160000,
                "is_final": True,
            }
        )

        with patch.object(
            entry, "validate_stage1_artifacts",
            return_value=stage1_manifest,
        ) as validate:
            manifest = entry.validate_stage1_for_slake(c)
            validate.assert_called_once()
            self.assertEqual(manifest["global_step"], 5000)

        with patch.object(
            entry, "validate_stage1_artifacts",
            return_value={**stage1_manifest, "global_step": 1000},
        ):
            with self.assertRaises(RuntimeError):
                entry.validate_stage1_for_slake(c)

    def test_save_stage2_metadata_and_visual_adapter(self):
        c = Stage2TrainConfig(ablation_id="A3")
        model = FakeModel()
        with tempfile.TemporaryDirectory() as temp:
            entry.save_stage2_sidecars(
                model=model,
                directory=temp,
                cfg=c,
                stage1_manifest={"global_step": 5000},
                global_step=77,
                train_count=1000,
            )
            root = Path(temp)
            path = root / "visual_adapter.pt"
            manifest = json.loads(
                (root / "stage2_manifest.json").read_text(encoding="utf-8")
            )
            restored = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(restored), set(model.base_model.model.model.visual.res_adapter.state_dict()))
            self.assertEqual(manifest["global_step"], 77)
            self.assertEqual(manifest["stage1_global_step"], 5000)
            self.assertEqual(manifest["dataset"], "SLAKE")
            self.assertEqual(manifest["visual_adapter_sha256"], entry._sha256(path))

    def test_resume_requires_complete_adapter_and_optimizer(self):
        cfg = Stage2TrainConfig(ablation_id="A3")
        with tempfile.TemporaryDirectory() as temp:
            ckpt = Path(temp) / "checkpoint-50"
            ckpt.mkdir()
            fake = FakeModel()
            entry.save_stage2_sidecars(
                model=fake,
                directory=ckpt,
                cfg=cfg,
                stage1_manifest={"global_step": 5000},
                global_step=50,
                train_count=1000,
            )
            for file_name in (
                "trainer_state.json",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
            ):
                (ckpt / file_name).write_text("test")

            result = entry.resolve_resume_checkpoint(
                cfg, temp, "auto"
            )
            self.assertEqual(result, str(ckpt))
            (ckpt / "optimizer.pt").unlink()
            with self.assertRaises(RuntimeError):
                entry.resolve_resume_checkpoint(cfg, temp, "auto")
            with self.assertRaises(RuntimeError):
                entry.resolve_resume_checkpoint(cfg, temp, None)

    def test_lora_restore_accepts_expected_bf16_rounding(self):
        # 保存时为 FP32、恢复后的 LoRA Parameter 为 BF16。
        # 正确的 dtype 舍入不应被判断成权重没有加载。
        from safetensors.torch import save_file

        class FakePeftModel:
            def __init__(self):
                self.weight = torch.zeros(2, 2, dtype=torch.bfloat16)

        fake_peft = types.ModuleType("peft")

        def set_state(model, weights):
            model.weight = weights["lora_A.weight"].to(torch.bfloat16)

        def get_state(model):
            return {"lora_A.weight": model.weight}

        fake_peft.set_peft_model_state_dict = set_state
        fake_peft.get_peft_model_state_dict = get_state

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adapter_model.safetensors"
            tensor = torch.tensor(
                [[0.013579, -0.003827], [0.85423, 0.41793]],
                dtype=torch.float32,
            )
            save_file({"lora_A.weight": tensor}, str(path))
            with patch.dict(sys.modules, {"peft": fake_peft}):
                model = FakePeftModel()
                entry._load_lora_strict(model, Path(temp))
                self.assertTrue(
                    torch.equal(model.weight, tensor.to(torch.bfloat16))
                )

                def broken_set_state(model, weights):
                    pass

                fake_peft.set_peft_model_state_dict = broken_set_state
                with self.assertRaises(RuntimeError):
                    entry._load_lora_strict(FakePeftModel(), Path(temp))

    def test_checkpoint_callback_saves_visual_adapter(self):
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.TrainerCallback = type(
            "TrainerCallback", (), {},
        )

        cfg = Stage2TrainConfig(ablation_id="A3")
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            callback = entry.create_checkpoint_callback(
                cfg=cfg,
                stage1_manifest={"global_step": 5000},
                train_count=123,
            )

        with tempfile.TemporaryDirectory() as temp:
            args = types.SimpleNamespace(output_dir=temp)
            state = types.SimpleNamespace(
                is_world_process_zero=True,
                global_step=50,
            )
            control = object()
            result = callback.on_save(
                args, state, control, model=FakeModel(),
            )
            self.assertIs(result, control)
            self.assertTrue(
                (Path(temp) / "checkpoint-50" / "visual_adapter.pt").is_file()
            )
            self.assertTrue(
                (Path(temp) / "checkpoint-50" / "stage2_manifest.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()

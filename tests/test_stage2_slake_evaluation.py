"""SLAKE Stage 2 Validation/Test 的轻量 CPU 合约测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from config.slake.stage2_eval_config_slake import Stage2EvalConfig
from evaling.stage2_validate_checkpoints_slake import (
    DIAGNOSTIC_DIRECTIONS,
    _stats_for_image,
    calculate_metrics,
    discover_checkpoints,
    validate_stage2_checkpoint,
)
from evaling.stage2_test_checkpoints_slake import choose_test_checkpoints
from training.stage2_trainer_slake import _stage2_manifest
from utils.data_tools.collator.slake.slake_datasets_eval_collator import SLAKEEvalCollator


class FakeProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="left")

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "prompt " + messages[0]["content"][1]["text"]

    def __call__(self, *, text, images, return_tensors, padding):
        return {
            "input_ids": torch.arange(len(text) * 5).reshape(len(text), 5),
            "attention_mask": torch.ones((len(text), 5), dtype=torch.long),
        }


class TestSlakeEvaluation(unittest.TestCase):
    def test_eval_config_identity_paths_and_no_old_root(self):
        for name in ("A0", "A1", "A2", "A3", "A3-NoRMS"):
            cfg = Stage2EvalConfig(ablation_id=name)
            self.assertEqual(cfg.seed, 2048)
            self.assertEqual(cfg.stage1_seed, 2048)
            self.assertTrue(cfg.validation_output_dir.startswith(cfg.eval_output_root))
            self.assertTrue(cfg.test_output_dir.startswith(cfg.eval_output_root))
            self.assertIn("CM_DA_SSM", cfg.eval_output_root)
            self.assertEqual(cfg.lambda_init, 0.4)
            self.assertEqual(cfg.eta, 0.5)
            self.assertNotIn("MoRA_Med", cfg.eval_output_root)
            self.assertEqual(cfg.stage2_run_dir, cfg.output_dir)

    def test_eval_collator_biomed_is_only_for_cmradf_and_uses_question(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "image.png"
            Image.new("RGB", (16, 16)).save(file)
            record = {
                "image_path": str(file),
                "question": "Is there a fracture?",
                "answer": "yes",
                "answer_type": "CLOSED",
            }
            observed = []
            for name in ("A0", "A1", "A2", "A3", "A3-NoRMS"):
                cfg = Stage2EvalConfig(ablation_id=name)
                kwargs = {}
                if cfg.requires_biomedclip:
                    kwargs = {
                        "biomed_transform": lambda image: torch.ones(3, 8, 8),
                        "biomed_tokenizer": lambda texts: (
                            observed.append(texts)
                            or torch.ones((len(texts), 256), dtype=torch.long)
                        ),
                    }
                collator = SLAKEEvalCollator(
                    processor=FakeProcessor(), cfg=cfg, **kwargs,
                )
                batch, metadata = collator([record])
                self.assertEqual("biomed_image_tensors" in batch, cfg.requires_biomedclip)
                self.assertEqual("biomed_text_tokens" in batch, cfg.requires_biomedclip)
                self.assertEqual(metadata[0]["question"], record["question"])
            self.assertEqual(observed, [[record["question"]]] * 3)

    def test_four_direction_diagnostics(self):
        self.assertEqual(DIAGNOSTIC_DIRECTIONS, ("RF", "RR", "CF", "CR"))
        stats = _stats_for_image({
            "routing_weights": torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.3, 0.2, 0.1, 0.4]]]),
            "rho": torch.tensor(1.5),
            "lambda": torch.tensor(0.4),
            "delta_rms": torch.tensor(0.05),
        })
        self.assertAlmostEqual(stats["direction_rf"], 0.2, places=6)
        self.assertAlmostEqual(stats["direction_cr"], 0.4, places=6)
        self.assertAlmostEqual(stats["lambda_times_rho"], 0.6, places=6)
        self.assertNotIn("routing_f3", stats)
        self.assertNotIn("residual_gate", stats)

    def test_metrics_do_not_confuse_strict_and_llm_judge(self):
        cfg = Stage2EvalConfig(ablation_id="A3", strict_only=True)
        rows = [
            {"question_category": "closed", "final_correct": True, "lambda_value": None},
            {"question_category": "open", "final_correct": False, "lambda_value": None},
        ]
        metrics = calculate_metrics(rows, cfg)
        self.assertEqual(metrics["metric_protocol"], "strict_only")
        self.assertEqual(metrics["overall_strict_acc"], 0.5)
        self.assertEqual(metrics["closed_acc"], 1.0)
        self.assertEqual(metrics["open_strict_acc"], 0.0)

    def test_stage2_checkpoint_and_validation_best_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Stage2EvalConfig(
                ablation_id="A3",
                output_root=str(root / "models"),
                eval_output_root=str(root / "eval"),
                strict_only=True,
            )
            checkpoint = Path(cfg.stage2_run_dir) / "checkpoint-50"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
            visual = checkpoint / "visual_adapter.pt"
            visual.write_bytes(b"fixture")
            import hashlib
            meta = _stage2_manifest(cfg, {"global_step": 5000}, 50, 123)
            meta["visual_adapter_sha256"] = hashlib.sha256(visual.read_bytes()).hexdigest()
            (checkpoint / "stage2_manifest.json").write_text(json.dumps(meta))

            validate_stage2_checkpoint(checkpoint, cfg)
            self.assertEqual(discover_checkpoints(cfg), [checkpoint])
            best = {
                "dataset": "SLAKE",
                "selection_split": "validation",
                "metric_protocol": "strict_only",
                "selected_checkpoint": checkpoint.name,
                "selected_checkpoint_path": str(checkpoint),
                "ablation_id": cfg.ablation_id,
                "architecture_fingerprint": cfg.architecture_fingerprint,
                "seed": cfg.seed,
                "stage1_seed": cfg.stage1_seed,
                "run_tag": cfg.run_tag,
            }
            target = Path(cfg.best_checkpoint_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(best))
            self.assertEqual(choose_test_checkpoints(cfg), [checkpoint])
            # 禁止 Test 读取另一个消融组/旧配置的最佳节点。
            best["ablation_id"] = "A0"
            target.write_text(json.dumps(best))
            with self.assertRaises(RuntimeError):
                choose_test_checkpoints(cfg)


if __name__ == "__main__":
    unittest.main()

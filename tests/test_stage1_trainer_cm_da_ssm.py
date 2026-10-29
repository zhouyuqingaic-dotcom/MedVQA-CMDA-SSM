"""
CM-DA-SSM Stage 1 的 CPU 单元测试。

无需 8B Qwen3-VL 权重、不需要 Mamba CUDA、不需要真实 BioMedCLIP。
测试覆盖：
    - 五组的 BioMedCLIP/Collator 输入协议；
    - Trainer ↔ cfg.wrapper_kwargs / 新 AdamW 的装配；
    - PEFT LoRA + Adapter + architecture.json 的保存；
    - 中途 Checkpoint / 精确恢复、损坏文件/跨组拒绝；
    - A0 纯 LoRA；
    - 缺失量化 Loader 的接口与构造参数。
"""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from config.stage1_train_config_mimic_cxr import TrainConfig
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_visual_adapter import (
    Qwen3VLCMDASSMVisualAdapter,
)
from utils.qwen3vl.qwen3_vl_8B_quant_loader import (
    Qwen3VLQuantizedLoader,
    _resolve_torch_dtype,
)
from utils.data_tools.collator.mimic_cxr.mimic_cxr_datasets_train_collator import (
    MIMICCXRTrainCollator,
)
from utils.training.stage1_checkpoint_utils import (
    ARCHITECTURE_FILE,
    VISUAL_ADAPTER_FILE,
    create_stage1_checkpoint_callback,
    get_visual_adapter,
    fingerprint_train_dataset,
    read_stage1_manifest,
    resolve_resume_checkpoint,
    restore_visual_adapter_strict,
    save_stage1_artifacts,
    validate_stage1_artifacts,
)
from training import stage1_trainer_mimic_cxr as entry


def make_adapter(abl):
    cfg = TrainConfig(ablation_id=abl, ssm_backend="torch_reference")
    return Qwen3VLCMDASSMVisualAdapter(
        hidden_dim=32,
        r=4,
        ssm_d_state=4,
        ssm_d_conv=3,
        ssm_expand=2,
        ssm_backend="torch_reference",
        condition_hidden_dim=16,
        query_dim=8,
        fusion_mode=cfg.fusion_mode,
        lambda_mode=cfg.lambda_mode,
        use_rms_norm=cfg.use_rms_norm,
        lambda_init=cfg.lambda_init,
        zero_init_up=True,
    )


class FakePeftModel(nn.Module):
    """模仿实际的 base_model.model.model.visual.res_adapter 路径。"""

    def __init__(self, ablation):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(2, 2))
        self.lora_B = nn.Parameter(torch.randn(2, 2))
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.model = nn.Module()
        self.base_model.model.model.visual = nn.Module()
        if ablation != "A0":
            self.base_model.model.model.visual.res_adapter = (
                make_adapter(ablation)
            )


class FakeProcessor:
    """
    通过逐字符 tokenize 保证 prompt-only 一定是 full text 前缀。

    不模拟真实模型，只检查 Collator 不把 IMPRESSION 当作
    BioMedCLIP 的文本条件，也不在 A0/A1 生成多余字段。
    """
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(padding_side="right")

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        question = messages[0]["content"][1]["text"]
        prefix = "image:" + question + "|assistant:"
        if len(messages) == 1:
            return prefix
        return prefix + messages[1]["content"][0]["text"] + "|end"

    def __call__(self, *, text, images, return_tensors, padding):
        encoded = [
            [ord(char) % 200 + 1 for char in item]
            for item in text
        ]
        longest = max(len(row) for row in encoded)
        ids = torch.zeros(len(encoded), longest, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(encoded):
            ids[index, :len(row)] = torch.tensor(row)
            mask[index, :len(row)] = 1
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "image_grid_thw": torch.tensor(
                [[1, 4, 6]] * len(text),
                dtype=torch.long,
            ),
            "pixel_values": torch.ones(1, 4),
        }


class TestPooledOnlyStage1Contract(unittest.TestCase):
    """
    基于模拟 Qwen Vision Tower 检查 pooled-only 与 DeepStack 不变性。

    这里验证的是 Wrapper 输出契约，不是实际 Qwen3-VL-8B 权重。
    """

    class FakeVision(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32,32)
            self.deep_proj = nn.Linear(32,32)
            self.native = None

        def forward(self, x, grid_thw):
            self.native = types.SimpleNamespace(
                pooler_output=self.proj(x),
                deepstack_features=[self.deep_proj(x)],
            )
            return self.native

    def _make_vision(self, group):
        from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper import (
            install_cm_da_ssm_pooled_only_vision_patch,
        )
        adapter = make_adapter(group)
        vision = self.FakeVision()
        install_cm_da_ssm_pooled_only_vision_patch(vision,adapter)
        if group != "A1":
            vision.current_biomed_img_feat = torch.randn(1,512)
            vision.current_biomed_txt_feat = torch.randn(1,512)
        return vision

    def test_zero_up_identity_and_deepstack_object_preservation(self):
        torch.manual_seed(888)
        x=torch.randn(6,32)
        grid=torch.tensor([[1,4,6]])
        for group in ("A1","A2","A3","A3-NoRMS"):
            with self.subTest(group=group):
                vision=self._make_vision(group)
                native_pooled=vision.proj(x)
                result=vision(x,grid)
                self.assertTrue(torch.equal(
                    result.pooler_output,native_pooled
                ))
                self.assertIs(
                    result.deepstack_features,
                    vision.native.deepstack_features,
                )
                self.assertIs(
                    result.deepstack_features[0],
                    vision.native.deepstack_features[0],
                )

    def test_nonzero_up_updates_only_pooled_tokens(self):
        torch.manual_seed(889)
        x=torch.randn(6,32,requires_grad=True)
        grid=torch.tensor([[1,4,6]])
        vision=self._make_vision("A3")
        with torch.no_grad():
            nn.init.normal_(
                vision.res_adapter.feature_adapter.cm_radf.up.weight,
                std=0.05,
            )

        result=vision(x,grid)
        native_pooled=vision.proj(x)
        self.assertGreater(
            (result.pooler_output-native_pooled).abs().sum().item(),
            0.0,
        )
        self.assertTrue(torch.equal(
            result.deepstack_features[0],
            vision.deep_proj(x),
        ))

        grad, = torch.autograd.grad(
            result.deepstack_features[0].sum(),x
        )
        self.assertGreater(grad.abs().sum().item(),0.0)


class TestStage1CollatorByAblation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image_path = str(Path(self.temp.name) / "x.png")
        Image.new("RGB", (32, 24), (180, 100, 100)).save(self.image_path)

        self.samples = [{
            "image_path": self.image_path,
            "target_text": "No acute cardiopulmonary abnormality.",
        }]

    @staticmethod
    def biomed_tokenizer(texts):
        return torch.ones(len(texts), 256, dtype=torch.long)

    @staticmethod
    def transform(image):
        return torch.ones(3, 224, 224)

    def _collate(self, abl):
        cfg = TrainConfig(ablation_id=abl)
        needs_clip = cfg.requires_biomedclip
        collator = MIMICCXRTrainCollator(
            processor=FakeProcessor(),
            cfg=cfg,
            biomed_transform=self.transform if needs_clip else None,
            biomed_tokenizer=(
                self.biomed_tokenizer if needs_clip else None
            ),
        )
        return collator(self.samples)

    def test_a0_a1_do_not_generate_biomed_inputs(self):
        for group in ("A0", "A1"):
            with self.subTest(group=group):
                batch = self._collate(group)
                self.assertNotIn("biomed_image_tensors", batch)
                self.assertNotIn("biomed_text_tokens", batch)
                self.assertTrue(torch.any(batch["labels"].ne(-100)))

    def test_a2_a3_no_rms_include_biomed_inputs(self):
        for group in ("A2", "A3", "A3-NoRMS"):
            with self.subTest(group=group):
                batch = self._collate(group)
                self.assertEqual(
                    tuple(batch["biomed_image_tensors"].shape),
                    (1, 3, 224, 224),
                )
                self.assertEqual(
                    tuple(batch["biomed_text_tokens"].shape),
                    (1, 256),
                )
                self.assertTrue(torch.any(batch["labels"].ne(-100)))

    def test_a2_fails_if_biomed_preprocessing_missing(self):
        with self.assertRaises(ValueError):
            MIMICCXRTrainCollator(
                processor=FakeProcessor(),
                cfg=TrainConfig(ablation_id="A2"),
                biomed_transform=None,
                biomed_tokenizer=None,
            )

    def test_a1_does_not_call_biomed_transform(self):
        def forbidden(_image):
            raise AssertionError("A1 不应该调用 BioMedCLIP transform。")

        collator = MIMICCXRTrainCollator(
            processor=FakeProcessor(),
            cfg=TrainConfig(ablation_id="A1"),
            biomed_transform=forbidden,
        )
        result = collator(self.samples)
        self.assertNotIn("biomed_image_tensors", result)


class TestStage1Checkpoint(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def _write_dummy_peft(directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_model.safetensors").write_bytes(
            b"fake PEFT LoRA weights for offline tests"
        )
        (directory / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}', encoding="utf-8"
        )

    @staticmethod
    def _write_dummy_trainer_states(directory,step=100):
        (directory / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        torch.save({"state": {}}, directory / "optimizer.pt")
        torch.save({"last_epoch": 3}, directory / "scheduler.pt")
        torch.save(torch.get_rng_state(), directory / "rng_state.pth")

    def _save(self,group,step=100,final=False):
        cfg = TrainConfig(
            ablation_id=group,
            output_root=str(self.root),
        )
        model = FakePeftModel(group)
        folder = (
            Path(cfg.final_weights_dir)
            if final else
            Path(cfg.output_dir) / f"checkpoint-{step}"
        )
        self._write_dummy_peft(folder)
        if not final:
            self._write_dummy_trainer_states(folder,step)
        manifest = save_stage1_artifacts(
            model=model,
            cfg=cfg,
            directory=folder,
            global_step=step,
            train_sample_count=23,
            is_final=final,
        )
        return cfg,model,folder,manifest

    def test_a0_only_saves_lora_and_manifest(self):
        cfg,model,folder,manifest = self._save("A0")
        self.assertFalse((folder / VISUAL_ADAPTER_FILE).exists())
        self.assertTrue((folder / ARCHITECTURE_FILE).exists())
        self.assertIsNone(manifest["artifacts"]["visual_adapter"])
        validated = validate_stage1_artifacts(
            cfg=cfg,
            directory=folder,
            expected_train_sample_count=23,
            require_optimizer_state=True,
        )
        self.assertEqual(validated["ablation_id"],"A0")

    def test_all_four_visual_groups_save_complete_artifacts(self):
        for group in ("A1","A2","A3","A3-NoRMS"):
            with self.subTest(group=group):
                cfg,model,folder,manifest=self._save(group)
                self.assertTrue((folder / VISUAL_ADAPTER_FILE).exists())
                self.assertTrue((folder / ARCHITECTURE_FILE).exists())
                self.assertEqual(manifest["ablation_id"],group)
                self.assertIn(
                    "architecture_fingerprint", manifest
                )
                self.assertEqual(
                    manifest["artifacts"]["visual_adapter"]["tensor_count"],
                    len(get_visual_adapter(model).state_dict()),
                )
                validate_stage1_artifacts(
                    cfg=cfg,
                    directory=folder,
                    expected_train_sample_count=23,
                    require_optimizer_state=True,
                )

    def test_restore_visual_adapter_strict_recovers_parameters(self):
        cfg,original,folder,_=self._save("A3")
        target=FakePeftModel("A3")
        p=next(get_visual_adapter(target).parameters())

        with torch.no_grad():
            p.add_(20.0)

        self.assertFalse(torch.equal(
            p, next(get_visual_adapter(original).parameters())
        ))
        restore_visual_adapter_strict(
            target,cfg=cfg,directory=folder
        )
        for key,val in get_visual_adapter(original).state_dict().items():
            self.assertTrue(torch.equal(
                val,
                get_visual_adapter(target).state_dict()[key],
            ),key)

    def test_cross_ablation_is_rejected(self):
        cfg_a3,_,folder,_ = self._save("A3")
        cfg_nor = TrainConfig(
            ablation_id="A3-NoRMS",
            output_root=str(self.root),
        )
        with self.assertRaises(ValueError):
            validate_stage1_artifacts(
                cfg=cfg_nor,
                directory=folder,
            )

    def test_corrupted_adapter_file_refused(self):
        cfg,_,folder,_=self._save("A2")
        with (folder/VISUAL_ADAPTER_FILE).open("ab") as f:
            f.write(b"corruption")
        with self.assertRaisesRegex(ValueError,"哈希"):
            validate_stage1_artifacts(
                cfg=cfg,directory=folder
            )

    def test_corrupted_peft_config_is_rejected(self):
        cfg, model, folder, _ = self._save("A3")
        (folder / "adapter_config.json").write_text(
            '{"peft_type":"ALTERED"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "adapter_config.json"):
            validate_stage1_artifacts(
                cfg=cfg,
                directory=folder,
            )

    def test_corrupted_lora_file_refused(self):
        cfg,_,folder,_=self._save("A3")
        (folder/"adapter_model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError,"LoRA 权重哈希"):
            validate_stage1_artifacts(
                cfg=cfg,directory=folder
            )

    def test_missing_optimizer_or_rng_not_resumeable(self):
        cfg,_,folder,_ = self._save("A3")
        (folder/"optimizer.pt").unlink()
        with self.assertRaises(RuntimeError):
            validate_stage1_artifacts(
                cfg=cfg,
                directory=folder,
                require_optimizer_state=True,
            )
        validate_stage1_artifacts(
            cfg=cfg,
            directory=folder,
            require_optimizer_state=False,
        )

    def test_auto_resume_skips_incomplete_latest_checkpoint(self):
        cfg,_,first,_=self._save("A3",step=100)
        _,_,last,_=self._save("A3",step=200)
        (last/"adapter_model.safetensors").unlink()
        resume=resolve_resume_checkpoint(
            cfg=cfg,
            output_dir=cfg.output_dir,
            requested="auto",
            expected_train_sample_count=23,
        )
        self.assertEqual(Path(resume).resolve(),first.resolve())

    def test_resume_rejects_external_and_missing_run(self):
        cfg,_,folder,_=self._save("A3")
        new_cfg=TrainConfig(
            ablation_id="A3",
            output_root=str(self.root),
            run_tag="second",
        )
        with self.assertRaises(ValueError):
            resolve_resume_checkpoint(
                cfg=new_cfg,
                output_dir=new_cfg.output_dir,
                requested=str(folder),
            )

        with self.assertRaises(FileNotFoundError):
            resolve_resume_checkpoint(
                cfg=new_cfg,
                output_dir=new_cfg.output_dir,
                requested="auto",
            )

    def test_new_run_does_not_silently_overwrite_existing_checkpoint(self):
        cfg,_,folder,_=self._save("A1")
        with self.assertRaises(RuntimeError):
            resolve_resume_checkpoint(
                cfg=cfg,
                output_dir=cfg.output_dir,
                requested=None,
            )

    def test_sample_fingerprint_detects_same_count_different_rows(self):
        class Data:
            def __init__(self, samples):
                self.samples = samples

        first = fingerprint_train_dataset(
            Data([{"id": "1"}, {"id": "2"}])
        )
        same = fingerprint_train_dataset(
            Data([{"id": "1"}, {"id": "2"}])
        )
        different = fingerprint_train_dataset(
            Data([{"id": "1"}, {"id": "3"}])
        )

        self.assertEqual(first, same)
        self.assertNotEqual(first, different)

    def test_resume_rejects_sample_fingerprint_mismatch(self):
        cfg, model, folder, _ = self._save("A3")
        # 模拟已有 checkpoint 记录了某个数据集指纹。
        save_stage1_artifacts(
            model=model,
            cfg=cfg,
            directory=folder,
            global_step=100,
            train_sample_count=23,
            is_final=False,
            train_sample_digest="abc123",
        )
        validate_stage1_artifacts(
            cfg=cfg,
            directory=folder,
            expected_train_sample_digest="abc123",
        )
        with self.assertRaisesRegex(ValueError, "样本顺序"):
            validate_stage1_artifacts(
                cfg=cfg,
                directory=folder,
                expected_train_sample_digest="def456",
            )

    def test_sample_count_mismatch_rejected(self):
        cfg,_,folder,_=self._save("A3")
        with self.assertRaisesRegex(ValueError,"样本数"):
            validate_stage1_artifacts(
                cfg=cfg,
                directory=folder,
                expected_train_sample_count=24,
            )

    def test_resume_training_protocol_change_rejected(self):
        cfg,_,folder,_=self._save("A3")
        modified_cfg=TrainConfig(
            ablation_id="A3",
            output_root=str(self.root),
            lora_learning_rate=9e-5,
        )
        with self.assertRaisesRegex(ValueError,"训练协议"):
            validate_stage1_artifacts(
                cfg=modified_cfg,
                directory=folder,
                require_optimizer_state=True,
            )
        # Stage 2 继承参数时可使用不同 LR，因此不要求训练状态。
        validate_stage1_artifacts(
            cfg=modified_cfg,
            directory=folder,
            require_optimizer_state=False,
        )

    def test_callback_saves_extra_files_only_on_global_main(self):
        cfg=TrainConfig(ablation_id="A1",output_root=str(self.root))
        model=FakePeftModel("A1")
        folder=Path(cfg.output_dir)/"checkpoint-100"
        self._write_dummy_peft(folder)

        fake_transformers=types.ModuleType("transformers")
        fake_transformers.TrainerCallback=type("TrainerCallback",(),{})

        with patch.dict("sys.modules",{"transformers":fake_transformers}):
            callback=create_stage1_checkpoint_callback(cfg,23)

        args=types.SimpleNamespace(output_dir=cfg.output_dir)
        control=object()

        not_main=types.SimpleNamespace(
            is_world_process_zero=False,
            global_step=100,
        )
        self.assertIs(
            callback.on_save(args,not_main,control,model=model),
            control,
        )
        self.assertFalse((folder/ARCHITECTURE_FILE).exists())

        main_state=types.SimpleNamespace(
            is_world_process_zero=True,
            global_step=100,
        )
        callback.on_save(args,main_state,control,model=model)
        self.assertTrue((folder/ARCHITECTURE_FILE).is_file())


class TestStage1Integration(unittest.TestCase):
    def test_each_group_optimizer_is_constructible_with_real_adapter(self):
        from utils.training.discriminative_optimizer import (
            build_discriminative_adamw,
        )
        for group in ("A0","A1","A2","A3","A3-NoRMS"):
            with self.subTest(group=group):
                model=FakePeftModel(group)
                cfg=TrainConfig(ablation_id=group)
                optimizer=build_discriminative_adamw(
                    model,
                    **cfg.optimizer_kwargs(),
                )
                actual = {
                    id(p)
                    for g in optimizer.param_groups
                    for p in g["params"]
                }
                expected = {
                    id(p) for p in model.parameters() if p.requires_grad
                }
                self.assertEqual(actual,expected)

    def test_wrapper_builder_uses_exact_configuration_fields(self):
        cfg=TrainConfig(ablation_id="A3-NoRMS")
        received={}

        class DummyWrapper:
            def __init__(self,**kwargs):
                received.update(kwargs)

            def wrap(self,base):
                return "wrapped"

        import utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper as module
        with patch.object(module,"Qwen3VLLoraAndCMDASSMWrapper",DummyWrapper):
            output=entry.wrap_stage1_model(cfg, "base", "clip")
        self.assertEqual(output,"wrapped")
        self.assertEqual(received["fusion_mode"],"cm_radf")
        self.assertEqual(received["lambda_mode"],"learnable")
        self.assertIs(received["use_rms_norm"],False)
        self.assertEqual(received["biomed_extractor"],"clip")

    def test_actual_loader_only_calls_biomed_for_a2_plus(self):
        import utils.qwen3vl.qwen3_vl_8B_quant_loader as qwen_module

        calls = []

        class DummyQwenLoader:
            def __init__(self, **kwargs):
                calls.append("qwen_init")
                self.kwargs = kwargs

            def load(self):
                calls.append("qwen_load")
                processor = types.SimpleNamespace(
                    tokenizer=types.SimpleNamespace(padding_side="left")
                )
                return object(), processor

        fake_biomed_module = types.ModuleType(
            "utils.biomedclip.biomed_clip_loader"
        )

        def load_biomedclip(**kwargs):
            calls.append("biomed_load")
            return ("extractor", "transform", "tokenizer")

        fake_biomed_module.load_biomedclip = load_biomedclip

        with patch.object(
            qwen_module,
            "Qwen3VLQuantizedLoader",
            DummyQwenLoader,
        ), patch.dict(
            "sys.modules",
            {"utils.biomedclip.biomed_clip_loader": fake_biomed_module},
        ):
            for group in ("A0", "A1", "A2", "A3", "A3-NoRMS"):
                calls.clear()
                cfg = TrainConfig(ablation_id=group)
                result = entry.load_base_and_biomed(cfg, -1)
                self.assertEqual(calls[:2],["qwen_init","qwen_load"])
                self.assertEqual(
                    "biomed_load" in calls,
                    cfg.requires_biomedclip,
                )
                self.assertEqual(
                    result[1].tokenizer.padding_side,
                    "right",
                )

    def test_a1_wrapper_builder_never_passes_biomed(self):
        cfg=TrainConfig(ablation_id="A1")
        received={}

        class DummyWrapper:
            def __init__(self,**kwargs):
                received.update(kwargs)
            def wrap(self,base):
                return "wrapped"

        import utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper as module
        with patch.object(module,"Qwen3VLLoraAndCMDASSMWrapper",DummyWrapper):
            output=entry.wrap_stage1_model(cfg,"base","unused clip")
        self.assertEqual(output,"wrapped")
        self.assertIsNone(received["biomed_extractor"])
        self.assertEqual(received["fusion_mode"],"uniform")
        self.assertEqual(received["lambda_mode"],"fixed")

    def test_trainer_creation_uses_new_optimizer_and_save_callback(self):
        from utils.training.discriminative_optimizer import (
            build_discriminative_adamw,
        )
        cfg=TrainConfig(ablation_id="A2")
        model=FakePeftModel("A2")
        args=types.SimpleNamespace(
            adam_beta1=0.9, adam_beta2=0.999,adam_epsilon=1e-8
        )

        received={}

        class FakeTrainer:
            def __init__(self,**kwargs):
                received.update(kwargs)

        tf=types.ModuleType("transformers")
        tf.Trainer=FakeTrainer
        tf.TrainerCallback=type("TrainerCallback",(),{})

        with patch.dict("sys.modules",{"transformers":tf}):
            trainer=entry.create_stage1_trainer(
                cfg=cfg,
                model=model,
                train_dataset=[0,1,2],
                collator=object(),
                training_args=args,
            )
        self.assertIsInstance(trainer,FakeTrainer)
        self.assertEqual(len(received["callbacks"]),1)
        opt,scheduler=received["optimizers"]
        self.assertIsInstance(opt,torch.optim.AdamW)
        self.assertIsNone(scheduler)

    def test_training_args_keep_ddp_and_essential_columns(self):
        cfg=TrainConfig(ablation_id="A3")

        captured={}
        tf=types.ModuleType("transformers")

        def arguments(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(**kwargs)

        tf.TrainingArguments=arguments

        with patch.dict("sys.modules",{"transformers":tf}):
            result=entry.build_training_arguments(cfg)

        self.assertIs(result.remove_unused_columns,False)
        self.assertIs(result.ddp_find_unused_parameters,False)
        self.assertIs(result.save_only_model,False)
        self.assertEqual(result.max_steps,cfg.max_steps)
        self.assertEqual(result.lr_scheduler_type,cfg.lr_scheduler_type)
        self.assertEqual(result.learning_rate,cfg.lora_learning_rate)

    def test_save_final_writes_lora_visual_processor_manifest(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        cfg=TrainConfig(ablation_id="A2",output_root=temp.name)
        model=FakePeftModel("A2")
        calls=[]

        class FakeTrainer:
            def __init__(self):
                self.state=types.SimpleNamespace(global_step=200)

            def save_model(self,path):
                calls.append(("save_model",path))
                folder=Path(path)
                folder.mkdir(parents=True,exist_ok=True)
                (folder/"adapter_model.safetensors").write_bytes(b"lora")
                (folder/"adapter_config.json").write_text("{}",encoding="utf-8")

            def is_world_process_zero(self):
                return True

        class FakeProcessor:
            def save_pretrained(self,path):
                calls.append(("save_processor",path))
                (Path(path)/"processor_config.json").write_text("{}")

        entry.save_final_stage1(
            trainer=FakeTrainer(),
            model=model,
            processor=FakeProcessor(),
            cfg=cfg,
            train_sample_count=23,
        )

        path=Path(cfg.final_weights_dir)
        self.assertTrue((path/"visual_adapter.pt").is_file())
        self.assertTrue((path/"adapter_model.safetensors").is_file())
        self.assertTrue((path/"architecture.json").is_file())
        self.assertTrue((path/"processor_config.json").is_file())
        manifest=read_stage1_manifest(path)
        self.assertTrue(manifest["is_final"])
        self.assertEqual(manifest["global_step"],200)
        self.assertEqual(len(calls),2)


class TestStage1MainOffline(unittest.TestCase):
    """
    用假的加载器和假的 Trainer 执行真正 entry.main() 的控制流。

    这不是 8B 模型训练，不宣称 GPU 前向验证；
    但可确认主入口从配置到 final_weights 的调用顺序和输出协议。
    """

    def test_main_ablation_routing_and_final_artifacts(self):
        for group in ("A0", "A1", "A3-NoRMS"):
            with self.subTest(group=group):
                with tempfile.TemporaryDirectory() as directory:
                    cfg = TrainConfig(
                        ablation_id=group,
                        output_root=directory,
                    )
                    peft_model = FakePeftModel(group)
                    steps = []

                    class Processor:
                        def save_pretrained(self, path):
                            steps.append("processor_save")
                            Path(path, "processor_config.json").write_text(
                                "{}", encoding="utf-8"
                            )

                    class FakeTrainer:
                        state = types.SimpleNamespace(global_step=20)

                        def train(self, *, resume_from_checkpoint):
                            steps.append("train")
                            self.resume = resume_from_checkpoint

                        def save_model(self, path):
                            steps.append("lora_save")
                            folder = Path(path)
                            folder.mkdir(parents=True, exist_ok=True)
                            (folder / "adapter_model.safetensors").write_bytes(
                                b"LoRA mock"
                            )
                            (folder / "adapter_config.json").write_text(
                                "{}", encoding="utf-8"
                            )

                        def is_world_process_zero(self):
                            return True

                    fake_trainer = FakeTrainer()

                    class FakeDataset(list):
                        @property
                        def samples(self):
                            return self

                    def fake_load(cfg, local_rank):
                        steps.append("load_base")
                        return ("base", Processor(), None, None, None)

                    def fake_wrap(cfg, base_model, biomed_extractor):
                        steps.append("wrap")
                        self.assertEqual(cfg.ablation_id,group)
                        self.assertEqual(base_model,"base")
                        return peft_model

                    def fake_create(**kwargs):
                        steps.append("build_trainer")
                        self.assertIs(kwargs["model"],peft_model)
                        self.assertEqual(kwargs["cfg"].ablation_id,group)
                        return fake_trainer

                    with patch.object(entry,"TrainConfig",return_value=cfg), \
                         patch.object(
                             entry,"parse_stage1_trainer_options",
                             return_value=types.SimpleNamespace(
                                 resume_from_checkpoint=None,
                                 check_config_only=False,
                             ),
                         ), \
                         patch.object(
                             entry,"initialize_distributed",
                             return_value=(-1,0,1),
                         ), \
                         patch.object(
                             entry,"load_dataset_rank0_first",
                             return_value=FakeDataset([0,1,2]),
                         ), \
                         patch.object(
                             entry,"load_base_and_biomed",side_effect=fake_load,
                         ), \
                         patch.object(
                             entry,"wrap_stage1_model",side_effect=fake_wrap,
                         ), \
                         patch.object(
                             entry,"build_stage1_collator",return_value=object(),
                         ), \
                         patch.object(
                             entry,"build_training_arguments",return_value=object(),
                         ), \
                         patch.object(
                             entry,"create_stage1_trainer",side_effect=fake_create,
                         ):
                        entry.main()

                    self.assertEqual(
                        steps,
                        [
                            "load_base", "wrap", "build_trainer",
                            "train", "lora_save", "processor_save",
                        ],
                    )
                    self.assertIsNone(fake_trainer.resume)

                    folder = Path(cfg.final_weights_dir)
                    self.assertTrue((folder / "architecture.json").is_file())
                    self.assertEqual(
                        (folder / "visual_adapter.pt").exists(),
                        group != "A0",
                    )
                    self.assertEqual(
                        read_stage1_manifest(folder)["ablation_id"],
                        group,
                    )


class TestQwenQuantizedLoader(unittest.TestCase):
    def test_dtype_conversion(self):
        self.assertEqual(_resolve_torch_dtype("bfloat16"),torch.bfloat16)
        self.assertEqual(_resolve_torch_dtype("float16"),torch.float16)
        with self.assertRaises(ValueError):
            _resolve_torch_dtype("bad")

    def test_load_uses_model_and_processor_from_same_base(self):
        captured={}

        class BitsAndBytes:
            def __init__(self,**kwargs):
                self.kwargs=kwargs

        class Qwen:
            @staticmethod
            def from_pretrained(path,**kwargs):
                captured["model"]=(path,kwargs)
                return object()

        class Processor:
            @staticmethod
            def from_pretrained(path,**kwargs):
                captured["processor"]=(path,kwargs)
                return object()

        tf=types.ModuleType("transformers")
        tf.BitsAndBytesConfig=BitsAndBytes
        tf.Qwen3VLForConditionalGeneration=Qwen
        tf.AutoProcessor=Processor

        with patch.dict("sys.modules",{"transformers":tf}):
            loader=Qwen3VLQuantizedLoader(
                model_path="local_Qwen3VL",
                load_in_4bit=True,
                device_map={"":0},
            )
            loader.load()

        self.assertEqual(captured["model"][0],"local_Qwen3VL")
        self.assertEqual(captured["processor"][0],"local_Qwen3VL")
        self.assertEqual(captured["model"][1]["dtype"],torch.bfloat16)
        self.assertEqual(captured["model"][1]["device_map"],{"":0})
        self.assertEqual(
            captured["model"][1]["quantization_config"].kwargs["bnb_4bit_quant_type"],
            "nf4",
        )


if __name__ == "__main__":
    unittest.main()

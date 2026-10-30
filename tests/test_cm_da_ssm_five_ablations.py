import io
import math
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from config.cm_da_ssm.ablation_config import (
    ABLATION_IDS,
    ABLATION_RULES,
)
from config.stage1_train_config_mimic_cxr import TrainConfig
from config.slake.stage2_train_config_slake import (
    Stage2TrainConfig as SLAKE,
)
from config.vqa_rad.stage2_train_config_vqa_rad import (
    Stage2TrainConfig as RAD,
)
from config.vqa_med_2019.stage2_train_config_vqa_med_2019 import (
    Stage2TrainConfig as MED2019,
)
from config.vqa_med_2021.stage2_train_config_vqa_med_2021 import (
    Stage2TrainConfig as MED2021,
)
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_visual_adapter import (
    Qwen3VLCMDASSMVisualAdapter,
)
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper import (
    Qwen3VLLoraAndCMDASSMWrapper,
    install_cm_da_ssm_pooled_only_vision_patch,
)
from utils.training.discriminative_optimizer import (
    build_discriminative_adamw,
    format_optimizer_groups,
)


def make_config(abl="A3", stage2=False, **extra):
    cls = SLAKE if stage2 else TrainConfig
    return cls(ablation_id=abl, **extra)


def make_adapter(abl="A3", dim=32, *, zero_init_up=True):
    """
    用真实的 CM-DA-SSM 类构造小维度模型；
    仅将 CUDA Mamba 后端改为 PyTorch reference。
    """
    cfg = make_config(abl, ssm_backend="torch_reference")
    assert abl != "A0"

    return Qwen3VLCMDASSMVisualAdapter(
        hidden_dim=dim,
        r=4,
        ssm_d_state=4,
        ssm_d_conv=3,
        ssm_expand=2,
        ssm_backend="torch_reference",
        condition_hidden_dim=16,
        query_dim=8,
        biomedclip_image_encoder_dim=512,
        biomedclip_text_encoder_dim=512,
        eta=cfg.eta,
        eta_mode="fixed",
        lambda_init=cfg.lambda_init,
        residual_norm_eps=cfg.residual_norm_eps,
        residual_norm_ratio_clip=cfg.residual_norm_ratio_clip,
        zero_init_up=zero_init_up,
        fusion_mode=cfg.fusion_mode,
        lambda_mode=cfg.lambda_mode,
        use_rms_norm=cfg.use_rms_norm,
    )


class SmallPEFT(nn.Module):
    """
    只模拟真实 PEFT 模型参数结构，不加载 8B 模型。
    """
    def __init__(self, abl):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(4, 4))
        self.lora_B = nn.Parameter(torch.randn(4, 4))
        self.frozen_qwen_param = nn.Parameter(
            torch.randn(4), requires_grad=False
        )

        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.model = nn.Module()
        self.base_model.model.model.visual = nn.Module()

        if abl != "A0":
            adapter = make_adapter(abl)
            self.base_model.model.model.visual.res_adapter = adapter

    @property
    def adapter(self):
        return getattr(
            self.base_model.model.model.visual,
            "res_adapter",
            None,
        )


class TestAblationConfig(unittest.TestCase):
    def test_exact_five_ablation_ids(self):
        self.assertEqual(
            ABLATION_IDS,
            ("A0", "A1", "A2", "A3", "A3-NoRMS"),
        )

    def test_five_rules_identical_to_plan(self):
        expected = {
            "A0": (False, "none", "none", False, False),
            "A1": (True, "uniform", "fixed", True, False),
            "A2": (True, "cm_radf", "fixed", True, True),
            "A3": (True, "cm_radf", "learnable", True, True),
            "A3-NoRMS": (True, "cm_radf", "learnable", False, True),
        }
        for abl, tup in expected.items():
            with self.subTest(ablation=abl):
                cfg = make_config(abl)
                self.assertEqual(
                    (
                        cfg.enable_visual_adapter,
                        cfg.fusion_mode,
                        cfg.lambda_mode,
                        cfg.use_rms_norm,
                        cfg.requires_biomedclip,
                    ),
                    tup,
                )
                self.assertEqual(cfg.lambda_init, 0.4)

    def test_all_four_stage2_configs_inherit_own_stage1(self):
        for abl in ABLATION_IDS:
            s1 = TrainConfig(ablation_id=abl)
            paths=[]
            for cls in [SLAKE, RAD, MED2019, MED2021]:
                s2=cls(ablation_id=abl)
                with self.subTest(abl=abl,dataset=cls.__module__):
                    self.assertEqual(
                        s2.stage1_weights_dir,
                        s1.final_weights_dir,
                    )
                    self.assertEqual(
                        s2.architecture_fingerprint,
                        s1.architecture_fingerprint,
                    )
                    self.assertEqual(
                        s2.seed, s2.stage1_seed
                    )
                    paths.append(s2.output_dir)
            self.assertEqual(len(paths),len(set(paths)))

    def test_stage1_paths_must_be_separate_and_no_cross_loading(self):
        stage1=[TrainConfig(ablation_id=abl) for abl in ABLATION_IDS]
        self.assertEqual(len({c.final_weights_dir for c in stage1}),5)
        self.assertEqual(len({c.architecture_fingerprint for c in stage1}),5)

        for abl in ABLATION_IDS:
            s2=SLAKE(ablation_id=abl)
            for other in ABLATION_IDS:
                if other == abl:
                    continue
                self.assertNotEqual(
                    s2.stage1_weights_dir,
                    TrainConfig(ablation_id=other).final_weights_dir,
                )

    def test_cross_ablation_stage1_manifest_is_rejected(self):
        for abl in ABLATION_IDS:
            manifest = TrainConfig(ablation_id=abl).stage1_manifest()
            for other in ABLATION_IDS:
                s2=RAD(ablation_id=other)
                if other == abl:
                    s2.assert_compatible_stage1_manifest(manifest)
                else:
                    with self.assertRaises(ValueError):
                        s2.assert_compatible_stage1_manifest(manifest)

    def test_stage2_lr_recipe_changes_without_repointing_stage1(self):
        stage1=TrainConfig(ablation_id="A3")
        normal=SLAKE(ablation_id="A3")
        changed=SLAKE(
            ablation_id="A3",
            lr_recipe_name="Stage2LRSweep",
        )
        self.assertEqual(normal.stage1_weights_dir,stage1.final_weights_dir)
        self.assertEqual(changed.stage1_weights_dir,stage1.final_weights_dir)
        self.assertNotEqual(normal.output_dir,changed.output_dir)

    def test_stage1_seed_and_run_tag_manifest_validation(self):
        stage1=TrainConfig(ablation_id="A3",seed=314,run_tag="trial_01")
        stage2=SLAKE(
            ablation_id="A3",
            stage1_seed=314,
            stage1_run_tag="trial_01",
        )
        self.assertEqual(stage2.stage1_weights_dir,stage1.final_weights_dir)
        stage2.assert_compatible_stage1_manifest(stage1.stage1_manifest())

        with self.assertRaises(ValueError):
            SLAKE(ablation_id="A3",stage1_seed=2048).assert_compatible_stage1_manifest(
                stage1.stage1_manifest()
            )
        with self.assertRaises(ValueError):
            SLAKE(ablation_id="A3",stage1_seed=314,stage1_run_tag="wrong").assert_compatible_stage1_manifest(
                stage1.stage1_manifest()
            )

    def test_a3_and_no_rms_have_matched_budget_and_initialization(self):
        for cls in [TrainConfig, SLAKE, RAD, MED2019, MED2021]:
            a3=cls(ablation_id="A3")
            no=cls(ablation_id="A3-NoRMS")
            for name in (
                "seed", "num_train_epochs", "max_steps",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "learning_rate", "lora_learning_rate",
                "ssm_learning_rate", "visual_projection_learning_rate",
                "cm_radf_learning_rate", "lambda_learning_rate",
                "visual_norm_learning_rate", "weight_decay",
                "warmup_steps", "lr_scheduler_type",
            ):
                self.assertEqual(
                    getattr(a3,name),
                    getattr(no,name),
                    (cls.__name__,name),
                )
            self.assertNotEqual(a3.architecture_fingerprint,
                                no.architecture_fingerprint)

    def test_a3_no_rms_is_not_a3_inference_switch(self):
        self.assertFalse(TrainConfig(ablation_id="A3-NoRMS").use_rms_norm)
        self.assertTrue(SLAKE(ablation_id="A3").use_rms_norm)

    def test_invalid_modes_and_run_tags_raise(self):
        with self.assertRaises(ValueError):
            TrainConfig(ablation_id="A6")
        with self.assertRaises(ValueError):
            TrainConfig(lambda_init=0)
        with self.assertRaises(ValueError):
            TrainConfig(lambda_init=0.3)
        with self.assertRaises(ValueError):
            TrainConfig(zero_init_up=False)
        with self.assertRaises(ValueError):
            TrainConfig(run_tag="../../not_allowed")
        with self.assertRaises(ValueError):
            TrainConfig(ssm_backend="typo")
        with self.assertRaises(ValueError):
            TrainConfig(visual_adapter_r=15)

    def test_stage1_stage2_labels_safely_guard_answer(self):
        self.assertEqual(
            TrainConfig().biomed_condition_text_mode,
            "fixed_instruction",
        )
        for cls in [SLAKE,RAD,MED2019,MED2021]:
            self.assertEqual(cls().biomed_condition_text_mode,"question")

    def test_wrapper_kwargs_match_ablation_execution(self):
        for abl in ABLATION_IDS:
            cfg=TrainConfig(ablation_id=abl)
            kwargs=cfg.wrapper_kwargs()
            self.assertEqual(
                kwargs["enable_visual_adapter"], abl!="A0"
            )
            if abl not in {"A0"}:
                self.assertEqual(kwargs["fusion_mode"],cfg.fusion_mode)
                self.assertEqual(kwargs["lambda_mode"],cfg.lambda_mode)
                self.assertEqual(kwargs["use_rms_norm"],cfg.use_rms_norm)
            self.assertIsNone(kwargs["biomed_extractor"])


class TestForwardAblations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.x = torch.randn(1, 6, 32)
        self.grid = torch.tensor([[1, 4, 6]])
        self.img = torch.randn(1, 512)
        self.txt = torch.randn(1, 512)

    def test_uniform_fusion_a1_exact_directional_mean(self):
        model=make_adapter("A1",zero_init_up=False)
        self.assertFalse(hasattr(model.feature_adapter.cm_radf,"query_proj"))
        self.assertFalse(any("router_backbone" in n for n,_ in model.named_parameters()))

        y=model.feature_adapter.four_directional_ssm(
            self.x,
            (1,2,3)
        )
        r,details=model.feature_adapter.cm_radf(
            directional_features=y,
            return_details=True,
        )
        mean = y.mean(dim=2).to(dtype=model.feature_adapter.cm_radf.up.weight.dtype)
        expected=model.feature_adapter.cm_radf.up(
            model.feature_adapter.cm_radf.ln_fuse(mean)
        ).to(dtype=y.dtype)
        self.assertTrue(torch.allclose(r,expected,atol=1e-6))
        self.assertTrue(torch.equal(
            details["routing_weights"],
            torch.full((1,6,4),0.25),
        ))
        self.assertTrue(torch.allclose(
            details["routing_weights"].sum(-1),
            torch.ones(1,6),
        ))

    def test_a1_does_not_need_biomedclip_features(self):
        model=make_adapter("A1",zero_init_up=False)
        result=model(self.x, None, None, self.grid)
        self.assertEqual(tuple(result.shape),tuple(self.x.shape))

    def test_full_cmradf_has_query_and_local_keys(self):
        model=make_adapter("A2",zero_init_up=False)
        fusion=model.feature_adapter.cm_radf
        self.assertTrue(hasattr(fusion,"query_proj"))
        self.assertTrue(hasattr(fusion,"local_key_proj"))

        output, details=model(
            self.x,self.img,self.txt,self.grid,return_details=True
        )
        self.assertEqual(tuple(output.shape),tuple(self.x.shape))
        self.assertEqual(tuple(details[0]["routing_weights"].shape),(1,6,4))
        self.assertTrue(torch.allclose(
            details[0]["routing_weights"].sum(-1),
            torch.ones(1,6),
            atol=1e-6,
        ))

    def test_a1_a2_fixed_lambda_is_buffer_not_parameter(self):
        for abl in ["A1","A2"]:
            model=make_adapter(abl,zero_init_up=False)
            self.assertIsNone(model.lambda_a)
            self.assertFalse(any(
                name == "lambda_a"
                for name,_ in model.named_parameters()
            ))
            self.assertAlmostEqual(model._compute_lambda().item(),0.4,places=6)
            self.assertIn("fixed_lambda",model.state_dict())

    def test_a3_a3_no_rms_learnable_lambda_parameter(self):
        for abl in ["A3","A3-NoRMS"]:
            model=make_adapter(abl,zero_init_up=False)
            self.assertIsInstance(model.lambda_a,nn.Parameter)
            self.assertTrue(model.lambda_a.requires_grad)
            self.assertAlmostEqual(model._compute_lambda().item(),0.4,places=6)
            self.assertIn("lambda_a",model.state_dict())
            self.assertNotIn("fixed_lambda",model.state_dict())

    def test_zero_up_all_four_adapters_exact_identity(self):
        for abl in ("A1","A2","A3","A3-NoRMS"):
            model=make_adapter(abl,zero_init_up=True)
            feats=(None,None) if abl=="A1" else (self.img,self.txt)
            out=model(self.x,*feats,self.grid)
            self.assertTrue(torch.equal(out,self.x),abl)

    def test_no_rms_is_exact_rho_one_even_when_residual_nonzero(self):
        class FixedResidual(nn.Module):
            def forward(self,x,biomed_img_feat,biomed_txt_feat,grid_thw,return_details=False):
                r=torch.ones_like(x)*2
                if return_details:
                    return r,[{}]
                return r

        a3=make_adapter("A3",zero_init_up=False)
        norm_off=make_adapter("A3-NoRMS",zero_init_up=False)
        a3.feature_adapter=FixedResidual()
        norm_off.feature_adapter=FixedResidual()

        tiny_x=torch.ones_like(self.x)*10

        a3_out,a3_stats=a3(
            tiny_x,self.img,self.txt,self.grid,return_details=True
        )
        off_out,off_stats=norm_off(
            tiny_x,self.img,self.txt,self.grid,return_details=True
        )

        self.assertEqual(off_stats[0]["rho"].item(),1.0)
        self.assertAlmostEqual(
            (off_out-tiny_x).mean().item(),0.8,places=4,
        )
        self.assertGreater(a3_stats[0]["rho"].item(),1.0)
        self.assertGreater(
            (a3_out-tiny_x).mean().item(),
            (off_out-tiny_x).mean().item(),
        )

    def test_no_rms_and_a3_same_initialized_weights_but_distinct_outputs(self):
        torch.manual_seed(99)
        model_a=make_adapter("A3",zero_init_up=False)
        torch.manual_seed(99)
        model_b=make_adapter("A3-NoRMS",zero_init_up=False)

        a_state=model_a.state_dict()
        b_state=model_b.state_dict()
        self.assertEqual(set(a_state),set(b_state))
        for key in a_state:
            self.assertTrue(torch.equal(a_state[key],b_state[key]),key)

        # 为了确定观察到 rho 差异，给残差小而非零的 Up。
        with torch.no_grad():
            model_a.feature_adapter.cm_radf.up.weight.mul_(0.001)
            model_a.feature_adapter.cm_radf.up.bias.mul_(0.001)
            model_b.load_state_dict(model_a.state_dict(),strict=True)

        _,info_a=model_a(self.x,self.img,self.txt,self.grid,return_details=True)
        _,info_b=model_b(self.x,self.img,self.txt,self.grid,return_details=True)
        self.assertEqual(info_b[0]["rho"].item(),1.0)
        self.assertGreater(info_a[0]["rho"].item(),1.0)

    def test_A1_weights_independent_of_biomedclip(self):
        model=make_adapter("A1",zero_init_up=False)
        y=model.feature_adapter.four_directional_ssm(self.x,(1,2,3))
        _,d1=model.feature_adapter.cm_radf(y,return_details=True)
        _,d2=model.feature_adapter.cm_radf(
            y,
            biomed_img_feat=torch.randn(1,512),
            biomed_txt_feat=torch.randn(1,512),
            return_details=True,
        )
        self.assertTrue(torch.equal(
            d1["routing_weights"],d2["routing_weights"]
        ))

    def test_A2_requires_biomedclip_when_using_cmradf(self):
        model=make_adapter("A2",zero_init_up=False)
        with self.assertRaises(ValueError):
            model(self.x,None,None,self.grid)


class TestAblationOptimizer(unittest.TestCase):
    def build_opt(self,abl,model=None):
        model=model or SmallPEFT(abl)
        config=make_config(abl)
        opt=build_discriminative_adamw(model,**config.optimizer_kwargs())
        return model,opt

    @staticmethod
    def components(opt):
        return {g["group_name"].split("/")[0] for g in opt.param_groups}

    def test_groups_match_exact_ablation_plan(self):
        expected={
            "A0": {"lora"},
            "A1": {"lora","ssm","visual_projection","visual_norm"},
            "A2": {"lora","ssm","visual_projection","cm_radf","visual_norm"},
            "A3": {"lora","ssm","visual_projection","cm_radf","lambda","visual_norm"},
            "A3-NoRMS": {"lora","ssm","visual_projection","cm_radf","lambda","visual_norm"},
        }
        for abl in ABLATION_IDS:
            with self.subTest(abl=abl):
                model,opt=self.build_opt(abl)
                self.assertEqual(self.components(opt),expected[abl])
                assigned=[id(p) for g in opt.param_groups for p in g["params"]]
                live={id(p) for p in model.parameters() if p.requires_grad}
                self.assertEqual(set(assigned),live)
                self.assertEqual(len(assigned),len(set(assigned)))
                self.assertTrue(opt.param_groups[0]["group_name"].startswith("lora/"))
                self.assertNotIn(id(model.frozen_qwen_param),set(assigned))
                self.assertIn("lora",format_optimizer_groups(opt))

    def test_A3_no_rms_matches_A3_optimizer_group_lrs(self):
        _,a=self.build_opt("A3")
        _,b=self.build_opt("A3-NoRMS")
        ag=[(g["group_name"],g["lr"],g["weight_decay"],len(g["params"]))
            for g in a.param_groups]
        bg=[(g["group_name"],g["lr"],g["weight_decay"],len(g["params"]))
            for g in b.param_groups]
        self.assertEqual(ag,bg)

    def test_decay_exclusion_for_lambda_norm_ssm_state_and_embedding(self):
        model,opt=self.build_opt("A3")
        mapping={}
        for g in opt.param_groups:
            for p in g["params"]:
                mapping[id(p)]=g

        for n,p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                n.endswith((".bias",".A_log",".D",".direction_embeddings",".raw_eta",".lambda_a"))
                or "ln_in." in n or "ln_dir." in n or "ln_fuse." in n
                or "router_backbone.1." in n
            ):
                self.assertEqual(mapping[id(p)]["weight_decay"],0.0,n)

    def test_A1_optimizer_is_valid_without_biomedclip(self):
        model=SmallPEFT("A1")
        self.assertFalse(hasattr(model.adapter.feature_adapter.cm_radf,"router_backbone"))
        _,opt=self.build_opt("A1",model)
        self.assertNotIn("cm_radf",self.components(opt))
        self.assertNotIn("lambda",self.components(opt))

    def test_mismatched_model_flags_raise(self):
        model=SmallPEFT("A3")
        with self.assertRaises(RuntimeError):
            self.build_opt("A3-NoRMS",model)

        with self.assertRaises(RuntimeError):
            self.build_opt("A2",SmallPEFT("A3"))

        with self.assertRaises(RuntimeError):
            self.build_opt("A0",SmallPEFT("A1"))

    def test_unknown_trainable_parameter_raises(self):
        model=SmallPEFT("A3")
        model.frozen_qwen_param.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            self.build_opt("A3",model)

    def test_missing_ssm_branch_training_raises(self):
        model=SmallPEFT("A1")
        model.adapter.feature_adapter.four_directional_ssm.ssm_rr.requires_grad_(False)
        with self.assertRaises(RuntimeError):
            self.build_opt("A1",model)

    def test_no_old_optimizer_interface_or_legacy_expert_groups(self):
        model,opt=self.build_opt("A3")
        self.assertTrue(
            all("experts" not in g["group_name"]
                and "router_gate" not in g["group_name"]
                for g in opt.param_groups)
        )
        with self.assertRaises(TypeError):
            build_discriminative_adamw(
                model,
                ablation_id="A3",
                visual_expert_learning_rate=1e-4,
            )

    def test_step_updates_shared_up_with_real_adapter(self):
        model,opt=self.build_opt("A3")
        x=torch.randn(1,6,32)
        img=torch.randn(1,512)
        txt=torch.randn(1,512)
        grid=torch.tensor([[1,4,6]])
        before=model.adapter.feature_adapter.cm_radf.up.weight.detach().clone()
        loss=(model.adapter(x,img,txt,grid)-torch.randn_like(x)).square().mean()
        loss.backward()
        opt.step()
        after=model.adapter.feature_adapter.cm_radf.up.weight.detach()
        self.assertFalse(torch.equal(before,after))


class MockVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(32,32)
        self.deep_proj=nn.Linear(32,32)
        self.last_native=None

    def forward(self,x,grid_thw):
        native=types.SimpleNamespace(
            pooler_output=self.proj(x),
            deepstack_features=[self.deep_proj(x)]
        )
        self.last_native=native
        return native


class TestVisionIdentityAndDeepstack(unittest.TestCase):
    def test_all_adapter_groups_keep_deepstack_reference_unchanged(self):
        torch.manual_seed(2026)
        x=torch.randn(6,32)
        grid=torch.tensor([[1,4,6]])
        for abl in ("A1","A2","A3","A3-NoRMS"):
            with self.subTest(abl=abl):
                vision=MockVision()
                adapter=make_adapter(abl,zero_init_up=True)
                install_cm_da_ssm_pooled_only_vision_patch(vision,adapter)
                if abl!="A1":
                    vision.current_biomed_img_feat=torch.randn(1,512)
                    vision.current_biomed_txt_feat=torch.randn(1,512)
                native_expected=vision.proj(x).detach().clone()
                result=vision(x,grid)
                self.assertTrue(torch.equal(result.pooler_output,native_expected))
                self.assertIs(
                    result.deepstack_features,
                    vision.last_native.deepstack_features,
                )
                self.assertIs(
                    result.deepstack_features[0],
                    vision.last_native.deepstack_features[0],
                )

    def test_A1_wrapper_can_construct_without_biomedclip(self):
        # 不调用真实 PEFT，验证参数配置允许 A1 不带 BioMedCLIP。
        config=make_config("A1")
        wrapper=Qwen3VLLoraAndCMDASSMWrapper(
            **config.wrapper_kwargs()
        )
        self.assertIsNone(wrapper.biomed_extractor)
        self.assertEqual(wrapper.fusion_mode,"uniform")
        self.assertEqual(wrapper.lambda_mode,"fixed")


class FakeBioMed(nn.Module):
    """
    不加载真实 BioMedCLIP，只模拟冻结后返回两组 512 维特征。
    用于验证 Wrapper 的条件缓存与 A1 不加载编码器。
    """
    def __init__(self):
        super().__init__()
        self.img_proj=nn.Linear(4,512)
        self.txt_proj=nn.Linear(4,512)

    def forward(self, image, tokens):
        return (
            self.img_proj(image.float()),
            self.txt_proj(tokens.float()),
        )


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.visual=MockVision()
        self.config=types.SimpleNamespace(use_cache=True)

    def forward(self,pixel_values,grid_thw):
        return self.model.visual(pixel_values,grid_thw)


class FakePEFTForWrap(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.base_model=nn.Module()
        self.base_model.model=backbone
        self.lora_A=nn.Parameter(torch.ones(3,3))
        self.lora_B=nn.Parameter(torch.ones(3,3))

    def forward(self,pixel_values,grid_thw):
        return self.base_model.model(pixel_values,grid_thw)

    def enable_input_require_grads(self):
        pass

    def print_trainable_parameters(self):
        pass


class TestFullWrapperAblations(unittest.TestCase):
    """
    在模拟的 PEFT / Vision Tower 上运行原 Wrapper.wrap()，
    保证修改不只是直接调用 Adapter，而是接通真实的挂载代码。
    """
    def setUp(self):
        torch.manual_seed(2031)
        self.fake_peft=types.ModuleType("peft")

        def prepare(model,**kwargs):
            model.requires_grad_(False)
            return model

        self.fake_peft.prepare_model_for_kbit_training=prepare
        self.fake_peft.get_peft_model=lambda model,cfg: FakePEFTForWrap(model)
        self.fake_peft.LoraConfig=lambda **kwargs: kwargs

        self.x=torch.randn(6,32)
        self.grid=torch.tensor([[1,4,6]])

    def _wrap(self, abl):
        cfg=TrainConfig(
            ablation_id=abl,
            ssm_backend="torch_reference",
            visual_adapter_hidden_dim=32,
            visual_adapter_r=4,
            ssm_d_state=4,
        )
        bio=FakeBioMed() if cfg.requires_biomedclip else None
        args=cfg.wrapper_kwargs(biomed_extractor=bio)
        args.update(
            condition_hidden_dim=16,
            query_dim=8,
        )

        with patch.dict(sys.modules,{"peft":self.fake_peft}):
            model=Qwen3VLLoraAndCMDASSMWrapper(**args).wrap(
                FakeBackbone()
            )
        return model,cfg

    def test_A0_full_wrapper_is_pure_lora(self):
        model,cfg=self._wrap("A0")
        vision=model.base_model.model.model.visual
        self.assertFalse(hasattr(vision,"res_adapter"))

        result=model(
            pixel_values=self.x,
            grid_thw=self.grid,
            biomed_image_tensors=torch.randn(1,4),
            biomed_text_tokens=torch.randn(1,4),
        )
        self.assertEqual(tuple(result.pooler_output.shape),(6,32))
        self.assertEqual(cfg.fusion_mode,"none")

    def test_A1_full_wrapper_no_biomed_required(self):
        model,cfg=self._wrap("A1")
        vision=model.base_model.model.model.visual
        self.assertFalse(hasattr(model,"biomed_extractor"))
        self.assertEqual(vision.res_adapter.fusion_mode,"uniform")

        result=model(
            pixel_values=self.x,
            grid_thw=self.grid,
        )
        self.assertTrue(torch.equal(
            result.pooler_output,vision.proj(self.x)
        ))
        self.assertIs(
            result.deepstack_features,
            vision.last_native.deepstack_features,
        )

        # A1 即使 Collator 发送 BioMed 字段也必须忽略，而不是
        # 错误地把它们传给 Qwen 原生 forward。
        result2=model(
            pixel_values=self.x,
            grid_thw=self.grid,
            biomed_image_tensors=torch.randn(1,4),
            biomed_text_tokens=torch.randn(1,4),
        )
        self.assertTrue(torch.equal(
            result2.pooler_output,vision.proj(self.x)
        ))

    def test_A2_full_wrapper_uses_frozen_biomed(self):
        model,cfg=self._wrap("A2")
        vision=model.base_model.model.model.visual

        self.assertTrue(hasattr(model,"biomed_extractor"))
        self.assertFalse(any(
            p.requires_grad
            for p in model.biomed_extractor.parameters()
        ))

        result=model(
            pixel_values=self.x,
            grid_thw=self.grid,
            biomed_image_tensors=torch.randn(1,4),
            biomed_text_tokens=torch.randn(1,4),
        )

        self.assertTrue(torch.equal(
            result.pooler_output,vision.proj(self.x)
        ))
        self.assertIs(
            result.deepstack_features,
            vision.last_native.deepstack_features,
        )
        self.assertIsNone(vision.current_biomed_img_feat)
        self.assertIsNone(vision.current_biomed_txt_feat)



if __name__ == "__main__":
    unittest.main()

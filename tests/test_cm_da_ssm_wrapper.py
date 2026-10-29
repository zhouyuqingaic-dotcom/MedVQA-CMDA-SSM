import io
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_visual_adapter import (
    Qwen3VLCMDASSMVisualAdapter,
)
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_lora_wrapper import (
    FrozenBioMedCLIPFeatureExtractor,
    Qwen3VLLoraAndCMDASSMWrapper,
    install_cm_da_ssm_biomed_forward_patch,
    install_cm_da_ssm_pooled_only_vision_patch,
)


def create_adapter(
    *,
    zero_init_up=True,
    hidden_dim=32,
    lambda_init=0.2,
    residual_norm_ratio_clip=10.0,
):
    """
    在 CPU 测试时使用真实的 4D-SSM + CM-RADF 算法逻辑。
    仅将 Mamba CUDA 后端替换为 torch_reference。
    """
    return Qwen3VLCMDASSMVisualAdapter(
        hidden_dim=hidden_dim,
        r=4 if hidden_dim == 32 else 16,
        ssm_d_state=4,
        ssm_d_conv=3,
        ssm_expand=2,
        ssm_backend="torch_reference",
        query_dim=64,
        condition_hidden_dim=128,
        eta=1.0,
        zero_init_up=zero_init_up,
        lambda_init=lambda_init,
        residual_norm_ratio_clip=residual_norm_ratio_clip,
    )


class FakeQwenVisionTower(nn.Module):
    """
    只模拟 Qwen Vision Tower 的输出契约：
        .pooler_output
        .deepstack_features

    模拟的 DeepStack 由单独的可微线性层产生。
    测试可以检查 wrapper 是否改动了同一个原生输出对象。

    这里并不是实际的预训练 Qwen3-VL 模型。
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        self.pooled_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.deepstack_proj1 = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.deepstack_proj2 = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.last_native_outputs = None
        self.native_forward_count = 0

    def forward(self, hidden_states, grid_thw):
        self.native_forward_count += 1

        # 简化版原生 Qwen Vision 输出。
        outputs = types.SimpleNamespace(
            pooler_output=self.pooled_proj(hidden_states),
            deepstack_features=[
                self.deepstack_proj1(hidden_states),
                self.deepstack_proj2(hidden_states),
            ],
        )

        # 保存原生输出的对象引用：
        # patch 执行前后的 DeepStack 必须完全相同。
        self.last_native_outputs = outputs
        return outputs


class DummyResidual(nn.Module):
    """
    单独验证 RMS 数值时使用的受控残差生成器。
    不用于真实视觉适配器模型。
    """

    def __init__(self, residual):
        super().__init__()
        self.register_buffer(
            "stored_residual",
            residual,
        )

    def forward(
        self,
        x,
        biomed_img_feat,
        biomed_txt_feat,
        grid_thw,
        return_details=False,
    ):
        residual = self.stored_residual.to(
            device=x.device,
            dtype=x.dtype,
        )

        if return_details:
            return residual, [
                {},
                {},
            ]
        return residual


class TestCMDASSMVisualResidualAdapter(unittest.TestCase):
    """
    完整视觉残差适配器测试。

    核验：
        - 输入/输出的形状、dtype；
        - Shared Up=0 时完全恒等；
        - lambda 初始值 0.2；
        - Capped RMS 按图像独立计算；
        - BF16 支持；
        - 非零残差的梯度回传与参数保存。
    """

    def setUp(self):
        torch.manual_seed(314159)

        self.grid_thw = torch.tensor(
            [
                [1, 4, 6],  # merger 后 2x3，N=6
                [1, 2, 4],  # merger 后 1x2，N=2
            ],
            dtype=torch.long,
        )

        self.x = torch.randn(
            1,
            8,
            32,
        )

        self.img = torch.randn(
            2, 512,
        )

        self.txt = torch.randn(
            2, 512,
        )

    def test_zero_up_produces_exact_identity_3d(self):
        model = create_adapter(zero_init_up=True)

        out, details = model(
            self.x,
            self.img,
            self.txt,
            self.grid_thw,
            return_details=True,
        )

        self.assertTrue(
            torch.equal(out, self.x)
        )

        self.assertEqual(
            tuple(out.shape),
            (1, 8, 32),
        )

        self.assertEqual(
            out.dtype,
            self.x.dtype,
        )

        self.assertEqual(
            len(details),
            2,
        )

        self.assertAlmostEqual(
            details[0]["lambda"].item(),
            0.2,
            places=6,
        )

        self.assertAlmostEqual(
            details[1]["lambda"].item(),
            0.2,
            places=6,
        )

        # 残差为 0 时 rho 仍是有限非负值（因为 eps）。
        # 真实的注入幅度 delta_rms 必须严格为 0。
        for item in details:
            self.assertEqual(item["delta_rms"].item(), 0.0)
            self.assertTrue(
                torch.isfinite(item["rho"]).item()
            )
            self.assertGreaterEqual(
                item["rho"].item(),
                0.0,
            )
            self.assertLessEqual(
                item["rho"].item(),
                10.0,
            )

    def test_zero_up_produces_exact_identity_2d(self):
        model = create_adapter()

        x2 = self.x.squeeze(0)

        out = model(
            x2,
            self.img,
            self.txt,
            self.grid_thw,
        )

        self.assertEqual(
            tuple(out.shape),
            tuple(x2.shape),
        )

        self.assertTrue(
            torch.equal(out, x2)
        )

    def test_bfloat16_zero_up_still_exact_identity(self):
        model = create_adapter()

        # 模拟 Qwen3-VL BF16；lambda_a 仍保留 FP32。
        model.feature_adapter.to(
            dtype=torch.bfloat16,
        )

        self.assertEqual(
            model.lambda_a.dtype,
            torch.float32,
        )

        x = self.x.bfloat16()

        out = model(
            x,
            self.img,
            self.txt,
            self.grid_thw,
        )

        self.assertEqual(
            out.dtype,
            torch.bfloat16,
        )

        self.assertTrue(
            torch.equal(out, x)
        )

    def test_rms_is_computed_per_visual_item_not_across_batch(self):
        model = create_adapter(
            zero_init_up=False,
            residual_norm_ratio_clip=3.0,
        )

        # 令两张图片 pooled token 范围差距明显。
        x = torch.cat(
            (
                torch.full((1, 6, 32), 2.0),
                torch.full((1, 2, 32), 20.0),
            ),
            dim=1,
        )

        # 两张图的未校准残差都固定为 1：
        # 第一张 rho 接近 2；
        # 第二张 rho 接近 20，必须截断到 3。
        model.feature_adapter = DummyResidual(
            torch.ones_like(x)
        )

        out, details = model(
            x=x,
            biomed_img_feat=self.img,
            biomed_txt_feat=self.txt,
            grid_thw=self.grid_thw,
            return_details=True,
        )

        rho1 = details[0]["rho"].item()
        rho2 = details[1]["rho"].item()

        expected1 = min(
            ((4.0 + 1e-6) / (1.0 + 1e-6)) ** 0.5,
            3.0,
        )

        expected2 = 3.0

        self.assertAlmostEqual(
            rho1,
            expected1,
            places=5,
        )
        self.assertAlmostEqual(
            rho2,
            expected2,
            places=5,
        )

        self.assertNotAlmostEqual(
            rho1,
            rho2,
            places=2,
        )

        expected = x.clone()
        expected[:, :6] += 0.2 * expected1
        expected[:, 6:] += 0.2 * expected2

        self.assertTrue(
            torch.allclose(
                out,
                expected,
                atol=1e-5,
            )
        )

    def test_nonzero_residual_changes_only_pooled_tokens(self):
        model = create_adapter(
            zero_init_up=False,
        )

        out, details = model(
            self.x,
            self.img,
            self.txt,
            self.grid_thw,
            return_details=True,
        )

        self.assertEqual(
            out.shape,
            self.x.shape,
        )

        self.assertGreater(
            (out - self.x).abs().sum().item(),
            0.0,
        )

        # 缩放系数受 rho_max 截断。
        for item in details:
            self.assertLessEqual(
                item["rho"].item(),
                10.0 + 1e-6,
            )

    def test_lambda_and_four_ssm_receive_gradient_after_up_updates(self):
        model = create_adapter(zero_init_up=True)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.2,
        )

        target = torch.randn_like(self.x)

        # 初始化 Up=0，第一步 Up 能获得梯度。
        optimizer.zero_grad(set_to_none=True)

        out = model(
            self.x,
            self.img,
            self.txt,
            self.grid_thw,
        )

        loss = (out - target).square().mean()
        loss.backward()

        up = model.feature_adapter.cm_radf.up

        self.assertIsNotNone(up.weight.grad)
        self.assertGreater(
            up.weight.grad.abs().sum().item(),
            0,
        )

        optimizer.step()

        # 更新 Up 后，整个计算图中的 SSM、Query、Key、lambda
        # 应能获得梯度。
        optimizer.zero_grad(set_to_none=True)

        out2 = model(
            self.x,
            self.img,
            self.txt,
            self.grid_thw,
        )

        loss2 = (
            out2 - target
        ).square().mean()
        loss2.backward()

        ssm = model.feature_adapter.four_directional_ssm
        fusion = model.feature_adapter.cm_radf

        checked = {
            "SSM_RF": ssm.ssm_rf.in_proj.weight,
            "SSM_RR": ssm.ssm_rr.in_proj.weight,
            "SSM_CF": ssm.ssm_cf.in_proj.weight,
            "SSM_CR": ssm.ssm_cr.in_proj.weight,
            "SharedDown": ssm.down.weight,
            "Query": fusion.query_proj.weight,
            "GlobalKey": fusion.global_key_proj.weight,
            "LocalKey": fusion.local_key_proj.weight,
            "Lambda": model.lambda_a,
        }

        for name, parameter in checked.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(
                    parameter.grad
                )
                self.assertGreater(
                    parameter.grad.abs().sum().item(),
                    0.0,
                )

    def test_visual_adapter_state_dict_can_load_strictly(self):
        source = create_adapter(
            zero_init_up=False
        )
        target = create_adapter(
            zero_init_up=True
        )

        archive = io.BytesIO()
        torch.save(
            source.state_dict(),
            archive,
        )

        archive.seek(0)
        state = torch.load(
            archive,
            weights_only=True,
        )

        target.load_state_dict(
            state,
            strict=True,
        )

        a = source(
            self.x,
            self.img,
            self.txt,
            self.grid_thw,
        )

        b = target(
            self.x,
            self.img,
            self.txt,
            self.grid_thw,
        )

        self.assertTrue(
            torch.equal(a, b)
        )


    def test_qwen_4096_dim_zero_up_identity(self):
        # 使用目标 Qwen3-VL-8B pooled token 维度 D=4096，
        # Shared Down 维度 C=4096/16=256。
        #
        # 仅测试一个 2x2 合并后网格，避免 CPU 上执行长序列。
        model = create_adapter(
            zero_init_up=True,
            hidden_dim=4096,
        )

        x = torch.randn(4, 4096)
        grid = torch.tensor([[1, 4, 4]])

        with torch.no_grad():
            result = model(
                x=x,
                biomed_img_feat=self.img[:1],
                biomed_txt_feat=self.txt[:1],
                grid_thw=grid,
            )

        self.assertEqual(
            model.feature_adapter.four_directional_ssm.bottleneck_dim,
            256,
        )
        self.assertEqual(
            tuple(result.shape),
            (4, 4096),
        )
        self.assertTrue(
            torch.equal(result, x)
        )

    def test_invalid_lambda_and_rms_configurations_raise(self):
        for invalid in (0.0, 1.0, -0.1, float("nan")):
            with self.subTest(lambda_init=invalid):
                with self.assertRaises(ValueError):
                    create_adapter(lambda_init=invalid)

        for invalid in (0.0, -1.0, float("inf")):
            with self.subTest(clip=invalid):
                with self.assertRaises(ValueError):
                    create_adapter(
                        residual_norm_ratio_clip=invalid
                    )


class TestPooledOnlyVisionPatch(unittest.TestCase):
    """
    通过假的 Qwen3-VL Vision Tower 测试真实的 patch 函数。

    不依赖 Transformers/PEFT，也不加载预训练权重。
    """

    def setUp(self):
        torch.manual_seed(271828)

        self.vision = FakeQwenVisionTower()
        self.adapter = create_adapter()

        self.grid_thw = torch.tensor(
            [[1, 4, 6]]
        )

        self.x = torch.randn(
            6,
            32,
            requires_grad=True,
        )

        self.vision = install_cm_da_ssm_pooled_only_vision_patch(
            self.vision,
            self.adapter,
        )

        self.vision.current_biomed_img_feat = torch.randn(
            1, 512,
        )
        self.vision.current_biomed_txt_feat = torch.randn(
            1, 512,
        )

    def test_pooled_identity_and_deepstack_reference_unchanged(self):
        outputs = self.vision(
            self.x,
            self.grid_thw,
        )

        native = self.vision.last_native_outputs

        # Qwen 输出对象本身仍是原生返回对象。
        self.assertIs(
            outputs,
            native,
        )

        expected_pooled = self.vision.pooled_proj(
            self.x
        )

        self.assertTrue(
            torch.equal(
                outputs.pooler_output,
                expected_pooled,
            )
        )

        # DeepStack 的 list 对象引用不变。
        self.assertIs(
            outputs.deepstack_features,
            native.deepstack_features,
        )

        # 使用额外 snapshot/记录避免只验证“值相同”。
        for index, native_deepstack in enumerate(
            native.deepstack_features
        ):
            with self.subTest(deepstack_index=index):
                self.assertIs(
                    outputs.deepstack_features[index],
                    native_deepstack,
                )

                self.assertEqual(
                    outputs.deepstack_features[index].data_ptr(),
                    native_deepstack.data_ptr(),
                )

                self.assertTrue(
                    torch.equal(
                        outputs.deepstack_features[index],
                        native_deepstack,
                    )
                )

        self.assertEqual(
            tuple(outputs.pooler_output.shape),
            tuple(expected_pooled.shape),
        )

    def test_nonzero_adapter_changes_pooled_not_deepstack(self):
        # 修改 Shared Up，使残差路径真实生效。
        with torch.no_grad():
            nn.init.normal_(
                self.adapter.feature_adapter.cm_radf.up.weight,
                std=0.05,
            )

        outputs = self.vision(
            self.x,
            self.grid_thw,
        )
        native = self.vision.last_native_outputs

        original_pooled = self.vision.pooled_proj(
            self.x
        )

        self.assertGreater(
            (
                outputs.pooler_output
                - original_pooled
            ).abs().sum().item(),
            0.0,
        )

        self.assertIs(
            outputs.deepstack_features,
            native.deepstack_features,
        )

        expected_deepstack = [
            self.vision.deepstack_proj1(self.x),
            self.vision.deepstack_proj2(self.x),
        ]

        for actual, expected in zip(
            outputs.deepstack_features,
            expected_deepstack,
        ):
            self.assertTrue(
                torch.equal(actual, expected)
            )

        # 深层原生梯度仍能单独回传到 Qwen 输出输入。
        grad, = torch.autograd.grad(
            outputs.deepstack_features[0].sum(),
            self.x,
        )

        self.assertIsNotNone(grad)
        self.assertGreater(
            grad.abs().sum().item(),
            0.0,
        )

    def test_res_adapter_is_called_exactly_once_per_vision_forward(self):
        calls = []

        def on_forward(module, args, output):
            calls.append(output)

        handle = self.vision.res_adapter.register_forward_hook(
            on_forward
        )

        try:
            self.vision(
                self.x,
                self.grid_thw,
            )
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            tuple(calls[0].shape),
            (6, 32),
        )

    def test_missing_biomed_features_raise(self):
        self.vision.current_biomed_img_feat = None

        with self.assertRaises(ValueError):
            self.vision(
                self.x,
                self.grid_thw,
            )

    def test_missing_pooler_output_raises(self):
        class BrokenVision(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.ones(1)
                )

            def forward(self, x, grid_thw):
                return types.SimpleNamespace(
                    deepstack_features=[x],
                )

        broken = install_cm_da_ssm_pooled_only_vision_patch(
            BrokenVision(),
            create_adapter(),
        )

        broken.current_biomed_img_feat = torch.randn(1, 512)
        broken.current_biomed_txt_feat = torch.randn(1, 512)

        with self.assertRaises(RuntimeError):
            broken(
                self.x,
                self.grid_thw,
            )

    def test_cannot_patch_same_tower_twice(self):
        with self.assertRaises(RuntimeError):
            install_cm_da_ssm_pooled_only_vision_patch(
                self.vision,
                create_adapter(),
            )


class FakeBioMedExtractor(nn.Module):
    """
    已冻结的假特征提取器，仅用于 forward patch 测试。
    """

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(
            4,
            512,
        )
        self.calls = 0

    def forward(self, biomed_img_tensors, biomed_txt_tokens):
        self.calls += 1
        self.last_image_dtype = biomed_img_tensors.dtype
        self.last_text_dtype = biomed_txt_tokens.dtype

        # 模拟实际 BioMedCLIP 图像路径：
        # 图片输入应提前与冻结编码器权重的 dtype 对齐。
        #
        # 文本路径的真实实现使用 Embedding 接收 long tokens；
        # 此处用线性层代替，故仅在假模型内部转换文本 dtype。
        return (
            self.projection(biomed_img_tensors),
            self.projection(
                biomed_txt_tokens.to(
                    dtype=self.projection.weight.dtype
                )
            ),
        )


class FakeQwenBackbone(nn.Module):
    """
    与原 MoRA-Med PEFT wrapper 的路径兼容：
        peft_model.base_model.model.model.visual
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Module()
        self.model.visual = FakeQwenVisionTower()

        self.config = types.SimpleNamespace(
            use_cache=True,
        )

    def forward(self, pixel_values, grid_thw):
        return self.model.visual(
            pixel_values,
            grid_thw,
        )


class FakePEFTModel(nn.Module):
    """
    假 PEFT 对象，只用于在没有真实 peft 包时验证挂载逻辑。
    """

    def __init__(self, base):
        super().__init__()

        self.base_model = nn.Module()
        self.base_model.model = base

        self.lora_placeholder = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(self, pixel_values, grid_thw):
        return self.base_model.model(
            pixel_values,
            grid_thw,
        )

    def enable_input_require_grads(self):
        pass

    def print_trainable_parameters(self):
        pass


class TestCMDAssmFullWrapper(unittest.TestCase):
    """
    模拟 prepare_model_for_kbit_training + PEFT 装配流程。

    这样不仅能测试独立 vision patch，还能检查真正的 wrap()：
        - res_adapter 是否注册为可训练子模块；
        - DeepStack 在经过外层 forward 后是否保留；
        - BioMedCLIP 是否只对当前 batch 编码一次；
        - 缓存是否被 finally 清空；
        - lambda_a 是否维持 FP32；
        - A0 / LoRA-only 是否可用。
    """

    def setUp(self):
        torch.manual_seed(1989)

        self.fake_peft = types.ModuleType("peft")

        def fake_prepare(model, **kwargs):
            # 模拟 QLoRA 的冻结骨干行为。
            model.requires_grad_(False)
            return model

        def fake_get_peft_model(model, config):
            return FakePEFTModel(model)

        self.fake_peft.prepare_model_for_kbit_training = fake_prepare
        self.fake_peft.get_peft_model = fake_get_peft_model
        self.fake_peft.LoraConfig = lambda **kwargs: kwargs

        self.grid_thw = torch.tensor(
            [[1, 4, 6]]
        )
        self.x = torch.randn(6, 32)

        self.bio_img = torch.randn(1, 4)
        self.bio_txt = torch.randint(
            0, 20, (1, 4),
        )

    def _make_wrapper(self, enable=True):
        return Qwen3VLLoraAndCMDASSMWrapper(
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target_modules=["q_proj", "v_proj"],
            gradient_checkpointing=False,
            visual_adapter_hidden_dim=32,
            visual_adapter_r=4,
            enable_visual_adapter=enable,
            biomed_extractor=FakeBioMedExtractor() if enable else None,
            ssm_backend="torch_reference",
            ssm_d_state=4,
            ssm_d_conv=3,
            ssm_expand=2,
            lambda_init=0.2,
            zero_init_up=True,
        )

    def test_wrap_registers_adapter_and_preserves_deepstack(self):
        wrapper = self._make_wrapper()

        with patch.dict(
            sys.modules,
            {"peft": self.fake_peft},
        ):
            peft_model = wrapper.wrap(
                FakeQwenBackbone()
            )

        vision = peft_model.base_model.model.model.visual

        self.assertIs(
            vision.res_adapter.feature_adapter,
            vision.res_adapter.feature_adapter,
        )

        self.assertTrue(
            hasattr(vision, "original_forward")
        )

        self.assertEqual(
            vision.res_adapter.lambda_a.dtype,
            torch.float32,
        )

        self.assertAlmostEqual(
            vision.res_adapter._compute_lambda().item(),
            0.2,
            places=6,
        )

        trainable = {
            name for name, parameter in peft_model.named_parameters()
            if parameter.requires_grad
        }

        self.assertTrue(
            any("res_adapter.feature_adapter" in name for name in trainable)
        )
        self.assertTrue(
            any("res_adapter.lambda_a" in name for name in trainable)
        )
        self.assertFalse(
            any("biomed_extractor" in name for name in trainable)
        )

        # 先显式调用 model.train()，模拟 Trainer 的标准行为。
        peft_model.train()

        outputs = peft_model(
            pixel_values=self.x,
            grid_thw=self.grid_thw,
            biomed_image_tensors=self.bio_img,
            biomed_text_tokens=self.bio_txt,
        )

        # 在 Zero-Up 初始化时输出保持恒等。
        native_pooled = vision.pooled_proj(
            self.x
        )
        self.assertTrue(
            torch.equal(
                outputs.pooler_output,
                native_pooled,
            )
        )

        native_outputs = vision.last_native_outputs

        self.assertIs(
            outputs.deepstack_features,
            native_outputs.deepstack_features,
        )

        self.assertEqual(
            peft_model.biomed_extractor.calls,
            1,
        )

        self.assertFalse(
            peft_model.biomed_extractor.training
        )

        self.assertIsNone(
            vision.current_biomed_img_feat
        )
        self.assertIsNone(
            vision.current_biomed_txt_feat
        )


    def test_wrap_bfloat16_vision_with_fp32_biomed_image(self):
        # 模拟真实 Trainer 常见组合：
        #   Qwen Vision 及 BioMedCLIP 权重为 BF16；
        #   Collator 输出的 biomed_image_tensors 仍是 FP32；
        #   BiomedCLIP text tokens 仍是 torch.long。
        #
        # Wrapper 应在编码前将图像转换成 BioMedCLIP 权重 dtype，
        # 而不是要求 Trainer 预先为所有数据手动转精度。
        wrapper = self._make_wrapper()

        base = FakeQwenBackbone().to(
            dtype=torch.bfloat16
        )

        with patch.dict(sys.modules, {"peft": self.fake_peft}):
            model = wrapper.wrap(base)

        vision = model.base_model.model.model.visual

        self.assertEqual(
            vision.res_adapter.lambda_a.dtype,
            torch.float32,
        )

        self.assertEqual(
            vision.res_adapter.feature_adapter.cm_radf.up.weight.dtype,
            torch.bfloat16,
        )

        self.assertEqual(
            model.biomed_extractor.projection.weight.dtype,
            torch.bfloat16,
        )

        output = model(
            pixel_values=self.x.bfloat16(),
            grid_thw=self.grid_thw,
            biomed_image_tensors=self.bio_img.float(),
            biomed_text_tokens=self.bio_txt.long(),
        )

        expected_pooled = vision.pooled_proj(
            self.x.bfloat16()
        )

        self.assertTrue(
            torch.equal(
                output.pooler_output,
                expected_pooled,
            )
        )

        self.assertEqual(
            output.pooler_output.dtype,
            torch.bfloat16,
        )

        # 明确验证外层 patch 完成了图像精度转换，
        # 且没有擅自把文本 token 从 long 改成浮点。
        self.assertEqual(
            model.biomed_extractor.last_image_dtype,
            torch.bfloat16,
        )
        self.assertEqual(
            model.biomed_extractor.last_text_dtype,
            torch.long,
        )

        self.assertIs(
            output.deepstack_features,
            vision.last_native_outputs.deepstack_features,
        )

    def test_outer_patch_drops_biomed_kwargs_and_clears_cache_after_exception(self):
        wrapper = self._make_wrapper()

        with patch.dict(sys.modules, {"peft": self.fake_peft}):
            model = wrapper.wrap(
                FakeQwenBackbone()
            )

        vision = model.base_model.model.model.visual

        # 不匹配的 grid_thw 会在 Adapter 中抛异常，
        # 外层 finally 仍应清掉 BioMedCLIP 条件缓存。
        with self.assertRaises(ValueError):
            model(
                pixel_values=self.x,
                grid_thw=torch.tensor([[1, 8, 8]]),
                biomed_image_tensors=self.bio_img,
                biomed_text_tokens=self.bio_txt,
            )

        self.assertIsNone(
            vision.current_biomed_img_feat
        )
        self.assertIsNone(
            vision.current_biomed_txt_feat
        )

        # 缺失 BioMedCLIP 条件不能静默使用上一次的特征。
        with self.assertRaises(ValueError):
            model(
                pixel_values=self.x,
                grid_thw=self.grid_thw,
            )

    def test_lora_only_path_drops_unused_biomed_kwargs(self):
        wrapper = self._make_wrapper(enable=False)

        with patch.dict(sys.modules, {"peft": self.fake_peft}):
            model = wrapper.wrap(
                FakeQwenBackbone()
            )

        output = model(
            pixel_values=self.x,
            grid_thw=self.grid_thw,
            biomed_image_tensors=self.bio_img,
            biomed_text_tokens=self.bio_txt,
        )

        self.assertEqual(
            tuple(output.pooler_output.shape),
            (6, 32),
        )

        self.assertFalse(
            hasattr(
                model.base_model.model.model.visual,
                "res_adapter",
            )
        )


class DummyRawBioMed(nn.Module):
    """
    测试新 FrozenBioMedCLIPFeatureExtractor.train(False) 行为。
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 512)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, image, text):
        a = self.dropout(self.proj(image.float()))
        b = self.dropout(self.proj(text.float()))

        return a, b, None


class TestFrozenBioMedClip(unittest.TestCase):
    def test_stays_in_eval_even_when_train_is_called(self):
        extractor = FrozenBioMedCLIPFeatureExtractor(
            DummyRawBioMed()
        )

        extractor.train(True)

        self.assertFalse(
            extractor.training
        )
        self.assertFalse(
            extractor.model.training
        )

        self.assertFalse(
            any(p.requires_grad for p in extractor.parameters())
        )

        img = torch.randn(1, 4)
        txt = torch.randn(1, 4)

        a, b = extractor(img, txt)

        self.assertFalse(
            a.requires_grad
        )
        self.assertFalse(
            b.requires_grad
        )

        self.assertTrue(torch.isfinite(a).all().item())
        self.assertTrue(torch.isfinite(b).all().item())


if __name__ == "__main__":
    unittest.main()

import io
import math
import unittest

import torch

from utils.qwen3vl.qwen3_vl_8B_global_local_cm_radf import (
    GlobalLocalCMRADF,
)
from utils.qwen3vl.qwen3_vl_8B_four_directional_cm_radf import (
    Qwen3VLFourDirectionalCMRADF,
)
from utils.qwen3vl.qwen3_vl_8B_cm_da_ssm_utils import (
    build_biomedclip_route_features,
)


class TestGlobalLocalCMRADF(unittest.TestCase):
    """
    CM-RADF 核心模块测试。

    独立验证：
        - BiomedCLIP 2049 维先验是否与原项目完全一致；
        - Global / Local Key 和点积分数是否符合公式；
        - Softmax 方向和 token 级权重；
        - eta=0 的 Global-only 行为；
        - Shared Up 零初始化与后续梯度恢复。
    """

    def setUp(self):
        torch.manual_seed(2026)

        self.model = GlobalLocalCMRADF(
            hidden_dim=32,
            bottleneck_dim=8,
            biomedclip_image_encoder_dim=512,
            biomedclip_text_encoder_dim=512,
            condition_hidden_dim=128,
            query_dim=64,
            eta=1.0,
            eta_mode="fixed",
            zero_init_up=True,
        )

        self.y = torch.randn(
            1, 6, 4, 8,
        )

        self.biomed_img = torch.randn(
            1, 512,
        )
        self.biomed_txt = torch.randn(
            1, 512,
        )

    def _run(self, model=None, y=None, txt=None):
        return (model or self.model)(
            directional_features=self.y if y is None else y,
            biomed_img_feat=self.biomed_img,
            biomed_txt_feat=self.biomed_txt if txt is None else txt,
            return_details=True,
        )

    def test_2049_prior_is_identical_to_original_mora_med(self):
        captured_inputs = []

        # 捕获 Condition MLP 的真正输入，而不是自行重写先验。
        hook = self.model.router_backbone[0].register_forward_pre_hook(
            lambda module, args: captured_inputs.append(
                args[0].detach().clone()
            )
        )

        try:
            self._run()
        finally:
            hook.remove()

        expected = build_biomedclip_route_features(
            biomed_img_feat=self.biomed_img.float(),
            biomed_txt_feat=self.biomed_txt.float(),
            use_cross_modal_prior=True,
        )

        self.assertEqual(
            tuple(captured_inputs[0].shape),
            (1, 2049),
        )

        self.assertTrue(
            torch.equal(
                captured_inputs[0],
                expected,
            )
        )

        # 旧特征构建公式中的 cos 为最后一个元素。
        self.assertTrue(
            torch.allclose(
                expected[:, -1:],
                torch.nn.functional.cosine_similarity(
                    self.biomed_img,
                    self.biomed_txt,
                    dim=-1,
                ).unsqueeze(-1),
                atol=1e-6,
            )
        )

    def test_forward_shapes_and_zero_initial_residual(self):
        residual, details = self._run()

        self.assertEqual(
            tuple(residual.shape),
            (1, 6, 32),
        )

        self.assertTrue(
            torch.equal(
                residual,
                torch.zeros_like(residual),
            )
        )

        expected_shapes = {
            "query": (1, 64),
            "global_scores": (1, 4),
            "local_scores": (1, 6, 4),
            "combined_scores": (1, 6, 4),
            "routing_weights": (1, 6, 4),
            "mean_direction_weights": (1, 4),
            "fused_directional_features": (1, 6, 8),
        }

        for key, expected_shape in expected_shapes.items():
            with self.subTest(name=key):
                self.assertEqual(
                    tuple(details[key].shape),
                    expected_shape,
                )

        self.assertEqual(
            details["eta"].item(),
            1.0,
        )

    def test_global_and_local_scores_match_the_equations(self):
        _, details = self._run()
        q = details["query"].float()

        global_keys = (
            self.model.global_key_proj(
                self.y.mean(dim=1)
            )
            + self.model.direction_embeddings.unsqueeze(0)
        )

        expected_global = torch.einsum(
            "bq,bdq->bd",
            q,
            global_keys,
        ) / math.sqrt(64.0)

        local_keys = self.model.local_key_proj(
            self.y
        )

        expected_local = torch.einsum(
            "bq,bndq->bnd",
            q,
            local_keys,
        ) / math.sqrt(64.0)

        self.assertTrue(
            torch.allclose(
                details["global_scores"],
                expected_global,
                atol=1e-6,
            )
        )

        self.assertTrue(
            torch.allclose(
                details["local_scores"],
                expected_local,
                atol=1e-6,
            )
        )

        expected_scores = (
            expected_global.unsqueeze(1)
            + expected_local
        )

        self.assertTrue(
            torch.allclose(
                details["combined_scores"],
                expected_scores,
                atol=1e-6,
            )
        )

    def test_softmax_over_directions_and_correct_weighted_fusion(self):
        _, details = self._run()

        beta = details["routing_weights"]

        # 对每个 token，四方向权重之和必须为 1。
        self.assertTrue(
            torch.allclose(
                beta.sum(dim=-1),
                torch.ones(1, 6),
                atol=1e-7,
            )
        )

        expected_beta = torch.softmax(
            details["combined_scores"],
            dim=-1,
        )
        self.assertTrue(
            torch.allclose(
                beta,
                expected_beta,
                atol=1e-7,
            )
        )

        # M[n] = sum_d beta[n,d] * Y[n,d]
        expected_fused = (
            beta.unsqueeze(-1) * self.y
        ).sum(dim=2)

        self.assertTrue(
            torch.allclose(
                details["fused_directional_features"],
                expected_fused,
                atol=1e-6,
            )
        )

    def test_local_matching_can_assign_different_weights_per_token(self):
        # 手动设置 Query 和 Local Key，构造完全可解释的路由。
        model = GlobalLocalCMRADF(
            hidden_dim=8,
            bottleneck_dim=4,
            query_dim=64,
            eta=1.0,
            eta_mode="fixed",
            zero_init_up=False,
        )

        with torch.no_grad():
            model.query_proj.weight.zero_()
            model.query_proj.bias.zero_()
            model.query_proj.bias[0] = 8.0

            model.global_key_proj.weight.zero_()
            model.direction_embeddings.zero_()

            model.local_key_proj.weight.zero_()
            model.local_key_proj.weight[0, 0] = 1.0

        y = torch.zeros(1, 2, 4, 4)

        # 两个 token 的四方向响应顺序刚好相反。
        y[0, 0, :, 0] = torch.tensor(
            [0.0, 1.0, 2.0, 3.0]
        )
        y[0, 1, :, 0] = torch.tensor(
            [3.0, 2.0, 1.0, 0.0]
        )

        _, details = model(
            directional_features=y,
            biomed_img_feat=self.biomed_img,
            biomed_txt_feat=self.biomed_txt,
            return_details=True,
        )

        expected_first = torch.softmax(
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            dim=0,
        )
        expected_second = expected_first.flip(0)

        self.assertTrue(
            torch.allclose(
                details["routing_weights"][0, 0],
                expected_first,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                details["routing_weights"][0, 1],
                expected_second,
                atol=1e-6,
            )
        )

    def test_global_only_eta_zero_produces_shared_direction_weights(self):
        model = GlobalLocalCMRADF(
            hidden_dim=32,
            bottleneck_dim=8,
            eta=0.0,
            eta_mode="fixed",
            zero_init_up=False,
        )

        _, details = self._run(model=model)
        weights = details["routing_weights"]

        self.assertTrue(model.global_only)
        self.assertTrue(
            torch.equal(
                details["local_scores"],
                torch.zeros_like(
                    details["local_scores"]
                ),
            )
        )

        for index in range(1, self.y.size(1)):
            self.assertTrue(
                torch.allclose(
                    weights[:, index, :],
                    weights[:, 0, :],
                    atol=1e-7,
                )
            )

        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.local_key_proj.parameters()
            )
        )

    def test_question_changes_query_and_direction_weights(self):
        # 非零 Shared Up 不影响 beta，但有助于后续端到端测试。
        model = GlobalLocalCMRADF(
            hidden_dim=32,
            bottleneck_dim=8,
            eta=1.0,
            zero_init_up=False,
        )

        different_question = torch.randn_like(
            self.biomed_txt
        ) * 5.0

        _, details_a = self._run(
            model=model,
        )

        _, details_b = self._run(
            model=model,
            txt=different_question,
        )

        self.assertGreater(
            (
                details_a["query"]
                - details_b["query"]
            ).abs().max().item(),
            1e-6,
        )
        self.assertGreater(
            (
                details_a["routing_weights"]
                - details_b["routing_weights"]
            ).abs().max().item(),
            1e-7,
        )

    def test_zero_initialized_up_starts_identity_ready_and_can_train(self):
        # 此步暂时只返回残差。
        # 零初始化 Up 使 R=0，将来 X+R 才能从恒等映射开始。
        model = self.model

        self.assertTrue(
            torch.count_nonzero(model.up.weight).item() == 0
        )
        self.assertTrue(
            torch.count_nonzero(model.up.bias).item() == 0
        )

        target = torch.randn(
            1, 6, 32,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.3,
        )

        # 第 1 次更新：仅 Up 可直接从 loss 获得有效梯度。
        optimizer.zero_grad(set_to_none=True)

        residual = model(
            self.y,
            self.biomed_img,
            self.biomed_txt,
        )
        first_loss = (
            residual - target
        ).square().mean()

        first_loss.backward()

        self.assertGreater(
            model.up.weight.grad.abs().sum().item(),
            0.0,
        )

        optimizer.step()

        self.assertGreater(
            model.up.weight.abs().sum().item(),
            0.0,
        )

        # 第 2 次更新：Up 不再为零，梯度可传入 Query / Keys。
        optimizer.zero_grad(set_to_none=True)

        second_residual = model(
            self.y,
            self.biomed_img,
            self.biomed_txt,
        )

        second_loss = (
            second_residual - target
        ).square().mean()

        second_loss.backward()

        for name, parameter in (
            (
                "router_backbone",
                model.router_backbone[0].weight,
            ),
            ("query_proj", model.query_proj.weight),
            ("global_key_proj", model.global_key_proj.weight),
            ("local_key_proj", model.local_key_proj.weight),
            ("direction_embeddings", model.direction_embeddings),
        ):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(
                    parameter.grad.abs().sum().item(),
                    0.0,
                )

    def test_learnable_eta_remains_nonnegative_and_gets_gradient(self):
        model = GlobalLocalCMRADF(
            hidden_dim=32,
            bottleneck_dim=8,
            eta=0.7,
            eta_mode="learnable",
            zero_init_up=False,
        )

        residual, details = self._run(model=model)

        self.assertAlmostEqual(
            details["eta"].item(),
            0.7,
            places=5,
        )

        self.assertGreater(
            details["eta"].item(),
            0.0,
        )

        loss = (
            residual * torch.randn_like(residual)
        ).sum()
        loss.backward()

        self.assertIsNotNone(model.raw_eta.grad)
        self.assertGreater(
            model.raw_eta.grad.abs().sum().item(),
            0.0,
        )


    def test_bfloat16_module_with_fp32_biomedclip_features(self):
        # 检查混合精度：Qwen/Adapter 可能以 BF16 运行，
        # 冻结 BiomedCLIP 的输出则可能保持 FP32。
        #
        # 匹配分数与 Softmax 应为 FP32，
        # 最终输出应回到 BF16。
        model = GlobalLocalCMRADF(
            hidden_dim=32,
            bottleneck_dim=8,
            zero_init_up=False,
        ).bfloat16()

        residual, details = model(
            directional_features=self.y.bfloat16(),
            biomed_img_feat=self.biomed_img.float(),
            biomed_txt_feat=self.biomed_txt.float(),
            return_details=True,
        )

        self.assertEqual(
            residual.dtype,
            torch.bfloat16,
        )
        self.assertEqual(
            details["combined_scores"].dtype,
            torch.float32,
        )
        self.assertTrue(
            torch.isfinite(residual).all().item()
        )
        self.assertTrue(
            torch.allclose(
                details["routing_weights"].sum(dim=-1),
                torch.ones(1, 6),
                atol=1e-6,
            )
        )

    def test_state_dict_save_and_load_preserve_behavior(self):
        model = GlobalLocalCMRADF(
            hidden_dim=32,
            bottleneck_dim=8,
            eta=0.5,
            eta_mode="learnable",
            zero_init_up=False,
        )

        out_before = self._run(model=model)[0]

        buffer = io.BytesIO()
        torch.save(
            model.state_dict(),
            buffer,
        )
        buffer.seek(0)

        copied_model = GlobalLocalCMRADF(
            hidden_dim=32,
            bottleneck_dim=8,
            eta=0.5,
            eta_mode="learnable",
            zero_init_up=False,
        )

        copied_model.load_state_dict(
            torch.load(
                buffer,
                weights_only=True,
            ),
            strict=True,
        )

        out_after = self._run(model=copied_model)[0]

        self.assertTrue(
            torch.equal(
                out_before,
                out_after,
            )
        )

    def test_invalid_inputs_raise_clear_errors(self):
        for bad_y in (
            torch.zeros(1, 6, 3, 8),
            torch.zeros(1, 6, 4, 9),
            torch.zeros(2, 6, 4, 8),
            torch.zeros(1, 0, 4, 8),
            torch.zeros(6, 4, 8),
        ):
            with self.subTest(shape=tuple(bad_y.shape)):
                with self.assertRaises(ValueError):
                    self.model(
                        bad_y,
                        self.biomed_img,
                        self.biomed_txt,
                    )

        with self.assertRaises(ValueError):
            self.model(
                self.y,
                torch.zeros(2, 512),
                self.biomed_txt,
            )

        with self.assertRaises(ValueError):
            self.model(
                self.y,
                self.biomed_img,
                torch.zeros(1, 511),
            )

        with self.assertRaises(ValueError):
            GlobalLocalCMRADF(
                hidden_dim=32,
                bottleneck_dim=8,
                eta=-1.0,
            )

        with self.assertRaises(ValueError):
            GlobalLocalCMRADF(
                hidden_dim=32,
                bottleneck_dim=8,
                eta=0.0,
                eta_mode="learnable",
            )


class TestFourDirectionalCMRADFIntegration(unittest.TestCase):
    """
    两阶段拼接测试。

    沿用原 visual_token_split，把各视觉条目的 Qwen pooled
    tokens 与各自的 BiomedCLIP 图文条件一一对应。
    """

    def setUp(self):
        torch.manual_seed(2027)

        self.model = Qwen3VLFourDirectionalCMRADF(
            hidden_dim=32,
            r=4,
            ssm_d_state=4,
            ssm_d_conv=3,
            ssm_expand=2,
            ssm_backend="torch_reference",
            eta=1.0,
            zero_init_up=False,
        )

        # 第一张图原 grid: [1,4,6]
        # Spatial merge=2 后：t=1,H=2,W=3 => N=6
        #
        # 第二张图原 grid: [1,2,4]
        # Spatial merge=2 后：t=1,H=1,W=2 => N=2
        self.grid_thw = torch.tensor(
            [
                [1, 4, 6],
                [1, 2, 4],
            ],
            dtype=torch.long,
        )

        self.x = torch.randn(1, 8, 32)
        self.img = torch.randn(2, 512)
        self.txt = torch.randn(2, 512)

    def test_qwen_3d_input_and_two_image_shapes(self):
        residual, details = self.model(
            x=self.x,
            biomed_img_feat=self.img,
            biomed_txt_feat=self.txt,
            grid_thw=self.grid_thw,
            return_details=True,
        )

        self.assertEqual(
            tuple(residual.shape),
            (1, 8, 32),
        )
        self.assertEqual(len(details), 2)
        self.assertEqual(
            tuple(details[0]["routing_weights"].shape),
            (1, 6, 4),
        )
        self.assertEqual(
            tuple(details[1]["routing_weights"].shape),
            (1, 2, 4),
        )
        self.assertEqual(
            details[0]["grid_shape"],
            (1, 2, 3),
        )
        self.assertEqual(
            details[1]["grid_shape"],
            (1, 1, 2),
        )

    def test_2d_input_retains_original_shape(self):
        flat_x = self.x.squeeze(0)

        residual = self.model(
            x=flat_x,
            biomed_img_feat=self.img,
            biomed_txt_feat=self.txt,
            grid_thw=self.grid_thw,
        )

        self.assertEqual(
            tuple(residual.shape),
            tuple(flat_x.shape),
        )

    def test_two_images_equal_individual_image_processing(self):
        self.model.eval()

        combined = self.model(
            x=self.x,
            biomed_img_feat=self.img,
            biomed_txt_feat=self.txt,
            grid_thw=self.grid_thw,
        )

        first = self.model(
            x=self.x[:, :6, :],
            biomed_img_feat=self.img[:1],
            biomed_txt_feat=self.txt[:1],
            grid_thw=self.grid_thw[:1],
        )

        second = self.model(
            x=self.x[:, 6:, :],
            biomed_img_feat=self.img[1:2],
            biomed_txt_feat=self.txt[1:2],
            grid_thw=self.grid_thw[1:2],
        )

        self.assertTrue(
            torch.allclose(
                combined,
                torch.cat((first, second), dim=1),
                atol=1e-6,
            )
        )

    def test_different_question_for_second_image_does_not_change_first(self):
        self.model.eval()

        base_result, base_details = self.model(
            self.x,
            self.img,
            self.txt,
            self.grid_thw,
            return_details=True,
        )

        changed_txt = self.txt.clone()
        changed_txt[1] = torch.randn(512) * 4

        changed_result, changed_details = self.model(
            self.x,
            self.img,
            changed_txt,
            self.grid_thw,
            return_details=True,
        )

        # 第一张图应完全不受第二张图文本变化影响。
        self.assertTrue(
            torch.equal(
                base_result[:, :6],
                changed_result[:, :6],
            )
        )
        self.assertTrue(
            torch.equal(
                base_details[0]["routing_weights"],
                changed_details[0]["routing_weights"],
            )
        )

        # 第二张图的 token 级方向权重应能变化。
        self.assertGreater(
            (
                base_details[1]["routing_weights"]
                - changed_details[1]["routing_weights"]
            ).abs().max().item(),
            1e-7,
        )

    def test_combined_gradients_reach_four_ssm_and_cm_radf(self):
        x = self.x.clone().requires_grad_()

        residual = self.model(
            x=x,
            biomed_img_feat=self.img,
            biomed_txt_feat=self.txt,
            grid_thw=self.grid_thw,
        )

        loss = (
            residual * torch.randn_like(residual)
        ).sum()

        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

        for name, branch in (
            ("RF", self.model.four_directional_ssm.ssm_rf),
            ("RR", self.model.four_directional_ssm.ssm_rr),
            ("CF", self.model.four_directional_ssm.ssm_cf),
            ("CR", self.model.four_directional_ssm.ssm_cr),
        ):
            with self.subTest(direction=name):
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and parameter.grad.abs().sum().item() > 0
                        for parameter in branch.parameters()
                    )
                )

        self.assertGreater(
            self.model.cm_radf.query_proj.weight.grad.abs().sum().item(),
            0,
        )

        self.assertGreater(
            self.model.four_directional_ssm.down.weight.grad.abs().sum().item(),
            0,
        )

    def test_initial_shared_up_produces_zero_residual_for_all_images(self):
        model = Qwen3VLFourDirectionalCMRADF(
            hidden_dim=32,
            r=4,
            ssm_d_state=4,
            ssm_backend="torch_reference",
            zero_init_up=True,
        )

        residual = model(
            self.x,
            self.img,
            self.txt,
            self.grid_thw,
        )

        self.assertTrue(
            torch.equal(
                residual,
                torch.zeros_like(residual),
            )
        )


    def test_video_temporal_grid_is_accepted(self):
        # 原始 Qwen grid: t=2,h=4,w=6
        # merge 后 H=2,W=3，全部 N=12。
        #
        # 4D-SSM 内部按帧拆成 [t,H*W,C]；
        # CM-RADF 对这个视觉条目的全部有效 token
        # 生成一组 [1,12,4] 的方向权重。
        video_tokens = torch.randn(
            1, 12, 32,
        )
        video_grid_thw = torch.tensor(
            [[2, 4, 6]],
            dtype=torch.long,
        )

        residual, details = self.model(
            x=video_tokens,
            biomed_img_feat=self.img[:1],
            biomed_txt_feat=self.txt[:1],
            grid_thw=video_grid_thw,
            return_details=True,
        )

        self.assertEqual(
            tuple(residual.shape),
            (1, 12, 32),
        )
        self.assertEqual(
            details[0]["grid_shape"],
            (2, 2, 3),
        )
        self.assertEqual(
            tuple(details[0]["routing_weights"].shape),
            (1, 12, 4),
        )

    def test_input_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.model(
                self.x,
                self.img[:1],
                self.txt,
                self.grid_thw,
            )

        with self.assertRaises(ValueError):
            self.model(
                self.x[:, :7],
                self.img,
                self.txt,
                self.grid_thw,
            )

        with self.assertRaises(ValueError):
            self.model(
                self.x,
                self.img,
                self.txt,
                torch.tensor([[1, 4, 6]]),
            )


if __name__ == "__main__":
    unittest.main()

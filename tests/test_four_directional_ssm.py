import importlib.util
import unittest

import torch
import torch.nn as nn

from utils.qwen3vl.qwen3_vl_8B_four_directional_ssm import (
    FourDirectionalSSM,
)
from utils.qwen3vl.qwen3_vl_8B_four_directional_ssm_utils import (
    DIRECTION_NAMES,
    build_directional_indices,
    directional_scan,
    inverse_directional_scan,
)


class TestFourDirectionalScan(unittest.TestCase):
    """
    单独验证方向扫描及逆扫描。

    核心要求：
        1. 四条扫描路径与定义严格一致；
        2. 逆扫描恢复原始 token 的位置与数值；
        3. 视频帧之间不串联；
        4. 置换操作保持可微。
    """

    def test_exact_directional_indices(self):
        # 原始 H=2、W=3 网格：
        #   0 1 2
        #   3 4 5
        expected_indices = {
            "RF": [0, 1, 2, 3, 4, 5],
            "RR": [5, 4, 3, 2, 1, 0],
            "CF": [0, 3, 1, 4, 2, 5],
            "CR": [5, 2, 4, 1, 3, 0],
        }

        for direction, expected in expected_indices.items():
            with self.subTest(direction=direction):
                actual = build_directional_indices(
                    grid_h=2,
                    grid_w=3,
                    direction=direction,
                    device=torch.device("cpu"),
                )

                self.assertEqual(
                    actual.tolist(),
                    expected,
                )

    def test_scan_then_inverse_exactly_recovers_input(self):
        # t=2, H=2, W=3, C=4。
        # 数据每个元素都不同，以检查空间顺序。
        x = torch.arange(
            2 * 2 * 3 * 4,
            dtype=torch.float32,
        ).reshape(1, 12, 4)

        for direction in DIRECTION_NAMES:
            with self.subTest(direction=direction):
                scanned = directional_scan(
                    x=x,
                    grid_shape=(2, 2, 3),
                    direction=direction,
                )

                self.assertEqual(
                    tuple(scanned.shape),
                    (2, 6, 4),
                )

                restored = inverse_directional_scan(
                    scanned_x=scanned,
                    grid_shape=(2, 2, 3),
                    direction=direction,
                )

                self.assertTrue(
                    torch.equal(restored, x)
                )

    def test_all_frames_use_same_spatial_scan(self):
        # 第 1 帧和第 2 帧使用不同的数值范围。
        x = torch.tensor(
            [
                0, 1, 2, 3, 4, 5,
                100, 101, 102, 103, 104, 105,
            ],
            dtype=torch.float32,
        ).reshape(1, 12, 1)

        expected_first = [0, 3, 1, 4, 2, 5]
        expected_second = [100, 103, 101, 104, 102, 105]

        scanned = directional_scan(
            x=x,
            grid_shape=(2, 2, 3),
            direction="CF",
        )

        self.assertEqual(
            scanned[0, :, 0].tolist(),
            expected_first,
        )

        self.assertEqual(
            scanned[1, :, 0].tolist(),
            expected_second,
        )

    def test_scan_and_inverse_preserve_gradient(self):
        x = torch.randn(
            1,
            24,
            5,
            requires_grad=True,
        )

        # 非正方形 H=3, W=4，防止误把 H/W 互换。
        for direction in DIRECTION_NAMES:
            scanned = directional_scan(
                x=x,
                grid_shape=(2, 3, 4),
                direction=direction,
            )

            restored = inverse_directional_scan(
                scanned_x=scanned,
                grid_shape=(2, 3, 4),
                direction=direction,
            )

            # 使用明确可求导的标量。
            loss = restored.square().sum()
            gradient, = torch.autograd.grad(
                loss,
                x,
                retain_graph=True,
            )

            self.assertTrue(
                torch.allclose(gradient, 2.0 * x)
            )

    def test_invalid_scan_arguments_raise(self):
        x = torch.zeros(1, 6, 8)

        with self.assertRaises(ValueError):
            directional_scan(
                x=x,
                grid_shape=(1, 2, 4),
                direction="RF",
            )

        with self.assertRaises(ValueError):
            directional_scan(
                x=x,
                grid_shape=(1, 2, 3),
                direction="unknown",
            )

        with self.assertRaises(ValueError):
            directional_scan(
                x=x,
                grid_shape=(0, 2, 3),
                direction="RF",
            )

        with self.assertRaises(ValueError):
            directional_scan(
                x=x,
                grid_shape=(1, 2.5, 3),
                direction="RF",
            )

        with self.assertRaises(ValueError):
            directional_scan(
                x=torch.zeros(2, 6, 8),
                grid_shape=(1, 2, 3),
                direction="RF",
            )

        with self.assertRaises(ValueError):
            inverse_directional_scan(
                scanned_x=torch.zeros(2, 5, 8),
                grid_shape=(2, 2, 3),
                direction="RF",
            )


class TestFourDirectionalSSM(unittest.TestCase):
    """
    完整验证独立 4D-SSM 模块。

    当前默认使用 torch_reference：
        CPU 即可运行所有基础逻辑测试。

    正式训练的 mamba 后端在 GPU 环境另有 smoke test。
    """

    def setUp(self):
        torch.manual_seed(2026)

        self.model = FourDirectionalSSM(
            hidden_dim=32,
            r=4,
            ssm_d_state=4,
            ssm_d_conv=3,
            ssm_expand=2,
            ssm_backend="torch_reference",
        )

    def test_forward_shape_single_image(self):
        # [1, N=6, D=32] -> [1, N=6, 4, C=8]
        x = torch.randn(1, 6, 32)

        y = self.model(
            x=x,
            grid_shape=(1, 2, 3),
        )

        self.assertEqual(
            tuple(y.shape),
            (1, 6, 4, 8),
        )

        self.assertTrue(
            torch.isfinite(y).all().item()
        )

    def test_forward_shape_multiple_frames(self):
        # 两帧，每帧 2x3。
        x = torch.randn(1, 12, 32)

        y = self.model(
            x=x,
            grid_shape=(2, 2, 3),
        )

        self.assertEqual(
            tuple(y.shape),
            (1, 12, 4, 8),
        )

    def test_forward_shape_single_token(self):
        x = torch.randn(1, 1, 32)

        y = self.model(
            x=x,
            grid_shape=(1, 1, 1),
        )

        self.assertEqual(
            tuple(y.shape),
            (1, 1, 4, 8),
        )

    def test_four_ssm_modules_have_independent_parameters(self):
        branches = (
            self.model.ssm_rf,
            self.model.ssm_rr,
            self.model.ssm_cf,
            self.model.ssm_cr,
        )

        # 四个 Module 必须是四个不同对象。
        self.assertEqual(
            len({id(module) for module in branches}),
            4,
        )

        # 参数不得跨方向共享。
        parameter_id_sets = [
            {id(p) for p in module.parameters()}
            for module in branches
        ]

        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(
                    parameter_id_sets[i]
                    & parameter_id_sets[j]
                )

        # LN_in / Down / LN_dir 都只有一份。
        self.assertEqual(
            self.model.down.in_features,
            32,
        )
        self.assertEqual(
            self.model.down.out_features,
            8,
        )
        self.assertIsInstance(
            self.model.ln_dir,
            nn.LayerNorm,
        )

    def test_identity_ssm_proves_all_outputs_spatially_aligned(self):
        # 只在本测试中把所有 SSM 替换成 Identity。
        #
        # 因为 Scan 后又执行了 InverseScan，
        # 四个方向输出必须与输入完全相同。
        identity_model = FourDirectionalSSM(
            hidden_dim=4,
            r=1,
            ssm_d_state=2,
            ssm_backend="torch_reference",
        )

        identity_model.ln_in = nn.Identity()
        identity_model.down = nn.Identity()
        identity_model.ln_dir = nn.Identity()

        identity_model.ssm_rf = nn.Identity()
        identity_model.ssm_rr = nn.Identity()
        identity_model.ssm_cf = nn.Identity()
        identity_model.ssm_cr = nn.Identity()

        # 两帧不同内容，且为非正方形网格。
        x = torch.arange(
            1 * 2 * 2 * 3 * 4,
            dtype=torch.float32,
        ).reshape(1, 12, 4)

        y = identity_model(
            x=x,
            grid_shape=(2, 2, 3),
        )

        for index, direction in enumerate(DIRECTION_NAMES):
            with self.subTest(direction=direction):
                self.assertTrue(
                    torch.equal(y[:, :, index, :], x)
                )


    def test_stateful_scan_outputs_are_correctly_realigned(self):
        # 使用一个确定性的“累计传播”模块，
        # 验证每个方向不仅可以逆排列，而且传播结果与方向相符。
        #
        # 原始 2x3 网格：
        #   1 2 3
        #   4 5 6
        #
        # CumulativeSSM 仅用于这一单元测试，
        # 不作为正式的 Selective SSM 实现。
        class CumulativeSSM(nn.Module):
            def forward(self, x):
                return torch.cumsum(
                    x,
                    dim=1,
                )

        model = FourDirectionalSSM(
            hidden_dim=1,
            r=1,
            ssm_d_state=2,
            ssm_backend="torch_reference",
        )

        model.ln_in = nn.Identity()
        model.down = nn.Identity()
        model.ln_dir = nn.Identity()
        model.ssm_rf = CumulativeSSM()
        model.ssm_rr = CumulativeSSM()
        model.ssm_cf = CumulativeSSM()
        model.ssm_cr = CumulativeSSM()

        x = torch.tensor(
            [1, 2, 3, 4, 5, 6],
            dtype=torch.float32,
        ).reshape(1, 6, 1)

        result = model(
            x=x,
            grid_shape=(1, 2, 3),
        )

        # 所有期望结果均已恢复成原始 row-major 位置。
        expected = {
            "RF": [1, 3, 6, 10, 15, 21],
            "RR": [21, 20, 18, 15, 11, 6],
            "CF": [1, 7, 15, 5, 12, 21],
            "CR": [21, 16, 9, 20, 14, 6],
        }

        for direction_index, direction in enumerate(DIRECTION_NAMES):
            with self.subTest(direction=direction):
                aligned_values = result[
                    0,
                    :,
                    direction_index,
                    0,
                ].tolist()

                self.assertEqual(
                    aligned_values,
                    expected[direction],
                )

    def test_two_frames_do_not_share_ssm_state(self):
        self.model.eval()

        frame1 = torch.randn(1, 6, 32)
        frame2 = torch.randn(1, 6, 32)

        # 同一个视觉条目，包含两帧。
        full_input = torch.cat(
            (frame1, frame2),
            dim=1,
        )

        combined_output = self.model(
            x=full_input,
            grid_shape=(2, 2, 3),
        )

        # 两帧单独运行应得到同样的结果。
        out1 = self.model(
            x=frame1,
            grid_shape=(1, 2, 3),
        )
        out2 = self.model(
            x=frame2,
            grid_shape=(1, 2, 3),
        )

        separate_output = torch.cat(
            (out1, out2),
            dim=1,
        )

        self.assertTrue(
            torch.allclose(
                combined_output,
                separate_output,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_gradients_reach_all_four_branches(self):
        self.model.train()

        x = torch.randn(
            1,
            6,
            32,
            requires_grad=True,
        )

        y = self.model(
            x=x,
            grid_shape=(1, 2, 3),
        )

        # 随机权重可避免 LN 输出平方和近似常数的问题。
        loss = (
            y * torch.randn_like(y)
        ).sum()

        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(
            x.grad.abs().sum().item(),
            0.0,
        )

        checked_parameters = (
            self.model.down.weight,
            self.model.ln_in.weight,
            self.model.ln_dir.weight,
            self.model.ssm_rf.in_proj.weight,
            self.model.ssm_rr.in_proj.weight,
            self.model.ssm_cf.in_proj.weight,
            self.model.ssm_cr.in_proj.weight,
        )

        for parameter in checked_parameters:
            self.assertIsNotNone(
                parameter.grad
            )
            self.assertGreater(
                parameter.grad.abs().sum().item(),
                0.0,
            )

    def test_qwen_dimension_4096_to_256(self):
        # 验证与你旧 MoRA-Med 一致的 D=4096、r=16。
        model = FourDirectionalSSM(
            hidden_dim=4096,
            r=16,
            ssm_d_state=4,
            ssm_backend="torch_reference",
        )

        x = torch.randn(1, 4, 4096)

        with torch.no_grad():
            output = model(
                x=x,
                grid_shape=(1, 2, 2),
            )

        self.assertEqual(
            tuple(output.shape),
            (1, 4, 4, 256),
        )

    def test_invalid_model_inputs_raise(self):
        with self.assertRaises(ValueError):
            self.model(
                x=torch.randn(2, 6, 32),
                grid_shape=(1, 2, 3),
            )

        with self.assertRaises(ValueError):
            self.model(
                x=torch.randn(1, 6, 31),
                grid_shape=(1, 2, 3),
            )

        with self.assertRaises(ValueError):
            self.model(
                x=torch.randn(1, 6, 32),
                grid_shape=(1, 3, 3),
            )

        with self.assertRaises(ValueError):
            FourDirectionalSSM(
                hidden_dim=30,
                r=16,
                ssm_backend="torch_reference",
            )

    def test_model_output_is_not_residual(self):
        # 4D-SSM 输出还是 bottleneck feature，不是 D 维残差。
        x = torch.randn(1, 6, 32)
        y = self.model(x, (1, 2, 3))

        self.assertEqual(y.size(-1), 8)
        self.assertNotEqual(y.size(-1), x.size(-1))

    @unittest.skipUnless(
        torch.cuda.is_available()
        and importlib.util.find_spec("mamba_ssm") is not None,
        "当前环境没有 CUDA + mamba_ssm，跳过正式 Mamba GPU 测试",
    )
    def test_mamba_backend_gpu_forward_and_backward(self):
        # 正式训练环境下执行。
        model = FourDirectionalSSM(
            hidden_dim=32,
            r=4,
            ssm_d_state=4,
            ssm_d_conv=3,
            ssm_expand=2,
            ssm_backend="mamba",
        ).cuda()

        x = torch.randn(
            1,
            6,
            32,
            device="cuda",
            requires_grad=True,
        )

        y = model(
            x=x,
            grid_shape=(1, 2, 3),
        )

        self.assertEqual(
            tuple(y.shape),
            (1, 6, 4, 8),
        )

        y.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(
            torch.isfinite(x.grad).all().item()
        )


if __name__ == "__main__":
    unittest.main()

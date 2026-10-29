from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.qwen3vl.qwen3_vl_8B_four_directional_ssm_utils import (
    DIRECTION_NAMES,
    directional_scan,
    inverse_directional_scan,
    validate_grid_shape,
)


class TorchSelectiveSSMReference(nn.Module):
    """
    纯 PyTorch Selective SSM 参考实现。

    主要用途：
        1. 无 CUDA/Mamba 环境下的前向形状检查；
        2. 反向传播及梯度检查；
        3. 逐帧独立状态与四方向扫描的单元测试。

    这里实现输入依赖的离散状态空间递推：

        A = -exp(A_log)

        delta_t = softplus(W_delta(u_t))
        B_t = W_B(u_t)
        C_t = W_C(u_t)

        h_t = exp(delta_t * A) * h_{t-1}
              + delta_t * B_t * u_t

        y_t = C_t * h_t + D * u_t

    其中 delta_t、B_t、C_t 都随输入变化，
    因此它是真正的 Selective SSM，而非静态线性层。

    注意：
        这是供调试的 S6 风格参考实现，不等价于完整
        mamba_ssm.Mamba（后者还包含优化的卷积、投影与扫描）。

        两种后端的参数名称与权重不兼容，不能互相加载。

        参考后端使用 Python for 循环递推，长序列速度较慢。
        正式训练建议使用 backend="mamba"。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
    ):
        super().__init__()

        if d_model <= 0 or d_state <= 0:
            raise ValueError(
                "d_model 和 d_state 必须大于 0，"
                f"当前为 d_model={d_model}, d_state={d_state}"
            )

        self.d_model = int(d_model)
        self.d_state = int(d_state)

        # 输入投影为内容分支与门控分支。
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_model,
        )

        # Selective SSM 的三个输入依赖参数。
        self.dt_proj = nn.Linear(
            self.d_model,
            self.d_model,
        )

        self.b_proj = nn.Linear(
            self.d_model,
            self.d_state,
        )

        self.c_proj = nn.Linear(
            self.d_model,
            self.d_state,
        )

        # 输出投影将递推结果映射回 d_model。
        self.out_proj = nn.Linear(
            self.d_model,
            self.d_model,
        )

        # A 必须为负值，以获得稳定的衰减形式。
        # [d_model, d_state]
        a_initial = (
            torch.arange(
                1,
                self.d_state + 1,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(self.d_model, 1)
        )

        self.A_log = nn.Parameter(
            torch.log(a_initial)
        )

        # 跳跃连接系数 D: [d_model]
        self.D = nn.Parameter(
            torch.ones(self.d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                已按某个空间方向排列好的 token 序列。
                shape: [batch_frames, sequence_length, d_model]

        Returns:
            与输入 shape 相同的方向 SSM 响应。

        每个 batch 元素的 h 初始为零。
        因此不同帧之间不存在跨序列的状态传递。
        """
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                "TorchSelectiveSSMReference 输入必须为 "
                "[batch_frames, L, d_model]，"
                f"当前为 {tuple(x.shape)}"
            )

        batch_frames, sequence_length, _ = x.shape

        u, gate = torch.chunk(
            self.in_proj(x),
            chunks=2,
            dim=-1,
        )

        u = F.silu(u)

        # 三个参数均随每个输入 token 动态变化。
        delta = F.softplus(
            self.dt_proj(u)
        )
        b_values = self.b_proj(u)
        c_values = self.c_proj(u)

        # [d_model, d_state]
        a_values = -torch.exp(
            self.A_log
        ).to(dtype=u.dtype)

        d_values = self.D.to(
            dtype=u.dtype
        )

        # [batch_frames, d_model, d_state]
        # 不同帧/独立输入之间状态完全隔离。
        state = torch.zeros(
            batch_frames,
            self.d_model,
            self.d_state,
            dtype=u.dtype,
            device=u.device,
        )

        outputs = []

        for position in range(sequence_length):
            # [batch_frames, d_model, 1]
            delta_t = delta[:, position, :].unsqueeze(-1)

            # [batch_frames, 1, d_state]
            b_t = b_values[:, position, :].unsqueeze(1)
            c_t = c_values[:, position, :].unsqueeze(1)

            # [batch_frames, d_model, 1]
            u_t = u[:, position, :].unsqueeze(-1)

            # 输入条件化的状态转移。
            # [batch_frames, d_model, d_state]
            a_bar = torch.exp(
                delta_t * a_values.unsqueeze(0)
            )

            # 离散化的输入驱动项：
            # delta_t * B_t * u_t
            state = (
                a_bar * state
                + delta_t * b_t * u_t
            )

            # 输出读出 + skip connection。
            # [batch_frames, d_model]
            y_t = (
                (state * c_t).sum(dim=-1)
                + d_values * u[:, position, :]
            )

            # 输入条件化的输出门控。
            y_t = y_t * torch.sigmoid(
                gate[:, position, :]
            )

            outputs.append(y_t)

        # [batch_frames, sequence_length, d_model]
        y = torch.stack(
            outputs,
            dim=1,
        )

        return self.out_proj(y)


class FourDirectionalSSM(nn.Module):
    """
    CM-DA-SSM 第一阶段：Four-Directional Selective SSM。

    仅负责：
        1. LN_in；
        2. Shared Down Projection；
        3. RF / RR / CF / CR 四方向扫描；
        4. 四个参数独立的 Selective SSM；
        5. 逆扫描恢复原始 token 位置；
        6. 共享 LN_dir。

    本模块暂时不包括：
        - BiomedCLIP；
        - Global–Local CM-RADF；
        - Shared Up Projection；
        - Capped RMS；
        - lambda 残差注入；
        - DeepStack。

    输入：
        x: [1, N, hidden_dim]

        grid_shape=(t, H, W)
        N=t*H*W

        注意 H、W 必须是 Qwen spatial merge 之后的网格。

    输出：
        directional_features: [1, N, 4, bottleneck_dim]

        方向维度 index 固定为：
            0 -> RF
            1 -> RR
            2 -> CF
            3 -> CR

        四组方向响应已经通过 inverse scan
        对齐回原始 pooled visual token 顺序。

    backend:
        "mamba":
            使用官方 mamba_ssm.Mamba，供正式训练使用。

        "torch_reference":
            使用纯 PyTorch Selective SSM，供 CPU/逻辑测试使用。
            与 Mamba 实现不同，权重不可互换。
    """

    def __init__(
        self,
        hidden_dim: int,
        r: int = 16,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        ssm_backend: str = "mamba",
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim 必须为正整数，"
                f"当前为 {hidden_dim}"
            )

        if r <= 0:
            raise ValueError(
                f"r 必须为正整数，当前为 {r}"
            )

        if hidden_dim % r != 0:
            raise ValueError(
                "hidden_dim 必须能够被 r 整除："
                f"hidden_dim={hidden_dim}, r={r}"
            )

        if (
            ssm_d_state <= 0
            or ssm_d_conv <= 0
            or ssm_expand <= 0
        ):
            raise ValueError(
                "SSM 超参数必须为正整数："
                f"d_state={ssm_d_state}, "
                f"d_conv={ssm_d_conv}, "
                f"expand={ssm_expand}"
            )

        if ssm_backend not in {
            "mamba",
            "torch_reference",
        }:
            raise ValueError(
                "ssm_backend 只支持 'mamba' 或 'torch_reference'，"
                f"当前为 {ssm_backend}"
            )

        self.hidden_dim = int(hidden_dim)
        self.r = int(r)
        self.bottleneck_dim = self.hidden_dim // self.r

        self.ssm_d_state = int(ssm_d_state)
        self.ssm_d_conv = int(ssm_d_conv)
        self.ssm_expand = int(ssm_expand)
        self.ssm_backend = ssm_backend

        # 固定方向输出顺序，供后续 CM-RADF 使用。
        self.direction_names = DIRECTION_NAMES

        # Shared LN_in + Shared Down。
        #
        # [1, N, D]
        # -> [1, N, D]
        # -> [1, N, C]
        self.ln_in = nn.LayerNorm(
            self.hidden_dim
        )

        self.down = nn.Linear(
            self.hidden_dim,
            self.bottleneck_dim,
            bias=False,
        )

        # 四个方向的 Selective SSM 参数完全独立。
        #
        # 不能把同一实例复用于四个方向：
        #   self.ssm_rf is not self.ssm_rr, etc.
        self.ssm_rf = self._build_ssm()
        self.ssm_rr = self._build_ssm()
        self.ssm_cf = self._build_ssm()
        self.ssm_cr = self._build_ssm()

        # 四方向共享同一个 LN_dir，不创建四套 LN。
        self.ln_dir = nn.LayerNorm(
            self.bottleneck_dim
        )

    def _build_ssm(self) -> nn.Module:
        """
        构造一个独立的 Selective SSM 分支。

        每调用一次都必须创建新的 nn.Module 实例。
        """
        if self.ssm_backend == "torch_reference":
            return TorchSelectiveSSMReference(
                d_model=self.bottleneck_dim,
                d_state=self.ssm_d_state,
            )

        # 使用延迟导入，便于在普通 CPU 环境运行 shape/scan 测试。
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as exc:
            raise ImportError(
                "当前 ssm_backend='mamba'，"
                "但无法导入 mamba_ssm.Mamba。"
                "请在正式训练环境中安装兼容当前 "
                "PyTorch/CUDA 的 mamba-ssm，"
                "或者先用 ssm_backend='torch_reference' "
                "运行独立测试。"
            ) from exc

        # 官方 Mamba-1 block。
        # 输入和输出均为 [batch_frames, H*W, C]。
        return Mamba(
            d_model=self.bottleneck_dim,
            d_state=self.ssm_d_state,
            d_conv=self.ssm_d_conv,
            expand=self.ssm_expand,
        )

    def _forward_one_direction(
        self,
        z: torch.Tensor,
        grid_shape: Tuple[int, int, int],
        direction: str,
        ssm_module: nn.Module,
    ) -> torch.Tensor:
        """
        执行单个方向的完整处理流程：

            [1, N, C]
                ↓ Scan
            [t, H*W, C]
                ↓ Selective SSM
            [t, H*W, C]
                ↓ Inverse Scan
            [1, N, C]
                ↓ Shared LN_dir
            [1, N, C]

        返回结果的 token 位置与输入 z 完全对齐。
        """
        scanned_z = directional_scan(
            x=z,
            grid_shape=grid_shape,
            direction=direction,
        )

        # 不同时间帧作为独立 batch 进行 SSM，
        # 不会串联 hidden state。
        scanned_y = ssm_module(
            scanned_z
        )

        if scanned_y.shape != scanned_z.shape:
            raise RuntimeError(
                f"{direction} 方向的 SSM 改变了特征形状："
                f"input={tuple(scanned_z.shape)}, "
                f"output={tuple(scanned_y.shape)}"
            )

        restored_y = inverse_directional_scan(
            scanned_x=scanned_y,
            grid_shape=grid_shape,
            direction=direction,
        )

        # 四方向调用同一个共享 LN_dir。
        return self.ln_dir(
            restored_y
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            x:
                单张视觉条目的 pooled visual tokens。
                shape: [1, N, hidden_dim]

            grid_shape:
                空间合并后的视觉网格 (t, H, W)。
                必须满足 N == t*H*W。

        Returns:
            directional_features:
                [1, N, 4, bottleneck_dim]

                directional_features[:, :, 0, :] -> RF
                directional_features[:, :, 1, :] -> RR
                directional_features[:, :, 2, :] -> CF
                directional_features[:, :, 3, :] -> CR

        本函数不做：
            - 方向加权融合；
            - 上投影；
            - pooled tokens 残差更新。

        后续 CM-RADF 将直接接收这里返回的四方向特征。
        """
        t, grid_h, grid_w = validate_grid_shape(
            grid_shape
        )

        if x.ndim != 3:
            raise ValueError(
                "FourDirectionalSSM 要求 x 为三维张量 "
                "[1, N, hidden_dim]，"
                f"当前 shape={tuple(x.shape)}"
            )

        batch_size, token_count, input_dim = x.shape

        # 与原 MoRA-Med 的 BasicVisualAdapter 接口保持一致：
        # 每次 forward 只接收一张视觉条目的 pooled tokens。
        if batch_size != 1:
            raise ValueError(
                "FourDirectionalSSM 要求单张图片/视频条目输入，"
                f"即 batch_size=1，当前为 {batch_size}"
            )

        if input_dim != self.hidden_dim:
            raise ValueError(
                "输入 hidden_dim 与 4D-SSM 配置不一致："
                f"input={input_dim}, "
                f"expected={self.hidden_dim}"
            )

        expected_tokens = t * grid_h * grid_w

        if token_count != expected_tokens:
            raise ValueError(
                "视觉 token 数量与网格不匹配："
                f"tokens={token_count}, "
                f"grid_shape={(t, grid_h, grid_w)}, "
                f"expected_tokens={expected_tokens}"
            )

        # LN_in + Shared Down。
        #
        # [1, N, hidden_dim] -> [1, N, bottleneck_dim]
        z = self.down(
            self.ln_in(x)
        )

        # 按固定顺序执行四个参数独立的 SSM。
        # 四个分支共用 z 和 ln_dir，但不共享 SSM 参数。
        y_rf = self._forward_one_direction(
            z=z,
            grid_shape=grid_shape,
            direction="RF",
            ssm_module=self.ssm_rf,
        )

        y_rr = self._forward_one_direction(
            z=z,
            grid_shape=grid_shape,
            direction="RR",
            ssm_module=self.ssm_rr,
        )

        y_cf = self._forward_one_direction(
            z=z,
            grid_shape=grid_shape,
            direction="CF",
            ssm_module=self.ssm_cf,
        )

        y_cr = self._forward_one_direction(
            z=z,
            grid_shape=grid_shape,
            direction="CR",
            ssm_module=self.ssm_cr,
        )

        # [1, N, C] × 4
        # -> [1, N, 4, C]
        #
        # dim=2 为方向维，不能与 N 或 C 混淆。
        directional_features = torch.stack(
            (y_rf, y_rr, y_cf, y_cr),
            dim=2,
        )

        expected_shape = (
            1,
            token_count,
            4,
            self.bottleneck_dim,
        )

        if tuple(directional_features.shape) != expected_shape:
            raise RuntimeError(
                "四方向 SSM 输出形状错误："
                f"expected={expected_shape}, "
                f"actual={tuple(directional_features.shape)}"
            )

        return directional_features

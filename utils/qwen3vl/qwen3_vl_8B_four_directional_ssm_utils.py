from typing import Tuple

import torch


# 四方向扫描顺序固定，后续 CM-RADF 将按这个顺序建立方向权重。
#
# index 0: RF - Row Forward
# index 1: RR - Row Reverse
# index 2: CF - Column Forward
# index 3: CR - Column Reverse
DIRECTION_NAMES = ("RF", "RR", "CF", "CR")


def validate_grid_shape(
    grid_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    检查 Qwen 空间合并后的视觉网格形状。

    Args:
        grid_shape:
            (t, H, W)，注意 H、W 必须已经经过 spatial merge。

            例如原始 grid_thw=(1, 8, 12)，
            且 spatial_merge_size=2 时，此处传入：
                grid_shape=(1, 4, 6)

    Returns:
        (t, H, W)，三个元素均为正整数。
    """
    if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 3:
        raise ValueError(
            "grid_shape 必须为 (t, H, W)，"
            f"当前为 {grid_shape}"
        )

    # 防止浮点网格尺寸被 int() 静默截断。
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        for value in grid_shape
    ):
        raise ValueError(
            "grid_shape 中的 t、H、W 必须为整数，"
            f"当前为 {grid_shape}"
        )

    t, grid_h, grid_w = grid_shape

    if t <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            "grid_shape 中的 t、H、W 必须均为正整数，"
            f"当前为 {grid_shape}"
        )

    return t, grid_h, grid_w


def build_directional_indices(
    grid_h: int,
    grid_w: int,
    direction: str,
    device: torch.device,
) -> torch.Tensor:
    """
    构造单帧二维网格在指定方向下的扫描索引。

    以 H=2、W=3 为例，原始网格为：
        0  1  2
        3  4  5

    RF:
        0, 1, 2, 3, 4, 5

    RR:
        5, 4, 3, 2, 1, 0

    CF:
        0, 3, 1, 4, 2, 5

    CR:
        5, 2, 4, 1, 3, 0

    特别注意：
        RR 是完整行优先序列的逆序，
        CR 是完整列优先序列的逆序，
        并不是逐行/逐列单独翻转。

    Returns:
        scan_indices: [H * W]
    """
    if direction not in DIRECTION_NAMES:
        raise ValueError(
            f"未知扫描方向 {direction}，"
            f"支持的方向为 {DIRECTION_NAMES}"
        )

    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            "grid_h 和 grid_w 必须均大于 0，"
            f"当前为 ({grid_h}, {grid_w})"
        )

    # 以标准 row-major 顺序给二维网格编号。
    # shape: [H * W]
    row_major_indices = torch.arange(
        grid_h * grid_w,
        dtype=torch.long,
        device=device,
    )

    if direction in ("RF", "RR"):
        indices = row_major_indices
    else:
        # [H * W] -> [H, W] -> [W, H] -> [H * W]
        # 得到 column-major 扫描序列。
        indices = (
            row_major_indices
            .reshape(grid_h, grid_w)
            .transpose(0, 1)
            .reshape(-1)
        )

    # 反向扫描：对整个扫描序列逆序。
    if direction in ("RR", "CR"):
        indices = torch.flip(
            indices,
            dims=(0,),
        )

    return indices


def directional_scan(
    x: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    direction: str,
) -> torch.Tensor:
    """
    对一张图片的空间 token 执行指定方向的排列。

    Args:
        x:
            一张图片或一个视频条目经过 Shared Down 的特征。
            shape: [1, N, C]

        grid_shape:
            空间合并后的 (t, H, W)，满足 N=t*H*W。

        direction:
            "RF"、"RR"、"CF" 或 "CR"。

    Returns:
        scanned_x:
            [t, H*W, C]

    这里将 t 放入 batch 维度，而不是序列维度。

    这样每个时间帧独立从初始 SSM 状态开始计算，
    不会把前一帧的状态传给后一帧。
    """
    t, grid_h, grid_w = validate_grid_shape(
        grid_shape
    )

    if x.ndim != 3 or x.size(0) != 1:
        raise ValueError(
            "directional_scan 要求 x 为 [1, N, C]，"
            f"实际 shape={tuple(x.shape)}"
        )

    expected_tokens = t * grid_h * grid_w

    if x.size(1) != expected_tokens:
        raise ValueError(
            "视觉 token 数量与 grid_shape 不匹配："
            f"tokens={x.size(1)}, "
            f"grid_shape={grid_shape}, "
            f"expected_tokens={expected_tokens}"
        )

    channels = x.size(-1)
    tokens_per_frame = grid_h * grid_w

    # [1, t*H*W, C] -> [t, H*W, C]
    frames = x.reshape(
        t,
        tokens_per_frame,
        channels,
    )

    scan_indices = build_directional_indices(
        grid_h=grid_h,
        grid_w=grid_w,
        direction=direction,
        device=x.device,
    )

    # 对每个帧应用同一组扫描索引。
    # 每帧保持独立，不跨帧拼接 SSM 序列。
    return frames.index_select(
        dim=1,
        index=scan_indices,
    )


def inverse_directional_scan(
    scanned_x: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    direction: str,
) -> torch.Tensor:
    """
    把指定方向的 SSM 输出恢复成原始 row-major token 顺序。

    Args:
        scanned_x:
            shape: [t, H*W, C]

        grid_shape:
            空间合并后的 (t, H, W)。

        direction:
            与正向扫描使用的方向相同。

    Returns:
        restored_x:
            shape: [1, t*H*W, C]

    原理：
        scanned_x = original_x[scan_indices]

        因此：
        original_x = scanned_x[argsort(scan_indices)]

    保持可微，不调用 detach() 或在原张量上做索引赋值。
    """
    t, grid_h, grid_w = validate_grid_shape(
        grid_shape
    )

    if scanned_x.ndim != 3:
        raise ValueError(
            "inverse_directional_scan 要求输入为 [t, H*W, C]，"
            f"实际 shape={tuple(scanned_x.shape)}"
        )

    tokens_per_frame = grid_h * grid_w

    if (
        scanned_x.size(0) != t
        or scanned_x.size(1) != tokens_per_frame
    ):
        raise ValueError(
            "逆扫描的输入形状与 grid_shape 不匹配："
            f"input={tuple(scanned_x.shape)}, "
            f"grid_shape={grid_shape}"
        )

    scan_indices = build_directional_indices(
        grid_h=grid_h,
        grid_w=grid_w,
        direction=direction,
        device=scanned_x.device,
    )

    inverse_indices = torch.argsort(
        scan_indices
    )

    restored_frames = scanned_x.index_select(
        dim=1,
        index=inverse_indices,
    )

    # [t, H*W, C] -> [1, t*H*W, C]
    return restored_frames.reshape(
        1,
        t * tokens_per_frame,
        scanned_x.size(-1),
    )

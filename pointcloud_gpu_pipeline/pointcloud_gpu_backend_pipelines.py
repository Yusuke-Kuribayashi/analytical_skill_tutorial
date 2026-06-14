"""Numba CUDA の GPU 常駐点群生成を提供する。"""

from __future__ import annotations

from typing import Any

import numpy as np

from pointcloud_data_types import RgbdInput

try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None

try:
    from numba import cuda  # type: ignore[import-untyped]
except ImportError:
    cuda = None

if cuda is not None:

    @cuda.jit
    def _numba_rgbd_to_flat_pointcloud_kernel(
        rgb: Any,
        depth: Any,
        output: Any,
        valid_mask: Any,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        index = cuda.grid(1)
        height = depth.shape[0]
        width = depth.shape[1]
        point_count = height * width
        if index >= point_count:
            return

        y_pos = index // width
        x_pos = index - y_pos * width
        z_value = depth[y_pos, x_pos]
        valid = z_value == z_value and min_depth < z_value < max_depth
        valid_mask[index] = 1 if valid else 0
        if valid:
            output[index, 0] = (x_pos - cx) * z_value / fx
            output[index, 1] = (y_pos - cy) * z_value / fy
            output[index, 2] = z_value
        else:
            output[index, 0] = np.nan
            output[index, 1] = np.nan
            output[index, 2] = np.nan
        output[index, 3] = rgb[y_pos, x_pos, 0]
        output[index, 4] = rgb[y_pos, x_pos, 1]
        output[index, 5] = rgb[y_pos, x_pos, 2]
else:
    _numba_rgbd_to_flat_pointcloud_kernel = None


def require_cupy_for_gpu_pipeline() -> Any:
    """GPU 常駐パイプライン用の CuPy モジュールを返す。

    Returns:
        import 済み CuPy モジュール。

    Raises:
        RuntimeError: CuPy が利用できない場合。
    """
    if cp is None:
        msg = "CuPy がインストールされていません。"
        raise RuntimeError(msg)
    return cp


def validate_rgbd_shape(data: RgbdInput) -> None:
    """RGB と depth の形状を検証する。

    Args:
        data: RGBD 入力。

    Raises:
        ValueError: RGB と depth の画像サイズが一致しない場合。
    """
    if data.rgb.shape[:2] != data.depth.shape:
        msg = f"RGB {data.rgb.shape[:2]} と depth {data.depth.shape} のサイズが一致しません。"
        raise ValueError(msg)


def is_numba_cuda_pipeline_available() -> bool:
    """Numba CUDA GPU 常駐パイプラインが利用可能か返す。

    Returns:
        CuPy と Numba CUDA が利用可能なら True。
    """
    return cp is not None and cuda is not None and cuda.is_available()


def create_pointcloud_from_rgbd_numba_gpu(data: RgbdInput) -> Any:
    """Numba CUDA で RGBD から GPU 上の標準形式点群を作成する。

    Args:
        data: RGBD 入力。

    Returns:
        CuPy GPU 配列の Nx6 `[x, y, z, r, g, b]` 点群。

    Raises:
        RuntimeError: Numba CUDA または CuPy が利用できない場合。
        ValueError: RGB と depth の画像サイズが一致しない場合。
    """
    cupy = require_cupy_for_gpu_pipeline()
    validate_rgbd_shape(data)
    if (
        not is_numba_cuda_pipeline_available()
        or _numba_rgbd_to_flat_pointcloud_kernel is None
    ):
        msg = "Numba CUDA が利用できません。"
        raise RuntimeError(msg)

    height, width = data.depth.shape
    point_count = height * width
    rgb_device = cuda.to_device(data.rgb.astype(np.float32, copy=False))
    depth_device = cuda.to_device(data.depth.astype(np.float32, copy=False))
    output_device = cuda.device_array((point_count, 6), dtype=np.float32)
    valid_device = cuda.device_array(point_count, dtype=np.uint8)
    threads = 256
    blocks = (point_count + threads - 1) // threads
    _numba_rgbd_to_flat_pointcloud_kernel[blocks, threads](
        rgb_device,
        depth_device,
        output_device,
        valid_device,
        data.intrinsics.fx,
        data.intrinsics.fy,
        data.intrinsics.cx,
        data.intrinsics.cy,
        data.min_depth,
        data.max_depth,
    )
    output_gpu = cupy.asarray(output_device)
    valid_gpu = cupy.asarray(valid_device).astype(cupy.bool_)
    return output_gpu[valid_gpu].astype(cupy.float32, copy=False)


def synchronize_numba_cuda_pipeline() -> None:
    """Numba CUDA と CuPy の GPU 処理を同期する。"""
    if cuda is not None:
        cuda.synchronize()
    if cp is not None:
        cp.cuda.Stream.null.synchronize()

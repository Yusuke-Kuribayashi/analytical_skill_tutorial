"""RGBD から標準形式点群を生成するバックエンド実装を提供する。"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np

from pointcloud_cpu_algorithms import (
    create_dense_pointcloud_from_rgbd,
    create_pointcloud_from_rgbd,
    dense_to_standard_pointcloud,
)
from pointcloud_data_types import RgbdInput

try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None

try:
    from numba import cuda  # type: ignore[import-untyped]
except ImportError:
    cuda = None

create_pointcloud_open3d: Callable[[RgbdInput], np.ndarray] | None
create_pointcloud_open3d_cuda: Callable[[RgbdInput], Any] | None
open3d_cuda_to_numpy: Callable[[Any], np.ndarray] | None
synchronize_open3d_cuda: Callable[[], None] | None
try:
    from pointcloud_open3d_pipeline import (
        create_pointcloud_open3d,
        create_pointcloud_open3d_cuda,
        is_open3d_available,
        is_open3d_cuda_available,
        open3d_cuda_to_numpy,
        synchronize_open3d_cuda,
    )
except ImportError:
    create_pointcloud_open3d = None
    create_pointcloud_open3d_cuda = None
    open3d_cuda_to_numpy = None
    synchronize_open3d_cuda = None

    def is_open3d_cuda_available() -> bool:
        """Open3D CUDA が利用可能か返す。

        Returns:
            利用できないため False。
        """
        return False

    def is_open3d_available() -> bool:
        """Open3D が利用可能か返す。

        Returns:
            利用できないため False。
        """
        return False


class PointcloudGenerationBackend(Protocol):
    """RGBD 点群生成バックエンドのインターフェース。"""

    name: str

    def is_available(self) -> bool:
        """バックエンドが利用可能か返す。

        Returns:
            利用可能なら True。
        """

    def create_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から標準形式点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            Nx6 の `[x, y, z, r, g, b]` 点群。
        """

    def synchronize(self) -> None:
        """非同期 GPU 処理を同期する。"""


class NumpyPointcloudBackend:
    """NumPy による CPU 点群生成バックエンド。"""

    name = "numpy_cpu"

    def is_available(self) -> bool:
        """バックエンドが利用可能か返す。

        Returns:
            常に True。
        """
        return True

    def create_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から標準形式点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            Nx6 の `[x, y, z, r, g, b]` 点群。
        """
        return create_pointcloud_from_rgbd(data)

    def create_dense_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から dense 点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            HxWx6 の dense 点群。
        """
        return create_dense_pointcloud_from_rgbd(data)

    def synchronize(self) -> None:
        """CPU 処理のため同期は行わない。"""


class CupyPointcloudBackend:
    """CuPy による GPU 点群生成バックエンド。"""

    name = "cupy_gpu"

    def is_available(self) -> bool:
        """バックエンドが利用可能か返す。

        Returns:
            CuPy が import 可能なら True。
        """
        return cp is not None

    def create_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から標準形式点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            Nx6 の `[x, y, z, r, g, b]` 点群。

        Raises:
            RuntimeError: CuPy が利用できない場合。
        """
        dense = self.create_dense_pointcloud(data)
        return dense_to_standard_pointcloud(dense)

    def create_dense_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から dense 点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            HxWx6 の dense 点群。

        Raises:
            RuntimeError: CuPy が利用できない場合。
        """
        if cp is None:
            msg = "CuPy がインストールされていません。"
            raise RuntimeError(msg)
        depth = cp.asarray(data.depth, dtype=cp.float32)
        rgb = cp.asarray(data.rgb, dtype=cp.float32)
        height, width = depth.shape
        u_grid, v_grid = cp.meshgrid(cp.arange(width), cp.arange(height))
        valid = cp.isfinite(depth) & (depth > data.min_depth) & (depth < data.max_depth)
        output = cp.empty((height, width, 6), dtype=cp.float32)
        output[..., :3] = cp.nan
        output[..., 3:6] = rgb
        output[..., 0] = cp.where(
            valid, (u_grid - data.intrinsics.cx) * depth / data.intrinsics.fx, cp.nan
        )
        output[..., 1] = cp.where(
            valid, (v_grid - data.intrinsics.cy) * depth / data.intrinsics.fy, cp.nan
        )
        output[..., 2] = cp.where(valid, depth, cp.nan)
        return cp.asnumpy(output)

    def synchronize(self) -> None:
        """CuPy のデフォルト stream を同期する。"""
        if cp is not None:
            cp.cuda.Stream.null.synchronize()


if cuda is not None:

    @cuda.jit
    def _rgbd_to_pointcloud_kernel(
        rgb: Any,
        depth: Any,
        output: Any,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        y_pos, x_pos = cuda.grid(2)
        height = depth.shape[0]
        width = depth.shape[1]
        if y_pos >= height or x_pos >= width:
            return

        z_value = depth[y_pos, x_pos]
        valid = z_value == z_value and min_depth < z_value < max_depth
        if valid:
            output[y_pos, x_pos, 0] = (x_pos - cx) * z_value / fx
            output[y_pos, x_pos, 1] = (y_pos - cy) * z_value / fy
            output[y_pos, x_pos, 2] = z_value
        else:
            output[y_pos, x_pos, 0] = np.nan
            output[y_pos, x_pos, 1] = np.nan
            output[y_pos, x_pos, 2] = np.nan
        output[y_pos, x_pos, 3] = rgb[y_pos, x_pos, 0]
        output[y_pos, x_pos, 4] = rgb[y_pos, x_pos, 1]
        output[y_pos, x_pos, 5] = rgb[y_pos, x_pos, 2]
else:
    _rgbd_to_pointcloud_kernel = None


class NumbaCudaPointcloudBackend:
    """Numba CUDA による GPU 点群生成バックエンド。"""

    name = "numba_cuda"

    def is_available(self) -> bool:
        """バックエンドが利用可能か返す。

        Returns:
            Numba CUDA が実行可能なら True。
        """
        return cuda is not None and cuda.is_available()

    def create_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から標準形式点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            Nx6 の `[x, y, z, r, g, b]` 点群。
        """
        dense = self.create_dense_pointcloud(data)
        return dense_to_standard_pointcloud(dense)

    def create_dense_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から dense 点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            HxWx6 の dense 点群。

        Raises:
            RuntimeError: Numba CUDA が利用できない場合。
        """
        if not self.is_available() or _rgbd_to_pointcloud_kernel is None:
            msg = "Numba CUDA が利用できません。"
            raise RuntimeError(msg)
        height, width = data.depth.shape
        rgb_device = cuda.to_device(data.rgb.astype(np.float32))
        depth_device = cuda.to_device(data.depth.astype(np.float32))
        output_device = cuda.device_array((height, width, 6), dtype=np.float32)
        threads = (16, 16)
        blocks = (
            (height + threads[0] - 1) // threads[0],
            (width + threads[1] - 1) // threads[1],
        )
        _rgbd_to_pointcloud_kernel[blocks, threads](
            rgb_device,
            depth_device,
            output_device,
            data.intrinsics.fx,
            data.intrinsics.fy,
            data.intrinsics.cx,
            data.intrinsics.cy,
            data.min_depth,
            data.max_depth,
        )
        self.synchronize()
        return output_device.copy_to_host()

    def synchronize(self) -> None:
        """CUDA device を同期する。"""
        if cuda is not None:
            cuda.synchronize()


class Open3DPointcloudBackend:
    """Open3D による CPU 点群生成バックエンド。"""

    name = "open3d_cpu"

    def is_available(self) -> bool:
        """バックエンドが利用可能か返す。

        Returns:
            Open3D が import 可能なら True。
        """
        return is_open3d_available()

    def create_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から標準形式点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            Nx6 の `[x, y, z, r, g, b]` 点群。
        """
        if create_pointcloud_open3d is None:
            msg = "Open3D がインストールされていません。"
            raise RuntimeError(msg)
        return create_pointcloud_open3d(data)

    def synchronize(self) -> None:
        """CPU 処理のため同期は行わない。"""


class Open3DCudaPointcloudBackend:
    """Open3D CUDA tensor による点群生成バックエンド。"""

    name = "open3d_cuda"

    def is_available(self) -> bool:
        """バックエンドが利用可能か返す。

        Returns:
            Open3D CUDA が利用可能なら True。
        """
        return is_open3d_cuda_available()

    def create_pointcloud(self, data: RgbdInput) -> np.ndarray:
        """RGBD から標準形式点群を作成する。

        Args:
            data: RGBD 入力。

        Returns:
            Nx6 の `[x, y, z, r, g, b]` 点群。
        """
        if create_pointcloud_open3d_cuda is None or open3d_cuda_to_numpy is None:
            msg = "Open3D CUDA が利用できません。"
            raise RuntimeError(msg)
        return open3d_cuda_to_numpy(create_pointcloud_open3d_cuda(data))

    def synchronize(self) -> None:
        """Open3D CUDA 処理を同期する。"""
        if synchronize_open3d_cuda is not None:
            synchronize_open3d_cuda()


def default_generation_backends() -> list[PointcloudGenerationBackend]:
    """標準の点群生成バックエンドを返す。

    Returns:
        NumPy、CuPy、Numba CUDA、Open3D のバックエンド一覧。
    """
    return [
        NumpyPointcloudBackend(),
        CupyPointcloudBackend(),
        NumbaCudaPointcloudBackend(),
        Open3DCudaPointcloudBackend(),
    ]


# 旧 API 互換用エイリアス。
PointcloudBackend = PointcloudGenerationBackend
NumpyBackend = NumpyPointcloudBackend
CupyBackend = CupyPointcloudBackend
NumbaCudaBackend = NumbaCudaPointcloudBackend
Open3DBackend = Open3DPointcloudBackend
Open3DCudaBackend = Open3DCudaPointcloudBackend


def default_backends() -> list[PointcloudGenerationBackend]:
    """旧 API 互換用に標準バックエンドを返す。

    Returns:
        標準の点群生成バックエンド一覧。
    """
    return default_generation_backends()

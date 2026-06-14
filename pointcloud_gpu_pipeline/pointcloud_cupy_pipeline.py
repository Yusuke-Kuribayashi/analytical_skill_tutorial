"""CuPy による点群処理パイプラインの GPU 実装を提供する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pointcloud_data_types import RgbdInput

try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None


@dataclass(frozen=True)
class CupyPlaneResult:
    """CuPy 平面検出結果を保持する。

    Args:
        coefficients: GPU 上の平面係数 `[a, b, c, d]`。
        inlier_indices: GPU 上の平面 inlier index。
        remaining_pointcloud: GPU 上の平面除去後点群。
    """

    coefficients: Any
    inlier_indices: Any
    remaining_pointcloud: Any


def require_cupy() -> Any:
    """CuPy モジュールを返す。

    Returns:
        import 済みの CuPy モジュール。

    Raises:
        RuntimeError: CuPy が利用できない場合。
    """
    if cp is None:
        msg = "CuPy がインストールされていません。"
        raise RuntimeError(msg)
    return cp


def create_pointcloud_from_rgbd_gpu(data: RgbdInput) -> Any:
    """RGBD から GPU 上の標準形式点群を作成する。

    Args:
        data: RGBD 入力。

    Returns:
        GPU 上の Nx6 `[x, y, z, r, g, b]` 点群。

    Raises:
        ValueError: RGB と depth の画像サイズが一致しない場合。
    """
    cupy = require_cupy()
    if data.rgb.shape[:2] != data.depth.shape:
        msg = f"RGB {data.rgb.shape[:2]} と depth {data.depth.shape} のサイズが一致しません。"
        raise ValueError(msg)

    depth = cupy.asarray(data.depth, dtype=cupy.float32)
    rgb = cupy.asarray(data.rgb, dtype=cupy.float32)
    height, width = depth.shape
    u_grid, v_grid = cupy.meshgrid(
        cupy.arange(width, dtype=cupy.float32),
        cupy.arange(height, dtype=cupy.float32),
    )
    valid = cupy.isfinite(depth) & (depth > data.min_depth) & (depth < data.max_depth)
    z = depth[valid]
    x = (u_grid[valid] - data.intrinsics.cx) * z / data.intrinsics.fx
    y = (v_grid[valid] - data.intrinsics.cy) * z / data.intrinsics.fy
    colors = rgb[valid]
    return cupy.column_stack(
        (x, y, z, colors[:, 0], colors[:, 1], colors[:, 2])
    ).astype(cupy.float32)


def voxel_downsample_pointcloud_gpu(pointcloud: Any, voxel_size: float) -> Any:
    """GPU 上でボクセルグリッド方式のダウンサンプリングを行う。

    Args:
        pointcloud: GPU 上の Nx6 点群。
        voxel_size: ボクセルの一辺の長さ[m]。

    Returns:
        GPU 上のダウンサンプリング後点群。

    Raises:
        ValueError: voxel_size が不正な場合。
    """
    cupy = require_cupy()
    if voxel_size <= 0.0:
        msg = "voxel_size は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if pointcloud.shape[0] == 0:
        return pointcloud.copy()

    voxel_indices = cupy.floor(pointcloud[:, :3] / voxel_size).astype(cupy.int64)
    min_indices = cupy.min(voxel_indices, axis=0)
    shifted = voxel_indices - min_indices[None, :]
    ranges = cupy.max(shifted, axis=0) + 1
    keys = (
        shifted[:, 0] * ranges[1] * ranges[2]
        + shifted[:, 1] * ranges[2]
        + shifted[:, 2]
    )
    _, inverse = cupy.unique(keys, return_inverse=True)
    voxel_count = int(cupy.max(inverse).get()) + 1
    counts = cupy.bincount(inverse, minlength=voxel_count).astype(cupy.float32)
    columns = []
    for column_index in range(pointcloud.shape[1]):
        sums = cupy.bincount(
            inverse,
            weights=pointcloud[:, column_index],
            minlength=voxel_count,
        )
        columns.append(sums / counts)
    return cupy.stack(columns, axis=1).astype(cupy.float32)


def sample_ransac_indices(
    point_count: int, max_iterations: int, random_seed: int
) -> np.ndarray:
    """RANSAC 用の 3 点サンプル index を作成する。

    Args:
        point_count: 点数。
        max_iterations: RANSAC の最大反復回数。
        random_seed: 乱数 seed。

    Returns:
        max_iterations x 3 のサンプル index。
    """
    rng = np.random.default_rng(random_seed)
    samples = np.empty((max_iterations, 3), dtype=np.int64)
    for iteration in range(max_iterations):
        samples[iteration] = rng.choice(point_count, size=3, replace=False)
    return samples


def detect_plane_ransac_gpu(
    pointcloud: Any,
    distance_threshold: float,
    max_iterations: int,
    min_inlier_ratio: float,
    random_seed: int,
) -> CupyPlaneResult:
    """GPU 上で RANSAC 平面検出と平面削除を行う。

    Args:
        pointcloud: GPU 上の Nx6 点群。
        distance_threshold: 平面 inlier とみなす距離しきい値[m]。
        max_iterations: RANSAC の最大反復回数。
        min_inlier_ratio: 平面として採用する最小 inlier 比率。
        random_seed: 乱数 seed。

    Returns:
        GPU 上の平面検出結果。

    Raises:
        ValueError: 点数またはパラメータが不正な場合。
    """
    cupy = require_cupy()
    point_count = int(pointcloud.shape[0])
    if point_count < 3:
        msg = "平面推定には 3 点以上の点群が必要です。"
        raise ValueError(msg)
    if distance_threshold <= 0.0 or max_iterations <= 0:
        msg = "平面検出パラメータが不正です。"
        raise ValueError(msg)
    if not 0.0 <= min_inlier_ratio <= 1.0:
        msg = "min_inlier_ratio は 0 以上 1 以下を指定してください。"
        raise ValueError(msg)

    xyz = pointcloud[:, :3].astype(cupy.float32, copy=False)
    sample_indices = cupy.asarray(
        sample_ransac_indices(point_count, max_iterations, random_seed),
        dtype=cupy.int64,
    )
    samples = xyz[sample_indices]
    normals = cupy.cross(samples[:, 1] - samples[:, 0], samples[:, 2] - samples[:, 0])
    norms = cupy.linalg.norm(normals, axis=1)
    valid_planes = norms > 0.0
    safe_norms = cupy.where(valid_planes, norms, 1.0)
    normals = normals / safe_norms[:, None]
    offsets = -cupy.sum(normals * samples[:, 0], axis=1)
    distances = cupy.abs(xyz @ normals.T + offsets[None, :])
    inlier_matrix = (distances <= distance_threshold) & valid_planes[None, :]
    inlier_counts = cupy.sum(inlier_matrix, axis=0)
    best_index = int(cupy.argmax(inlier_counts).get())
    best_count = int(inlier_counts[best_index].get())
    min_inliers = max(3, int(np.ceil(point_count * min_inlier_ratio)))
    if best_count < min_inliers:
        coefficients = cupy.full(4, cupy.nan, dtype=cupy.float32)
        return CupyPlaneResult(
            coefficients=coefficients,
            inlier_indices=cupy.asarray([], dtype=cupy.int64),
            remaining_pointcloud=pointcloud.copy(),
        )

    inlier_mask = inlier_matrix[:, best_index]
    inlier_indices = cupy.flatnonzero(inlier_mask).astype(cupy.int64)
    remaining = pointcloud[~inlier_mask]
    coefficients = cupy.concatenate(
        (normals[best_index], offsets[best_index : best_index + 1])
    )
    return CupyPlaneResult(
        coefficients=coefficients.astype(cupy.float32),
        inlier_indices=inlier_indices,
        remaining_pointcloud=remaining.astype(cupy.float32, copy=False),
    )


def connected_components_labels_gpu(rows: Any, cols: Any, point_count: int) -> Any:
    """GPU ラベル伝播で connected components の root を計算する。

    Args:
        rows: 近傍エッジの始点 index GPU 配列。
        cols: 近傍エッジの終点 index GPU 配列。
        point_count: 点数。

    Returns:
        各点の root index を格納した GPU 配列。
    """
    cupy = require_cupy()
    labels = cupy.arange(point_count, dtype=cupy.int64)
    edge_count = int(rows.shape[0])
    if edge_count == 0:
        return labels

    for _ in range(point_count):
        next_labels = labels.copy()
        cupy.minimum.at(next_labels, rows, labels[cols])
        cupy.minimum.at(next_labels, cols, labels[rows])
        if bool(cupy.all(next_labels == labels).get()):
            break
        labels = next_labels
    return labels


def euclidean_cluster_labels_gpu(
    pointcloud: Any,
    cluster_tolerance: float,
    cluster_min_size: int,
    cluster_max_size: int | None = None,
) -> Any:
    """GPU 上で近傍抽出とラベル伝播を行いクラスタラベルを計算する。

    Args:
        pointcloud: GPU 上の Nx6 点群。
        cluster_tolerance: 同一クラスタとみなす距離しきい値[m]。
        cluster_min_size: 出力するクラスタの最小点数。
        cluster_max_size: 出力するクラスタの最大点数。None の場合は上限なし。

    Returns:
        GPU 上のクラスタラベル。除外点は -1。

    Raises:
        ValueError: パラメータが不正な場合。
    """
    cupy = require_cupy()
    if cluster_tolerance <= 0.0:
        msg = "cluster_tolerance は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if cluster_min_size <= 0:
        msg = "cluster_min_size は 1 以上を指定してください。"
        raise ValueError(msg)
    if cluster_max_size is not None and cluster_max_size < cluster_min_size:
        msg = "cluster_max_size は cluster_min_size 以上を指定してください。"
        raise ValueError(msg)

    point_count = int(pointcloud.shape[0])
    if point_count == 0:
        return cupy.asarray([], dtype=cupy.int64)

    xyz = pointcloud[:, :3].astype(cupy.float32, copy=False)
    diff = xyz[:, None, :] - xyz[None, :, :]
    adjacency = cupy.sum(diff * diff, axis=2) <= cluster_tolerance * cluster_tolerance
    rows_gpu, cols_gpu = cupy.nonzero(cupy.triu(adjacency, k=1))
    roots = connected_components_labels_gpu(rows_gpu, cols_gpu, point_count)
    unique_roots, inverse, counts = cupy.unique(
        roots, return_inverse=True, return_counts=True
    )
    valid_cluster = counts >= cluster_min_size
    if cluster_max_size is not None:
        valid_cluster &= counts <= cluster_max_size

    labels = cupy.full(point_count, -1, dtype=cupy.int64)
    compact_ids = cupy.cumsum(valid_cluster.astype(cupy.int64)) - 1
    valid_points = valid_cluster[inverse]
    labels[valid_points] = compact_ids[inverse[valid_points]]
    _ = unique_roots
    return labels.astype(cupy.int64, copy=False)


def labels_to_clusters(labels: Any) -> list[np.ndarray]:
    """GPU 上のクラスタラベルを CPU の index 配列一覧へ変換する。

    Args:
        labels: GPU 上のクラスタラベル。

    Returns:
        クラスタごとの CPU index 配列一覧。
    """
    labels_cpu = cp.asnumpy(labels) if cp is not None else np.asarray(labels)
    clusters: list[np.ndarray] = []
    for label in sorted(int(value) for value in np.unique(labels_cpu) if value >= 0):
        clusters.append(np.flatnonzero(labels_cpu == label).astype(np.int64))
    return clusters


def synchronize_cupy() -> None:
    """CuPy のデフォルト stream を同期する。"""
    cupy = require_cupy()
    cupy.cuda.Stream.null.synchronize()

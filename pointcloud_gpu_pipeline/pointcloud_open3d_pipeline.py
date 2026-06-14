"""Open3D による点群処理パイプラインを提供する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pointcloud_cpu_algorithms import create_pointcloud_from_rgbd
from pointcloud_data_types import PlaneSegmentationResult, RgbdInput

try:
    import open3d as o3d  # type: ignore[import-not-found]
except ImportError:
    o3d = None


@dataclass(frozen=True)
class Open3dPipelineOutput:
    """Open3D パイプライン出力を保持する。

    Args:
        pointcloud: RGBD から生成した点群。
        downsampled_pointcloud: VoxelGrid 後点群。
        plane: 平面削除結果。
        clusters: クラスタ index 一覧。
    """

    pointcloud: np.ndarray
    downsampled_pointcloud: np.ndarray
    plane: PlaneSegmentationResult
    clusters: list[np.ndarray]


def require_open3d() -> Any:
    """Open3D モジュールを返す。

    Returns:
        import 済み Open3D モジュール。

    Raises:
        RuntimeError: Open3D が利用できない場合。
    """
    if o3d is None:
        msg = "Open3D がインストールされていません。"
        raise RuntimeError(msg)
    return o3d


def is_open3d_available() -> bool:
    """Open3D が利用可能か返す。

    Returns:
        import 可能なら True。
    """
    return o3d is not None


def create_pointcloud_open3d(data: RgbdInput) -> np.ndarray:
    """RGBD から Open3D 評価用の標準形式点群を作成する。

    Args:
        data: RGBD 入力。

    Returns:
        Nx6 の `[x, y, z, r, g, b]` 点群。
    """
    require_open3d()
    return create_pointcloud_from_rgbd(data)


def numpy_to_open3d(pointcloud: np.ndarray) -> Any:
    """標準形式点群を Open3D PointCloud へ変換する。

    Args:
        pointcloud: Nx6 の `[x, y, z, r, g, b]` 点群。

    Returns:
        Open3D PointCloud。
    """
    open3d = require_open3d()
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(
        pointcloud[:, :3].astype(np.float64, copy=False)
    )
    cloud.colors = open3d.utility.Vector3dVector(
        np.clip(pointcloud[:, 3:6] / 255.0, 0.0, 1.0).astype(np.float64, copy=False)
    )
    return cloud


def open3d_to_numpy(cloud: Any) -> np.ndarray:
    """Open3D PointCloud を標準形式点群へ変換する。

    Args:
        cloud: Open3D PointCloud。

    Returns:
        Nx6 の `[x, y, z, r, g, b]` 点群。
    """
    xyz = np.asarray(cloud.points, dtype=np.float32)
    colors = (np.asarray(cloud.colors, dtype=np.float32) * 255.0).astype(np.float32)
    if xyz.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.column_stack((xyz, colors)).astype(np.float32, copy=False)


def voxel_downsample_open3d(pointcloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """Open3D で VoxelGrid ダウンサンプリングを行う。

    Args:
        pointcloud: Nx6 の点群。
        voxel_size: ボクセルサイズ。

    Returns:
        ダウンサンプリング後点群。

    Raises:
        ValueError: voxel_size が不正な場合。
    """
    if voxel_size <= 0.0:
        msg = "voxel_size は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    cloud = numpy_to_open3d(pointcloud)
    downsampled = cloud.voxel_down_sample(voxel_size)
    return open3d_to_numpy(downsampled)


def detect_plane_ransac_open3d(
    pointcloud: np.ndarray,
    distance_threshold: float,
    max_iterations: int,
    min_inlier_ratio: float,
) -> PlaneSegmentationResult:
    """Open3D で RANSAC 平面削除を行う。

    Args:
        pointcloud: Nx6 の点群。
        distance_threshold: inlier 距離しきい値。
        max_iterations: 最大反復回数。
        min_inlier_ratio: 最小 inlier 比率。

    Returns:
        平面削除結果。

    Raises:
        ValueError: 点数またはパラメータが不正な場合。
    """
    if pointcloud.shape[0] < 3:
        msg = "平面推定には 3 点以上の点群が必要です。"
        raise ValueError(msg)
    if distance_threshold <= 0.0 or max_iterations <= 0:
        msg = "平面検出パラメータが不正です。"
        raise ValueError(msg)
    open3d = require_open3d()
    open3d.utility.random.seed(0)
    cloud = numpy_to_open3d(pointcloud)
    model, inliers = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=max_iterations,
    )
    min_inliers = max(3, int(np.ceil(pointcloud.shape[0] * min_inlier_ratio)))
    if len(inliers) < min_inliers:
        return PlaneSegmentationResult(
            coefficients=np.full(4, np.nan, dtype=np.float32),
            inlier_indices=np.empty(0, dtype=np.int64),
            remaining_pointcloud=pointcloud.astype(np.float32, copy=True),
        )
    inlier_indices = np.asarray(inliers, dtype=np.int64)
    remaining = cloud.select_by_index(inliers, invert=True)
    return PlaneSegmentationResult(
        coefficients=np.asarray(model, dtype=np.float32),
        inlier_indices=inlier_indices,
        remaining_pointcloud=open3d_to_numpy(remaining),
    )


def euclidean_cluster_open3d(
    pointcloud: np.ndarray,
    cluster_tolerance: float,
    cluster_min_size: int,
    cluster_max_size: int | None = None,
) -> list[np.ndarray]:
    """Open3D DBSCAN で距離クラスタを抽出する。

    Args:
        pointcloud: Nx6 の点群。
        cluster_tolerance: 同一クラスタとみなす距離しきい値。
        cluster_min_size: 最小クラスタ点数。
        cluster_max_size: 最大クラスタ点数。

    Returns:
        クラスタ index 一覧。

    Raises:
        ValueError: パラメータが不正な場合。
    """
    if cluster_tolerance <= 0.0:
        msg = "cluster_tolerance は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if cluster_min_size <= 0:
        msg = "cluster_min_size は 1 以上を指定してください。"
        raise ValueError(msg)
    if pointcloud.shape[0] == 0:
        return []
    cloud = numpy_to_open3d(pointcloud)
    labels = np.asarray(
        cloud.cluster_dbscan(eps=cluster_tolerance, min_points=cluster_min_size),
        dtype=np.int64,
    )
    clusters: list[np.ndarray] = []
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        indices = np.flatnonzero(labels == label).astype(np.int64)
        if cluster_max_size is not None and indices.shape[0] > cluster_max_size:
            continue
        clusters.append(indices)
    return clusters


@dataclass(frozen=True)
class Open3dTensorPlaneResult:
    """Open3D tensor 平面削除結果を保持する。

    Args:
        coefficients: GPU 上の平面係数 tensor。
        inlier_indices: GPU 上の inlier index tensor。
        remaining_pointcloud: GPU 上の平面除去後点群。
    """

    coefficients: Any
    inlier_indices: Any
    remaining_pointcloud: Any


def is_open3d_cuda_available() -> bool:
    """Open3D CUDA tensor backend が利用可能か返す。

    Returns:
        Open3D CUDA が利用可能なら True。
    """
    return o3d is not None and bool(o3d.core.cuda.is_available())


def create_pointcloud_open3d_cuda(data: RgbdInput) -> Any:
    """RGBD から Open3D CUDA tensor 点群を作成する。

    Args:
        data: RGBD 入力。

    Returns:
        CUDA 上の Open3D tensor PointCloud。
    """
    open3d = require_open3d()
    pointcloud = create_pointcloud_from_rgbd(data)
    device = open3d.core.Device("CUDA:0")
    cloud = open3d.t.geometry.PointCloud(device)
    cloud.point.positions = open3d.core.Tensor(
        pointcloud[:, :3], dtype=open3d.core.Dtype.Float32, device=device
    )
    cloud.point.colors = open3d.core.Tensor(
        np.clip(pointcloud[:, 3:6] / 255.0, 0.0, 1.0),
        dtype=open3d.core.Dtype.Float32,
        device=device,
    )
    return cloud


def open3d_cuda_to_numpy(cloud: Any) -> np.ndarray:
    """Open3D CUDA tensor 点群を標準形式点群へ変換する。

    Args:
        cloud: Open3D tensor PointCloud。

    Returns:
        Nx6 の `[x, y, z, r, g, b]` 点群。
    """
    positions = cloud.point.positions.cpu().numpy().astype(np.float32, copy=False)
    colors = (cloud.point.colors.cpu().numpy() * 255.0).astype(np.float32, copy=False)
    if positions.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.column_stack((positions, colors)).astype(np.float32, copy=False)


def voxel_downsample_open3d_cuda(cloud: Any, voxel_size: float) -> Any:
    """Open3D CUDA tensor で VoxelGrid ダウンサンプリングを行う。

    Args:
        cloud: Open3D tensor PointCloud。
        voxel_size: ボクセルサイズ。

    Returns:
        ダウンサンプリング後の Open3D tensor PointCloud。
    """
    if voxel_size <= 0.0:
        msg = "voxel_size は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    return cloud.voxel_down_sample(voxel_size)


def detect_plane_ransac_open3d_cuda(
    cloud: Any,
    distance_threshold: float,
    max_iterations: int,
    min_inlier_ratio: float,
) -> Open3dTensorPlaneResult:
    """Open3D CUDA tensor で RANSAC 平面削除を行う。

    Args:
        cloud: Open3D tensor PointCloud。
        distance_threshold: inlier 距離しきい値。
        max_iterations: 最大反復回数。
        min_inlier_ratio: 最小 inlier 比率。

    Returns:
        Open3D tensor 平面削除結果。
    """
    point_count = int(cloud.point.positions.shape[0])
    if point_count < 3:
        msg = "平面推定には 3 点以上の点群が必要です。"
        raise ValueError(msg)
    open3d = require_open3d()
    open3d.utility.random.seed(0)
    model, inliers = cloud.segment_plane(distance_threshold, 3, max_iterations)
    min_inliers = max(3, int(np.ceil(point_count * min_inlier_ratio)))
    if int(inliers.shape[0]) < min_inliers:
        return Open3dTensorPlaneResult(model, inliers, cloud.clone())
    remaining = cloud.select_by_index(inliers, invert=True)
    return Open3dTensorPlaneResult(model, inliers, remaining)


def euclidean_cluster_open3d_cuda(
    cloud: Any,
    cluster_tolerance: float,
    cluster_min_size: int,
    cluster_max_size: int | None = None,
) -> Any:
    """Open3D CUDA tensor DBSCAN で距離クラスタを抽出する。

    Args:
        cloud: Open3D tensor PointCloud。
        cluster_tolerance: 距離しきい値。
        cluster_min_size: 最小クラスタ点数。
        cluster_max_size: 最大クラスタ点数。

    Returns:
        GPU 上のラベル tensor。
    """
    if cluster_tolerance <= 0.0:
        msg = "cluster_tolerance は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    labels = cloud.cluster_dbscan(cluster_tolerance, cluster_min_size)
    if cluster_max_size is None:
        return labels
    labels_cpu = labels.cpu().numpy()
    for label in np.unique(labels_cpu):
        if label < 0:
            continue
        if np.count_nonzero(labels_cpu == label) > cluster_max_size:
            labels_cpu[labels_cpu == label] = -1
    open3d = require_open3d()
    return open3d.core.Tensor(
        labels_cpu, dtype=open3d.core.Dtype.Int32, device=labels.device
    )


def open3d_cuda_labels_to_clusters(labels: Any) -> list[np.ndarray]:
    """Open3D CUDA tensor ラベルを CPU index 配列一覧へ変換する。

    Args:
        labels: GPU 上のクラスタラベル。

    Returns:
        クラスタ index 配列一覧。
    """
    labels_cpu = labels.cpu().numpy().astype(np.int64, copy=False)
    clusters: list[np.ndarray] = []
    for label in sorted(int(value) for value in np.unique(labels_cpu) if value >= 0):
        clusters.append(np.flatnonzero(labels_cpu == label).astype(np.int64))
    return clusters


def synchronize_open3d_cuda() -> None:
    """Open3D CUDA 処理を同期する。"""
    open3d = require_open3d()
    open3d.core.cuda.synchronize()

"""点群 CPU 基準アルゴリズムの仕様テスト。"""

from __future__ import annotations

import numpy as np
import pytest

from pointcloud_cpu_algorithms import (
    colorize_clusters,
    create_pointcloud_from_rgbd,
    detect_plane_ransac,
    euclidean_cluster_pointcloud,
    voxel_downsample_pointcloud,
)
from pointcloud_data_types import CameraIntrinsics, RgbdInput
from pointcloud_pipeline_benchmark import (
    is_backend_unavailable_error,
    measured_statistics,
    time_callable,
)


def test_create_pointcloud_from_rgbd_filters_invalid_depth() -> None:
    """RGBD から Nx6 点群を作成し、無効 depth を除外する。"""
    rgb = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[1.0, np.nan], [0.0, 2.0]], dtype=np.float32)
    data = RgbdInput(
        rgb=rgb,
        depth=depth,
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        min_depth=0.0,
        max_depth=10.0,
    )

    pointcloud = create_pointcloud_from_rgbd(data)

    expected = np.array(
        [
            [0.0, 0.0, 1.0, 10.0, 20.0, 30.0],
            [2.0, 2.0, 2.0, 100.0, 110.0, 120.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(pointcloud, expected)


def test_voxel_downsample_pointcloud_uses_centroid_and_mean_color() -> None:
    """VoxelGrid は同一ボクセル内の xyz と RGB の平均を返す。"""
    pointcloud = np.array(
        [
            [0.001, 0.001, 0.001, 10.0, 20.0, 30.0],
            [0.002, 0.002, 0.002, 30.0, 40.0, 50.0],
            [0.021, 0.0, 0.0, 100.0, 110.0, 120.0],
        ],
        dtype=np.float32,
    )

    downsampled = voxel_downsample_pointcloud(pointcloud, voxel_size=0.01)

    expected = np.array(
        [
            [0.0015, 0.0015, 0.0015, 20.0, 30.0, 40.0],
            [0.021, 0.0, 0.0, 100.0, 110.0, 120.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(downsampled, expected, atol=1e-6)


def test_detect_plane_ransac_removes_plane_inliers() -> None:
    """RANSAC 平面検出は平面点を inlier として除去する。"""
    plane_points = np.array(
        [[x, y, 1.0, 0.0, 0.0, 0.0] for x in range(3) for y in range(3)],
        dtype=np.float32,
    )
    outliers = np.array(
        [
            [0.2, 0.4, 1.8, 255.0, 0.0, 0.0],
            [1.4, 0.6, 2.6, 0.0, 255.0, 0.0],
            [2.1, 1.7, 3.2, 0.0, 0.0, 255.0],
        ],
        dtype=np.float32,
    )
    pointcloud = np.vstack((plane_points, outliers))

    result = detect_plane_ransac(
        pointcloud,
        distance_threshold=0.001,
        max_iterations=100,
        min_inlier_ratio=0.5,
        random_seed=1,
    )

    assert result.inlier_indices.shape[0] == 9
    assert result.remaining_pointcloud.shape[0] == 3
    assert np.isclose(np.linalg.norm(result.coefficients[:3]), 1.0)
    np.testing.assert_allclose(result.remaining_pointcloud, outliers)


def test_euclidean_cluster_pointcloud_filters_small_clusters() -> None:
    """Euclidean clustering は距離で分離したクラスタを返し、小さすぎるクラスタを除外する。"""
    pointcloud = np.array(
        [
            [0.00, 0.00, 0.00, 0.0, 0.0, 0.0],
            [0.01, 0.00, 0.00, 0.0, 0.0, 0.0],
            [1.00, 1.00, 1.00, 0.0, 0.0, 0.0],
            [1.01, 1.00, 1.00, 0.0, 0.0, 0.0],
            [3.00, 3.00, 3.00, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    clusters = euclidean_cluster_pointcloud(
        pointcloud,
        cluster_tolerance=0.05,
        cluster_min_size=2,
    )

    assert len(clusters) == 2
    np.testing.assert_array_equal(clusters[0], np.array([0, 1]))
    np.testing.assert_array_equal(clusters[1], np.array([2, 3]))


def test_colorize_clusters_assigns_cluster_colors_and_noise_color() -> None:
    """クラスタ点は固定色、未所属点はノイズ色へ置き換える。"""
    pointcloud = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [2.0, 0.0, 0.0, 3.0, 3.0, 3.0],
            [3.0, 0.0, 0.0, 4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )

    colored = colorize_clusters(
        pointcloud,
        [np.array([0, 2]), np.array([1])],
        noise_color=(9.0, 9.0, 9.0),
    )

    np.testing.assert_allclose(colored[:, :3], pointcloud[:, :3])
    np.testing.assert_allclose(colored[0, 3:6], np.array([230.0, 25.0, 75.0]))
    np.testing.assert_allclose(colored[2, 3:6], np.array([230.0, 25.0, 75.0]))
    np.testing.assert_allclose(colored[1, 3:6], np.array([60.0, 180.0, 75.0]))
    np.testing.assert_allclose(colored[3, 3:6], np.array([9.0, 9.0, 9.0]))


def test_time_callable_excludes_first_run_from_statistics() -> None:
    """21 回計測時に 1 回目を除外した統計を計算する。"""
    counter = {"value": 0}

    def increment() -> int:
        counter["value"] += 1
        return counter["value"]

    result, timings = time_callable(increment, runs=21, warmup_runs=1)
    first_run_ms, mean_ms, min_ms, max_ms = measured_statistics(
        [100.0, *([10.0] * 20)], 1
    )

    assert result == 21
    assert len(timings) == 21
    assert first_run_ms == 100.0
    assert mean_ms == 10.0
    assert min_ms == 10.0
    assert max_ms == 10.0


def test_unsupported_ptx_error_is_backend_unavailable() -> None:
    """PTX バージョン不一致はバックエンド利用不可として扱う。"""
    message = (
        "[222] Call to cuLinkAddData results in "
        "CUDA_ERROR_UNSUPPORTED_PTX_VERSION Unsupported .version 8.8"
    )

    assert is_backend_unavailable_error(message)


def test_invalid_parameters_raise_value_error() -> None:
    """不正パラメータは ValueError として扱う。"""
    pointcloud = np.zeros((3, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        voxel_downsample_pointcloud(pointcloud, voxel_size=0.0)
    with pytest.raises(ValueError):
        detect_plane_ransac(pointcloud, 0.0, 100, 0.1, 0)
    with pytest.raises(ValueError):
        euclidean_cluster_pointcloud(pointcloud, 0.0, 1)

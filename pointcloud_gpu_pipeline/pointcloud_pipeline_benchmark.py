"""点群処理パイプラインのステージ別ベンチマークを提供する。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from PIL import Image

from pointcloud_cpu_algorithms import (
    detect_plane_ransac,
    euclidean_cluster_pointcloud,
    load_rgbd,
    voxel_downsample_pointcloud,
)
from pointcloud_cupy_pipeline import (
    create_pointcloud_from_rgbd_gpu,
    detect_plane_ransac_gpu,
    euclidean_cluster_labels_gpu,
    labels_to_clusters,
    synchronize_cupy,
    voxel_downsample_pointcloud_gpu,
)
from pointcloud_data_types import (
    CameraIntrinsics,
    PipelineParameters,
    PointcloudFileConfig,
    RgbdInput,
)
from pointcloud_generation_backends import (
    PointcloudGenerationBackend,
    default_generation_backends,
)
from pointcloud_gpu_backend_pipelines import (
    create_pointcloud_from_rgbd_numba_gpu,
    is_numba_cuda_pipeline_available,
    synchronize_numba_cuda_pipeline,
)
from pointcloud_open3d_pipeline import (
    create_pointcloud_open3d,
    create_pointcloud_open3d_cuda,
    detect_plane_ransac_open3d,
    detect_plane_ransac_open3d_cuda,
    euclidean_cluster_open3d,
    euclidean_cluster_open3d_cuda,
    is_open3d_available,
    is_open3d_cuda_available,
    open3d_cuda_labels_to_clusters,
    open3d_cuda_to_numpy,
    synchronize_open3d_cuda,
    voxel_downsample_open3d,
    voxel_downsample_open3d_cuda,
)

T = TypeVar("T")


@dataclass(frozen=True)
class PipelineBenchmarkConfig:
    """パイプラインベンチマーク設定を保持する。

    Args:
        rgb_path: RGB 画像のパス。
        depth_path: depth NPY のパス。
        output_path: JSON 結果の出力先。
        backend_names: 実行する点群生成バックエンド名。
        intrinsics: カメラ内部パラメータ。
        min_depth: 採用する最小 depth[m]。
        max_depth: 採用する最大 depth[m]。
        parameters: パイプラインパラメータ。
        runs: 総実行回数。
        warmup_runs: 統計から除外する先頭実行回数。
    """

    rgb_path: Path
    depth_path: Path
    output_path: Path
    backend_names: list[str]
    intrinsics: CameraIntrinsics
    min_depth: float
    max_depth: float
    parameters: PipelineParameters
    runs: int = 21
    warmup_runs: int = 1


@dataclass(frozen=True)
class StageBenchmarkResult:
    """ステージ単位のベンチマーク結果を保持する。

    Args:
        backend: バックエンド名。
        stage: 計測対象ステージ名。
        status: 実行状態。
        runs: 総実行回数。
        warmup_runs: 統計から除外した回数。
        measured_runs: 統計対象回数。
        first_run_ms: 1 回目の処理時間。
        mean_ms: 統計対象の平均処理時間。
        min_ms: 統計対象の最小処理時間。
        max_ms: 統計対象の最大処理時間。
        max_abs_error_xyz: NumPy 基準との xyz 最大絶対誤差。
        mean_abs_error_xyz: NumPy 基準との xyz 平均絶対誤差。
        count_mismatch: 点数またはクラスタ数の不一致。
        error: エラーメッセージ。
    """

    backend: str
    stage: str
    status: str
    runs: int
    warmup_runs: int
    measured_runs: int
    first_run_ms: float | None
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None
    max_abs_error_xyz: float | None
    mean_abs_error_xyz: float | None
    count_mismatch: int | None
    error: str | None


def select_generation_backends(names: list[str]) -> list[PointcloudGenerationBackend]:
    """名前で点群生成バックエンドを選択する。

    Args:
        names: 実行対象バックエンド名。

    Returns:
        選択したバックエンド一覧。

    Raises:
        ValueError: 不明なバックエンド名が指定された場合。
    """
    backend_map = {backend.name: backend for backend in default_generation_backends()}
    unknown = sorted(set(names) - set(backend_map))
    if unknown:
        msg = f"不明なバックエンドです: {', '.join(unknown)}"
        raise ValueError(msg)
    return [backend_map[name] for name in names]


def time_callable(
    func: Callable[[], T],
    runs: int,
    warmup_runs: int,
    synchronize: Callable[[], None] | None = None,
) -> tuple[T, list[float]]:
    """任意の処理を複数回計測する。

    Args:
        func: 計測対象処理。
        runs: 総実行回数。
        warmup_runs: 統計から除外する先頭実行回数。
        synchronize: GPU 同期関数。

    Returns:
        最後の処理結果と各回の処理時間[ms]。

    Raises:
        ValueError: 実行回数設定が不正な場合。
    """
    if runs <= 0:
        msg = "runs は 1 以上を指定してください。"
        raise ValueError(msg)
    if warmup_runs < 0 or warmup_runs >= runs:
        msg = "warmup_runs は 0 以上 runs 未満を指定してください。"
        raise ValueError(msg)

    timings: list[float] = []
    output: T | None = None
    for _ in range(runs):
        start = time.perf_counter()
        output = func()
        if synchronize is not None:
            synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    if output is None:
        msg = "計測結果が取得できませんでした。"
        raise RuntimeError(msg)
    return output, timings


def measured_statistics(
    timings: list[float], warmup_runs: int
) -> tuple[float, float, float, float]:
    """初回を含む計測結果から統計値を計算する。

    Args:
        timings: 各回の処理時間[ms]。
        warmup_runs: 統計から除外する先頭実行回数。

    Returns:
        first_run_ms、平均、最小、最大。
    """
    measured = np.asarray(timings[warmup_runs:], dtype=np.float64)
    return (
        float(timings[0]),
        float(np.mean(measured)),
        float(np.min(measured)),
        float(np.max(measured)),
    )


def compare_pointcloud_xyz(
    output: np.ndarray, reference: np.ndarray
) -> tuple[float | None, float | None, int]:
    """2 つの標準形式点群の xyz 差分を計算する。

    Args:
        output: 比較対象点群。
        reference: 基準点群。

    Returns:
        xyz 最大絶対誤差、xyz 平均絶対誤差、点数不一致数。
    """
    mismatch = abs(output.shape[0] - reference.shape[0])
    common_count = min(output.shape[0], reference.shape[0])
    if common_count == 0:
        return None, None, mismatch
    diff = np.abs(output[:common_count, :3] - reference[:common_count, :3])
    return float(np.max(diff)), float(np.mean(diff)), mismatch


def success_result(
    backend: str,
    stage: str,
    timings: list[float],
    warmup_runs: int,
    max_error: float | None,
    mean_error: float | None,
    count_mismatch: int | None,
) -> StageBenchmarkResult:
    """成功状態のベンチマーク結果を作成する。

    Args:
        backend: バックエンド名。
        stage: ステージ名。
        timings: 各回の処理時間[ms]。
        warmup_runs: 統計から除外した回数。
        max_error: xyz 最大絶対誤差。
        mean_error: xyz 平均絶対誤差。
        count_mismatch: 点数またはクラスタ数の不一致。

    Returns:
        成功状態の結果。
    """
    first_run_ms, mean_ms, min_ms, max_ms = measured_statistics(timings, warmup_runs)
    return StageBenchmarkResult(
        backend=backend,
        stage=stage,
        status="ok",
        runs=len(timings),
        warmup_runs=warmup_runs,
        measured_runs=len(timings) - warmup_runs,
        first_run_ms=first_run_ms,
        mean_ms=mean_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        max_abs_error_xyz=max_error,
        mean_abs_error_xyz=mean_error,
        count_mismatch=count_mismatch,
        error=None,
    )


def unavailable_result(
    backend: str,
    stage: str,
    config: PipelineBenchmarkConfig,
    reason: str | None = None,
) -> StageBenchmarkResult:
    """未利用バックエンドの結果を作成する。

    Args:
        backend: バックエンド名。
        stage: ステージ名。
        config: ベンチマーク設定。
        reason: 未利用と判定した理由。

    Returns:
        未利用状態の結果。
    """
    return StageBenchmarkResult(
        backend,
        stage,
        "unavailable",
        config.runs,
        config.warmup_runs,
        config.runs - config.warmup_runs,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        reason,
    )


def error_result(
    backend: str, stage: str, config: PipelineBenchmarkConfig, error: str
) -> StageBenchmarkResult:
    """失敗バックエンドの結果を作成する。

    Args:
        backend: バックエンド名。
        stage: ステージ名。
        config: ベンチマーク設定。
        error: エラーメッセージ。

    Returns:
        失敗状態の結果。
    """
    return StageBenchmarkResult(
        backend,
        stage,
        "error",
        config.runs,
        config.warmup_runs,
        config.runs - config.warmup_runs,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        error,
    )


def compact_error_message(message: str) -> str:
    """例外メッセージを JSON と標準出力向けに 1 行へ整形する。

    Args:
        message: 元の例外メッセージ。

    Returns:
        改行と連続空白を整理したメッセージ。
    """
    return " ".join(message.split())


def is_backend_unavailable_error(message: str) -> bool:
    """実行環境起因でバックエンド利用不可とみなすエラーか返す。

    Args:
        message: 例外メッセージ。

    Returns:
        バックエンド利用不可として扱う場合は True。
    """
    unavailable_markers = (
        "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
        "Unsupported .version",
        "Numba CUDA が利用できません",
        "CuPy がインストールされていません",
        "Open3D がインストールされていません",
    )
    return any(marker in message for marker in unavailable_markers)


def benchmark_backend(
    backend: PointcloudGenerationBackend,
    data: RgbdInput,
    reference_pointcloud: np.ndarray,
    config: PipelineBenchmarkConfig,
) -> list[StageBenchmarkResult]:
    """1 つの点群生成バックエンドをステージ別に計測する。

    Args:
        backend: 計測対象バックエンド。
        data: RGBD 入力。
        reference_pointcloud: NumPy CPU 基準点群。
        config: ベンチマーク設定。

    Returns:
        ステージ別ベンチマーク結果。
    """
    stages = [
        "pointcloud_generation",
        "voxel_downsampling",
        "plane_removal",
        "clustering",
        "pipeline_total",
    ]
    if not backend.is_available():
        return [unavailable_result(backend.name, stage, config) for stage in stages]

    results: list[StageBenchmarkResult] = []
    try:
        pointcloud, timings = time_callable(
            lambda: backend.create_pointcloud(data),
            config.runs,
            config.warmup_runs,
            backend.synchronize,
        )
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            pointcloud, reference_pointcloud
        )
        results.append(
            success_result(
                backend.name,
                "pointcloud_generation",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        downsampled, timings = time_callable(
            lambda: voxel_downsample_pointcloud(
                pointcloud, config.parameters.voxel_size
            ),
            config.runs,
            config.warmup_runs,
        )
        reference_downsampled = voxel_downsample_pointcloud(
            reference_pointcloud, config.parameters.voxel_size
        )
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            downsampled, reference_downsampled
        )
        results.append(
            success_result(
                backend.name,
                "voxel_downsampling",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        plane, timings = time_callable(
            lambda: detect_plane_ransac(
                downsampled,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
                config.parameters.random_seed,
            ),
            config.runs,
            config.warmup_runs,
        )
        reference_plane = detect_plane_ransac(
            reference_downsampled,
            config.parameters.plane_distance_threshold,
            config.parameters.plane_max_iterations,
            config.parameters.min_inlier_ratio,
            config.parameters.random_seed,
        )
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            plane.remaining_pointcloud, reference_plane.remaining_pointcloud
        )
        results.append(
            success_result(
                backend.name,
                "plane_removal",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        clusters, timings = time_callable(
            lambda: euclidean_cluster_pointcloud(
                plane.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            ),
            config.runs,
            config.warmup_runs,
        )
        reference_clusters = euclidean_cluster_pointcloud(
            reference_plane.remaining_pointcloud,
            config.parameters.cluster_tolerance,
            config.parameters.cluster_min_size,
            config.parameters.cluster_max_size,
        )
        results.append(
            success_result(
                backend.name,
                "clustering",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(len(clusters) - len(reference_clusters)),
            )
        )

        def run_pipeline_once() -> list[np.ndarray]:
            current = backend.create_pointcloud(data)
            current_downsampled = voxel_downsample_pointcloud(
                current, config.parameters.voxel_size
            )
            current_plane = detect_plane_ransac(
                current_downsampled,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
                config.parameters.random_seed,
            )
            return euclidean_cluster_pointcloud(
                current_plane.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            )

        _, timings = time_callable(
            run_pipeline_once, config.runs, config.warmup_runs, backend.synchronize
        )
        results.append(
            success_result(
                backend.name,
                "pipeline_total",
                timings,
                config.warmup_runs,
                None,
                None,
                None,
            )
        )
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        message = compact_error_message(str(exc))
        if is_backend_unavailable_error(message):
            return [
                unavailable_result(backend.name, stage, config, message)
                for stage in stages
            ]
        results.append(error_result(backend.name, "pipeline", config, message))
    return results


def benchmark_cupy_gpu_pipeline(
    data: RgbdInput,
    reference_pointcloud: np.ndarray,
    config: PipelineBenchmarkConfig,
) -> list[StageBenchmarkResult]:
    """CuPy GPU 常駐パイプラインをステージ別に計測する。

    Args:
        data: RGBD 入力。
        reference_pointcloud: NumPy CPU 基準点群。
        config: ベンチマーク設定。

    Returns:
        ステージ別ベンチマーク結果。
    """
    backend_name = "cupy_gpu"
    stages = [
        "pointcloud_generation",
        "voxel_downsampling",
        "plane_removal",
        "clustering",
        "pipeline_total",
    ]
    results: list[StageBenchmarkResult] = []
    try:
        pointcloud_gpu, timings = time_callable(
            lambda: create_pointcloud_from_rgbd_gpu(data),
            config.runs,
            config.warmup_runs,
            synchronize_cupy,
        )
        pointcloud = pointcloud_gpu.get()
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            pointcloud, reference_pointcloud
        )
        results.append(
            success_result(
                backend_name,
                "pointcloud_generation",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        downsampled_gpu, timings = time_callable(
            lambda: voxel_downsample_pointcloud_gpu(
                pointcloud_gpu, config.parameters.voxel_size
            ),
            config.runs,
            config.warmup_runs,
            synchronize_cupy,
        )
        downsampled = downsampled_gpu.get()
        reference_downsampled = voxel_downsample_pointcloud(
            reference_pointcloud, config.parameters.voxel_size
        )
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            downsampled, reference_downsampled
        )
        results.append(
            success_result(
                backend_name,
                "voxel_downsampling",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        plane_gpu, timings = time_callable(
            lambda: detect_plane_ransac_gpu(
                downsampled_gpu,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
                config.parameters.random_seed,
            ),
            config.runs,
            config.warmup_runs,
            synchronize_cupy,
        )
        remaining = plane_gpu.remaining_pointcloud.get()
        reference_plane = detect_plane_ransac(
            reference_downsampled,
            config.parameters.plane_distance_threshold,
            config.parameters.plane_max_iterations,
            config.parameters.min_inlier_ratio,
            config.parameters.random_seed,
        )
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            remaining, reference_plane.remaining_pointcloud
        )
        results.append(
            success_result(
                backend_name,
                "plane_removal",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        labels_gpu, timings = time_callable(
            lambda: euclidean_cluster_labels_gpu(
                plane_gpu.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            ),
            config.runs,
            config.warmup_runs,
            synchronize_cupy,
        )
        clusters = labels_to_clusters(labels_gpu)
        reference_clusters = euclidean_cluster_pointcloud(
            reference_plane.remaining_pointcloud,
            config.parameters.cluster_tolerance,
            config.parameters.cluster_min_size,
            config.parameters.cluster_max_size,
        )
        results.append(
            success_result(
                backend_name,
                "clustering",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(len(clusters) - len(reference_clusters)),
            )
        )

        def run_pipeline_once() -> object:
            current = create_pointcloud_from_rgbd_gpu(data)
            current_downsampled = voxel_downsample_pointcloud_gpu(
                current, config.parameters.voxel_size
            )
            current_plane = detect_plane_ransac_gpu(
                current_downsampled,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
                config.parameters.random_seed,
            )
            return euclidean_cluster_labels_gpu(
                current_plane.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            )

        _, timings = time_callable(
            run_pipeline_once, config.runs, config.warmup_runs, synchronize_cupy
        )
        results.append(
            success_result(
                backend_name,
                "pipeline_total",
                timings,
                config.warmup_runs,
                None,
                None,
                None,
            )
        )
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        message = compact_error_message(str(exc))
        if is_backend_unavailable_error(message):
            return [
                unavailable_result(backend_name, stage, config, message)
                for stage in stages
            ]
        results.append(error_result(backend_name, "pipeline", config, message))
    return results


def benchmark_gpu_resident_pipeline(
    backend_name: str,
    create_pointcloud_gpu: Callable[[RgbdInput], Any],
    is_available: Callable[[], bool],
    synchronize: Callable[[], None],
    data: RgbdInput,
    reference_pointcloud: np.ndarray,
    config: PipelineBenchmarkConfig,
) -> list[StageBenchmarkResult]:
    """GPU 常駐パイプラインをステージ別に計測する。

    Args:
        backend_name: バックエンド名。
        create_pointcloud_gpu: RGBD から GPU 点群を作成する関数。
        is_available: バックエンドの利用可否を返す関数。
        synchronize: GPU 同期関数。
        data: RGBD 入力。
        reference_pointcloud: NumPy CPU 基準点群。
        config: ベンチマーク設定。

    Returns:
        ステージ別ベンチマーク結果。
    """
    stages = [
        "pointcloud_generation",
        "voxel_downsampling",
        "plane_removal",
        "clustering",
        "pipeline_total",
    ]
    if not is_available():
        return [unavailable_result(backend_name, stage, config) for stage in stages]

    results: list[StageBenchmarkResult] = []
    try:
        pointcloud_gpu, timings = time_callable(
            lambda: create_pointcloud_gpu(data),
            config.runs,
            config.warmup_runs,
            synchronize,
        )
        pointcloud = pointcloud_gpu.get()
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            pointcloud, reference_pointcloud
        )
        results.append(
            success_result(
                backend_name,
                "pointcloud_generation",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        downsampled_gpu, timings = time_callable(
            lambda: voxel_downsample_pointcloud_gpu(
                pointcloud_gpu, config.parameters.voxel_size
            ),
            config.runs,
            config.warmup_runs,
            synchronize,
        )
        downsampled = downsampled_gpu.get()
        reference_downsampled = voxel_downsample_pointcloud(
            reference_pointcloud, config.parameters.voxel_size
        )
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            downsampled, reference_downsampled
        )
        results.append(
            success_result(
                backend_name,
                "voxel_downsampling",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        plane_gpu, timings = time_callable(
            lambda: detect_plane_ransac_gpu(
                downsampled_gpu,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
                config.parameters.random_seed,
            ),
            config.runs,
            config.warmup_runs,
            synchronize,
        )
        remaining = plane_gpu.remaining_pointcloud.get()
        reference_plane = detect_plane_ransac(
            reference_downsampled,
            config.parameters.plane_distance_threshold,
            config.parameters.plane_max_iterations,
            config.parameters.min_inlier_ratio,
            config.parameters.random_seed,
        )
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            remaining, reference_plane.remaining_pointcloud
        )
        results.append(
            success_result(
                backend_name,
                "plane_removal",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        labels_gpu, timings = time_callable(
            lambda: euclidean_cluster_labels_gpu(
                plane_gpu.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            ),
            config.runs,
            config.warmup_runs,
            synchronize,
        )
        clusters = labels_to_clusters(labels_gpu)
        reference_clusters = euclidean_cluster_pointcloud(
            reference_plane.remaining_pointcloud,
            config.parameters.cluster_tolerance,
            config.parameters.cluster_min_size,
            config.parameters.cluster_max_size,
        )
        results.append(
            success_result(
                backend_name,
                "clustering",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(len(clusters) - len(reference_clusters)),
            )
        )

        def run_pipeline_once() -> object:
            current = create_pointcloud_gpu(data)
            current_downsampled = voxel_downsample_pointcloud_gpu(
                current, config.parameters.voxel_size
            )
            current_plane = detect_plane_ransac_gpu(
                current_downsampled,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
                config.parameters.random_seed,
            )
            return euclidean_cluster_labels_gpu(
                current_plane.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            )

        _, timings = time_callable(
            run_pipeline_once, config.runs, config.warmup_runs, synchronize
        )
        results.append(
            success_result(
                backend_name,
                "pipeline_total",
                timings,
                config.warmup_runs,
                None,
                None,
                None,
            )
        )
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        message = compact_error_message(str(exc))
        if is_backend_unavailable_error(message):
            return [
                unavailable_result(backend_name, stage, config, message)
                for stage in stages
            ]
        results.append(error_result(backend_name, "pipeline", config, message))
    return results


def benchmark_open3d_cuda_pipeline(
    data: RgbdInput,
    reference_pointcloud: np.ndarray,
    config: PipelineBenchmarkConfig,
) -> list[StageBenchmarkResult]:
    """Open3D CUDA tensor パイプラインをステージ別に計測する。

    Args:
        data: RGBD 入力。
        reference_pointcloud: NumPy CPU 基準点群。
        config: ベンチマーク設定。

    Returns:
        ステージ別ベンチマーク結果。
    """
    backend_name = "open3d_cuda"
    stages = [
        "pointcloud_generation",
        "voxel_downsampling",
        "plane_removal",
        "clustering",
        "pipeline_total",
    ]
    if not is_open3d_cuda_available():
        return [unavailable_result(backend_name, stage, config) for stage in stages]

    results: list[StageBenchmarkResult] = []
    try:
        pointcloud_gpu, timings = time_callable(
            lambda: create_pointcloud_open3d_cuda(data),
            config.runs,
            config.warmup_runs,
            synchronize_open3d_cuda,
        )
        pointcloud = open3d_cuda_to_numpy(pointcloud_gpu)
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            pointcloud, reference_pointcloud
        )
        results.append(
            success_result(
                backend_name,
                "pointcloud_generation",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        downsampled_gpu, timings = time_callable(
            lambda: voxel_downsample_open3d_cuda(
                pointcloud_gpu, config.parameters.voxel_size
            ),
            config.runs,
            config.warmup_runs,
            synchronize_open3d_cuda,
        )
        downsampled = open3d_cuda_to_numpy(downsampled_gpu)
        reference_downsampled = voxel_downsample_pointcloud(
            reference_pointcloud, config.parameters.voxel_size
        )
        results.append(
            success_result(
                backend_name,
                "voxel_downsampling",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(downsampled.shape[0] - reference_downsampled.shape[0]),
            )
        )

        plane_gpu, timings = time_callable(
            lambda: detect_plane_ransac_open3d_cuda(
                downsampled_gpu,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
            ),
            config.runs,
            config.warmup_runs,
            synchronize_open3d_cuda,
        )
        remaining = open3d_cuda_to_numpy(plane_gpu.remaining_pointcloud)
        reference_plane = detect_plane_ransac(
            reference_downsampled,
            config.parameters.plane_distance_threshold,
            config.parameters.plane_max_iterations,
            config.parameters.min_inlier_ratio,
            config.parameters.random_seed,
        )
        results.append(
            success_result(
                backend_name,
                "plane_removal",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(remaining.shape[0] - reference_plane.remaining_pointcloud.shape[0]),
            )
        )

        labels_gpu, timings = time_callable(
            lambda: euclidean_cluster_open3d_cuda(
                plane_gpu.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            ),
            config.runs,
            config.warmup_runs,
            synchronize_open3d_cuda,
        )
        clusters = open3d_cuda_labels_to_clusters(labels_gpu)
        reference_clusters = euclidean_cluster_pointcloud(
            reference_plane.remaining_pointcloud,
            config.parameters.cluster_tolerance,
            config.parameters.cluster_min_size,
            config.parameters.cluster_max_size,
        )
        results.append(
            success_result(
                backend_name,
                "clustering",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(len(clusters) - len(reference_clusters)),
            )
        )

        def run_pipeline_once() -> Any:
            current = create_pointcloud_open3d_cuda(data)
            current_downsampled = voxel_downsample_open3d_cuda(
                current, config.parameters.voxel_size
            )
            current_plane = detect_plane_ransac_open3d_cuda(
                current_downsampled,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
            )
            return euclidean_cluster_open3d_cuda(
                current_plane.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            )

        _, timings = time_callable(
            run_pipeline_once,
            config.runs,
            config.warmup_runs,
            synchronize_open3d_cuda,
        )
        results.append(
            success_result(
                backend_name,
                "pipeline_total",
                timings,
                config.warmup_runs,
                None,
                None,
                None,
            )
        )
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        message = compact_error_message(str(exc))
        if is_backend_unavailable_error(message):
            return [
                unavailable_result(backend_name, stage, config, message)
                for stage in stages
            ]
        results.append(error_result(backend_name, "pipeline", config, message))
    return results


def benchmark_open3d_pipeline(
    data: RgbdInput,
    reference_pointcloud: np.ndarray,
    config: PipelineBenchmarkConfig,
) -> list[StageBenchmarkResult]:
    """Open3D CPU パイプラインをステージ別に計測する。

    Args:
        data: RGBD 入力。
        reference_pointcloud: NumPy CPU 基準点群。
        config: ベンチマーク設定。

    Returns:
        ステージ別ベンチマーク結果。
    """
    backend_name = "open3d_cpu"
    stages = [
        "pointcloud_generation",
        "voxel_downsampling",
        "plane_removal",
        "clustering",
        "pipeline_total",
    ]
    if not is_open3d_available():
        return [unavailable_result(backend_name, stage, config) for stage in stages]

    results: list[StageBenchmarkResult] = []
    try:
        pointcloud, timings = time_callable(
            lambda: create_pointcloud_open3d(data),
            config.runs,
            config.warmup_runs,
        )
        max_error, mean_error, mismatch = compare_pointcloud_xyz(
            pointcloud, reference_pointcloud
        )
        results.append(
            success_result(
                backend_name,
                "pointcloud_generation",
                timings,
                config.warmup_runs,
                max_error,
                mean_error,
                mismatch,
            )
        )

        downsampled, timings = time_callable(
            lambda: voxel_downsample_open3d(pointcloud, config.parameters.voxel_size),
            config.runs,
            config.warmup_runs,
        )
        reference_downsampled = voxel_downsample_pointcloud(
            reference_pointcloud, config.parameters.voxel_size
        )
        results.append(
            success_result(
                backend_name,
                "voxel_downsampling",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(downsampled.shape[0] - reference_downsampled.shape[0]),
            )
        )

        plane, timings = time_callable(
            lambda: detect_plane_ransac_open3d(
                downsampled,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
            ),
            config.runs,
            config.warmup_runs,
        )
        reference_plane = detect_plane_ransac(
            reference_downsampled,
            config.parameters.plane_distance_threshold,
            config.parameters.plane_max_iterations,
            config.parameters.min_inlier_ratio,
            config.parameters.random_seed,
        )
        results.append(
            success_result(
                backend_name,
                "plane_removal",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(
                    plane.remaining_pointcloud.shape[0]
                    - reference_plane.remaining_pointcloud.shape[0]
                ),
            )
        )

        clusters, timings = time_callable(
            lambda: euclidean_cluster_open3d(
                plane.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            ),
            config.runs,
            config.warmup_runs,
        )
        reference_clusters = euclidean_cluster_pointcloud(
            reference_plane.remaining_pointcloud,
            config.parameters.cluster_tolerance,
            config.parameters.cluster_min_size,
            config.parameters.cluster_max_size,
        )
        results.append(
            success_result(
                backend_name,
                "clustering",
                timings,
                config.warmup_runs,
                None,
                None,
                abs(len(clusters) - len(reference_clusters)),
            )
        )

        def run_pipeline_once() -> list[np.ndarray]:
            current = create_pointcloud_open3d(data)
            current_downsampled = voxel_downsample_open3d(
                current, config.parameters.voxel_size
            )
            current_plane = detect_plane_ransac_open3d(
                current_downsampled,
                config.parameters.plane_distance_threshold,
                config.parameters.plane_max_iterations,
                config.parameters.min_inlier_ratio,
            )
            return euclidean_cluster_open3d(
                current_plane.remaining_pointcloud,
                config.parameters.cluster_tolerance,
                config.parameters.cluster_min_size,
                config.parameters.cluster_max_size,
            )

        _, timings = time_callable(run_pipeline_once, config.runs, config.warmup_runs)
        results.append(
            success_result(
                backend_name,
                "pipeline_total",
                timings,
                config.warmup_runs,
                None,
                None,
                None,
            )
        )
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        message = compact_error_message(str(exc))
        if is_backend_unavailable_error(message):
            return [
                unavailable_result(backend_name, stage, config, message)
                for stage in stages
            ]
        results.append(error_result(backend_name, "pipeline", config, message))
    return results


def benchmark_pipeline(config: PipelineBenchmarkConfig) -> list[StageBenchmarkResult]:
    """選択したバックエンドで点群処理パイプラインを比較する。

    Args:
        config: ベンチマーク設定。

    Returns:
        ステージ別ベンチマーク結果。
    """
    file_config = PointcloudFileConfig(
        rgb_path=config.rgb_path,
        depth_path=config.depth_path,
        output_path=Path("unused.npy"),
        ply_path=None,
        intrinsics=config.intrinsics,
        depth_scale=1000.0,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
        allow_8bit_depth=False,
    )
    data = load_rgbd(file_config)
    reference_backend = select_generation_backends(["numpy_cpu"])[0]
    reference_pointcloud = reference_backend.create_pointcloud(data)
    results: list[StageBenchmarkResult] = []
    for backend in select_generation_backends(config.backend_names):
        if backend.name == "cupy_gpu":
            results.extend(
                benchmark_cupy_gpu_pipeline(data, reference_pointcloud, config)
            )
            continue
        if backend.name == "numba_cuda":
            results.extend(
                benchmark_gpu_resident_pipeline(
                    backend.name,
                    create_pointcloud_from_rgbd_numba_gpu,
                    is_numba_cuda_pipeline_available,
                    synchronize_numba_cuda_pipeline,
                    data,
                    reference_pointcloud,
                    config,
                )
            )
            continue
        if backend.name == "open3d_cpu":
            results.extend(
                benchmark_open3d_pipeline(data, reference_pointcloud, config)
            )
            continue
        if backend.name == "open3d_cuda":
            results.extend(
                benchmark_open3d_cuda_pipeline(data, reference_pointcloud, config)
            )
            continue
        results.extend(benchmark_backend(backend, data, reference_pointcloud, config))
    return results


def save_benchmark_results(
    path: Path,
    config: PipelineBenchmarkConfig,
    results: list[StageBenchmarkResult],
) -> None:
    """ベンチマーク結果を JSON 保存する。

    Args:
        path: JSON 出力先。
        config: ベンチマーク設定。
        results: ステージ別結果。
    """
    payload = {
        "rgb": str(config.rgb_path),
        "depth": str(config.depth_path),
        "intrinsics": asdict(config.intrinsics),
        "parameters": asdict(config.parameters),
        "runs": config.runs,
        "warmup_runs": config.warmup_runs,
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_benchmark_results(results: list[StageBenchmarkResult]) -> None:
    """ベンチマーク結果を標準出力へ表示する。

    Args:
        results: ステージ別結果。
    """
    for result in results:
        if result.status != "ok":
            print(
                f"{result.backend}/{result.stage}: {result.status} {result.error or ''}".strip(),
                flush=True,
            )
            continue
        print(
            f"{result.backend}/{result.stage}: first={result.first_run_ms:.3f} ms "
            f"mean={result.mean_ms:.3f} ms min={result.min_ms:.3f} ms "
            f"max={result.max_ms:.3f} ms mismatch={result.count_mismatch}",
            flush=True,
        )


def default_config_from_image(
    rgb_path: Path,
    depth_path: Path,
    output_path: Path,
    fx: float,
    fy: float,
    cx: float | None,
    cy: float | None,
    min_depth: float,
    max_depth: float,
    backend_names: list[str],
    parameters: PipelineParameters,
    runs: int,
    warmup_runs: int,
) -> PipelineBenchmarkConfig:
    """画像サイズから主点を補完してベンチマーク設定を作成する。

    Args:
        rgb_path: RGB 画像のパス。
        depth_path: depth NPY のパス。
        output_path: JSON 出力先。
        fx: 焦点距離 fx[pixel]。
        fy: 焦点距離 fy[pixel]。
        cx: 主点 cx[pixel]。
        cy: 主点 cy[pixel]。
        min_depth: 採用する最小 depth[m]。
        max_depth: 採用する最大 depth[m]。
        backend_names: バックエンド名一覧。
        parameters: パイプラインパラメータ。
        runs: 総実行回数。
        warmup_runs: 統計除外回数。

    Returns:
        ベンチマーク設定。
    """
    rgb_size = Image.open(rgb_path).size
    resolved_cx = cx if cx is not None else (rgb_size[0] - 1) / 2.0
    resolved_cy = cy if cy is not None else (rgb_size[1] - 1) / 2.0
    return PipelineBenchmarkConfig(
        rgb_path=rgb_path,
        depth_path=depth_path,
        output_path=output_path,
        backend_names=backend_names,
        intrinsics=CameraIntrinsics(fx, fy, resolved_cx, resolved_cy),
        min_depth=min_depth,
        max_depth=max_depth,
        parameters=parameters,
        runs=runs,
        warmup_runs=warmup_runs,
    )

"""点群処理パイプラインの速度と精度をステージ別に比較する CLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pointcloud_data_types import PipelineParameters
from pointcloud_pipeline_benchmark import (
    PipelineBenchmarkConfig,
    benchmark_pipeline,
    default_config_from_image,
    print_benchmark_results,
    save_benchmark_results,
)


def parse_args() -> PipelineBenchmarkConfig:
    """CLI 引数を解析する。

    Returns:
        解析済みのベンチマーク設定。
    """
    parser = argparse.ArgumentParser(
        description="RGBD 点群処理パイプラインをステージ別・全体で比較します。"
    )
    parser.add_argument("--rgb", type=Path, required=True, help="RGB 画像のパス。")
    parser.add_argument("--depth", type=Path, required=True, help="depth NPY のパス。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark_pointcloud_pipeline.json"),
        help="ベンチマーク結果 JSON の出力先。",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["numpy_cpu", "cupy_gpu", "numba_cuda", "open3d_cuda"],
        help="実行する点群生成バックエンド名。",
    )
    parser.add_argument("--fx", type=float, required=True, help="焦点距離 fx[pixel]。")
    parser.add_argument("--fy", type=float, required=True, help="焦点距離 fy[pixel]。")
    parser.add_argument("--cx", type=float, default=None, help="主点 cx[pixel]。")
    parser.add_argument("--cy", type=float, default=None, help="主点 cy[pixel]。")
    parser.add_argument(
        "--min-depth", type=float, default=0.0, help="採用する最小 depth[m]。"
    )
    parser.add_argument(
        "--max-depth", type=float, default=np.inf, help="採用する最大 depth[m]。"
    )
    parser.add_argument("--runs", type=int, default=21, help="総実行回数。")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="統計から除外する先頭実行回数。",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.01, help="VoxelGrid の一辺の長さ[m]。"
    )
    parser.add_argument(
        "--plane-distance-threshold",
        type=float,
        default=0.01,
        help="RANSAC 平面 inlier の距離しきい値[m]。",
    )
    parser.add_argument(
        "--plane-max-iterations",
        type=int,
        default=1000,
        help="RANSAC の最大反復回数。",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.1,
        help="平面として採用する最小 inlier 比率。",
    )
    parser.add_argument(
        "--cluster-tolerance",
        type=float,
        default=0.03,
        help="Euclidean clustering の距離しきい値[m]。",
    )
    parser.add_argument(
        "--cluster-min-size",
        type=int,
        default=30,
        help="出力するクラスタの最小点数。",
    )
    parser.add_argument(
        "--cluster-max-size",
        type=int,
        default=None,
        help="出力するクラスタの最大点数。未指定時は上限なし。",
    )
    parser.add_argument(
        "--random-seed", type=int, default=0, help="RANSAC の乱数 seed。"
    )
    args = parser.parse_args()
    validate_args(args)
    parameters = PipelineParameters(
        voxel_size=args.voxel_size,
        plane_distance_threshold=args.plane_distance_threshold,
        plane_max_iterations=args.plane_max_iterations,
        min_inlier_ratio=args.min_inlier_ratio,
        cluster_tolerance=args.cluster_tolerance,
        cluster_min_size=args.cluster_min_size,
        cluster_max_size=args.cluster_max_size,
        random_seed=args.random_seed,
    )
    return default_config_from_image(
        rgb_path=args.rgb,
        depth_path=args.depth,
        output_path=args.output,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        backend_names=args.backends,
        parameters=parameters,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
    )


def validate_args(args: argparse.Namespace) -> None:
    """CLI 引数を検証する。

    Args:
        args: argparse で解析した引数。

    Raises:
        ValueError: 引数の値が不正な場合。
    """
    if args.fx <= 0.0 or args.fy <= 0.0:
        msg = "--fx と --fy は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if args.cx is not None and args.cx < 0.0:
        msg = "--cx は 0 以上を指定してください。"
        raise ValueError(msg)
    if args.cy is not None and args.cy < 0.0:
        msg = "--cy は 0 以上を指定してください。"
        raise ValueError(msg)
    if args.min_depth < 0.0 or args.max_depth <= args.min_depth:
        msg = "depth 範囲は 0 <= min < max で指定してください。"
        raise ValueError(msg)
    if args.runs <= 0 or args.warmup_runs < 0 or args.warmup_runs >= args.runs:
        msg = "--runs は 1 以上、--warmup-runs は 0 以上 runs 未満を指定してください。"
        raise ValueError(msg)
    if args.voxel_size <= 0.0:
        msg = "--voxel-size は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if args.plane_distance_threshold <= 0.0 or args.plane_max_iterations <= 0:
        msg = "平面検出パラメータが不正です。"
        raise ValueError(msg)
    if not 0.0 <= args.min_inlier_ratio <= 1.0:
        msg = "--min-inlier-ratio は 0 以上 1 以下を指定してください。"
        raise ValueError(msg)
    if args.cluster_tolerance <= 0.0 or args.cluster_min_size <= 0:
        msg = "クラスタリングパラメータが不正です。"
        raise ValueError(msg)
    if (
        args.cluster_max_size is not None
        and args.cluster_max_size < args.cluster_min_size
    ):
        msg = "--cluster-max-size は --cluster-min-size 以上を指定してください。"
        raise ValueError(msg)


def main() -> None:
    """点群処理パイプラインをステージ別・全体で比較する。"""
    config = parse_args()
    results = benchmark_pipeline(config)
    save_benchmark_results(config.output_path, config, results)
    print_benchmark_results(results)
    print(f"saved_json: {config.output_path}", flush=True)


if __name__ == "__main__":
    main()

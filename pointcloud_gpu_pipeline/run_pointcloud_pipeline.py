"""RGBD 入力から点群処理パイプライン結果を保存する CLI。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from pointcloud_cpu_algorithms import (
    colorize_clusters,
    load_rgbd,
    run_pointcloud_pipeline,
    save_ply,
)
from pointcloud_data_types import (
    CameraIntrinsics,
    PipelineParameters,
    PointcloudFileConfig,
)


def parse_args() -> tuple[PointcloudFileConfig, PipelineParameters, argparse.Namespace]:
    """CLI 引数を解析する。

    Returns:
        入出力設定、パイプラインパラメータ、元の argparse 結果。
    """
    parser = argparse.ArgumentParser(
        description="RGBD から点群生成、VoxelGrid、RANSAC 平面削除、クラスタリングを実行します。"
    )
    parser.add_argument("--rgb", type=Path, required=True, help="RGB 画像のパス。")
    parser.add_argument("--depth", type=Path, required=True, help="depth NPY のパス。")
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
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="整数 depth 画像をメートルへ変換する除数。",
    )
    parser.add_argument(
        "--allow-8bit-depth",
        action="store_true",
        help="8bit depth PNG を depth-scale でメートル換算して読み込む。",
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
        "--plane-max-iterations", type=int, default=1000, help="RANSAC の最大反復回数。"
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
        "--cluster-min-size", type=int, default=30, help="出力するクラスタの最小点数。"
    )
    parser.add_argument(
        "--cluster-max-size",
        type=int,
        default=None,
        help="出力するクラスタの最大点数。",
    )
    parser.add_argument(
        "--random-seed", type=int, default=0, help="RANSAC の乱数 seed。"
    )
    parser.add_argument(
        "--pointcloud-output",
        type=Path,
        default=Path("outputs/pipeline_pointcloud.npy"),
        help="RGBD から生成した標準点群 NPY の出力先。",
    )
    parser.add_argument(
        "--downsampled-output",
        type=Path,
        default=Path("outputs/pipeline_downsampled.npy"),
        help="VoxelGrid 後点群 NPY の出力先。",
    )
    parser.add_argument(
        "--remaining-output",
        type=Path,
        default=Path("outputs/pipeline_plane_removed.npy"),
        help="平面除去後点群 NPY の出力先。",
    )
    parser.add_argument(
        "--clustered-output",
        type=Path,
        default=Path("outputs/pipeline_clustered.npy"),
        help="クラスタ色付き点群 NPY の出力先。",
    )
    parser.add_argument(
        "--clustered-ply",
        type=Path,
        default=Path("outputs/pipeline_clustered.ply"),
        help="クラスタ色付き点群 PLY の出力先。",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/pipeline_summary.json"),
        help="平面係数とクラスタ数を含む JSON の出力先。",
    )
    args = parser.parse_args()
    validate_args(args)
    rgb_size = Image.open(args.rgb).size
    cx = args.cx if args.cx is not None else (rgb_size[0] - 1) / 2.0
    cy = args.cy if args.cy is not None else (rgb_size[1] - 1) / 2.0
    file_config = PointcloudFileConfig(
        rgb_path=args.rgb,
        depth_path=args.depth,
        output_path=args.pointcloud_output,
        ply_path=None,
        intrinsics=CameraIntrinsics(args.fx, args.fy, cx, cy),
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        allow_8bit_depth=args.allow_8bit_depth,
    )
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
    return file_config, parameters, args


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
    if args.depth_scale <= 0.0:
        msg = "--depth-scale は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if args.min_depth < 0.0 or args.max_depth <= args.min_depth:
        msg = "depth 範囲は 0 <= min < max で指定してください。"
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


def save_npy(path: Path, pointcloud: np.ndarray) -> None:
    """点群 NPY を保存する。

    Args:
        path: 出力先パス。
        pointcloud: 保存する点群。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pointcloud)


def save_summary(
    path: Path,
    parameters: PipelineParameters,
    cluster_sizes: list[int],
    coefficients: np.ndarray,
) -> None:
    """パイプライン結果の要約 JSON を保存する。

    Args:
        path: JSON 出力先。
        parameters: 実行パラメータ。
        cluster_sizes: クラスタごとの点数。
        coefficients: 平面係数 `[a, b, c, d]`。
    """
    payload = {
        "parameters": asdict(parameters),
        "plane_coefficients": coefficients.tolist(),
        "cluster_count": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """点群処理パイプラインを実行してクラスタ色付き点群を保存する。"""
    file_config, parameters, args = parse_args()
    data = load_rgbd(file_config)
    result = run_pointcloud_pipeline(data, parameters)
    clustered = colorize_clusters(result.plane.remaining_pointcloud, result.clusters)

    save_npy(args.pointcloud_output, result.pointcloud)
    save_npy(args.downsampled_output, result.downsampled_pointcloud)
    save_npy(args.remaining_output, result.plane.remaining_pointcloud)
    save_npy(args.clustered_output, clustered)
    save_ply(args.clustered_ply, clustered)
    save_summary(
        args.summary_output,
        parameters,
        [int(cluster.size) for cluster in result.clusters],
        result.plane.coefficients,
    )

    print(f"points: {result.pointcloud.shape[0]}", flush=True)
    print(f"downsampled_points: {result.downsampled_pointcloud.shape[0]}", flush=True)
    print(f"remaining_points: {result.plane.remaining_pointcloud.shape[0]}", flush=True)
    print(f"clusters: {len(result.clusters)}", flush=True)
    print(f"saved_clustered_npy: {args.clustered_output}", flush=True)
    print(f"saved_clustered_ply: {args.clustered_ply}", flush=True)
    print(f"saved_summary: {args.summary_output}", flush=True)


if __name__ == "__main__":
    main()

"""RGBD 入力から標準形式の点群ファイルを作成する CLI。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pointcloud_cpu_algorithms import (
    create_pointcloud_from_rgbd,
    load_depth,
    load_rgb,
    save_ply,
)
from pointcloud_data_types import CameraIntrinsics, PointcloudFileConfig, RgbdInput


def parse_args() -> PointcloudFileConfig:
    """CLI 引数を解析する。

    Returns:
        解析済みの点群ファイル入出力設定。
    """
    parser = argparse.ArgumentParser(
        description="RGB 画像と depth 画像から Nx6 [x, y, z, r, g, b] 点群を作成します。"
    )
    parser.add_argument("--rgb", type=Path, required=True, help="RGB 画像のパス。")
    parser.add_argument(
        "--depth",
        type=Path,
        required=True,
        help="depth の .npy または 16bit PNG 画像のパス。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pointcloud.npy"),
        help="点群 NPY の出力先。Nx6 の [x, y, z, r, g, b] を保存する。",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="確認用 PLY の出力先。指定した場合だけ保存する。",
    )
    parser.add_argument("--fx", type=float, required=True, help="焦点距離 fx[pixel]。")
    parser.add_argument("--fy", type=float, required=True, help="焦点距離 fy[pixel]。")
    parser.add_argument("--cx", type=float, default=None, help="主点 cx[pixel]。")
    parser.add_argument("--cy", type=float, default=None, help="主点 cy[pixel]。")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="整数 depth 画像をメートルへ変換する除数。例: mm なら 1000。",
    )
    parser.add_argument(
        "--min-depth", type=float, default=0.0, help="採用する最小 depth[m]。"
    )
    parser.add_argument(
        "--max-depth", type=float, default=np.inf, help="採用する最大 depth[m]。"
    )
    parser.add_argument(
        "--allow-8bit-depth",
        action="store_true",
        help="8bit depth PNG を depth-scale でメートル換算して読み込む。",
    )
    args = parser.parse_args()
    rgb_size = Image.open(args.rgb).size
    cx = args.cx if args.cx is not None else (rgb_size[0] - 1) / 2.0
    cy = args.cy if args.cy is not None else (rgb_size[1] - 1) / 2.0
    validate_args(args, cx, cy)
    return PointcloudFileConfig(
        rgb_path=args.rgb,
        depth_path=args.depth,
        output_path=args.output,
        ply_path=args.ply,
        intrinsics=CameraIntrinsics(args.fx, args.fy, cx, cy),
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        allow_8bit_depth=args.allow_8bit_depth,
    )


def validate_args(args: argparse.Namespace, cx: float, cy: float) -> None:
    """CLI 引数の値を検証する。

    Args:
        args: argparse で解析した引数。
        cx: 使用する主点 cx。
        cy: 使用する主点 cy。

    Raises:
        ValueError: 引数の値が不正な場合。
    """
    if args.fx <= 0.0 or args.fy <= 0.0:
        msg = "--fx と --fy は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if cx < 0.0 or cy < 0.0:
        msg = "--cx と --cy は 0 以上を指定してください。"
        raise ValueError(msg)
    if args.depth_scale <= 0.0:
        msg = "--depth-scale は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if args.min_depth < 0.0:
        msg = "--min-depth は 0 以上を指定してください。"
        raise ValueError(msg)
    if args.max_depth <= args.min_depth:
        msg = "--max-depth は --min-depth より大きい値を指定してください。"
        raise ValueError(msg)


def main() -> None:
    """RGBD 画像を読み込み、標準形式点群を保存する。"""
    config = parse_args()
    rgb = load_rgb(config.rgb_path)
    depth = load_depth(config)
    data = RgbdInput(rgb, depth, config.intrinsics, config.min_depth, config.max_depth)

    start_time = time.perf_counter()
    pointcloud = create_pointcloud_from_rgbd(data)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.output_path, pointcloud)
    if config.ply_path is not None:
        save_ply(config.ply_path, pointcloud)

    print(f"points: {pointcloud.shape[0]}", flush=True)
    print(f"create_pointcloud_ms: {elapsed_ms:.3f}", flush=True)
    print(f"saved_npy: {config.output_path}", flush=True)
    if config.ply_path is not None:
        print(f"saved_ply: {config.ply_path}", flush=True)


if __name__ == "__main__":
    main()

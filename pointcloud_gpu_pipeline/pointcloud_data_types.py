"""点群処理パイプラインで共有するデータ型を定義する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """カメラ内部パラメータを保持する。

    Args:
        fx: x 方向の焦点距離[pixel]。
        fy: y 方向の焦点距離[pixel]。
        cx: 主点 x 座標[pixel]。
        cy: 主点 y 座標[pixel]。
    """

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RgbdInput:
    """RGBD 入力を保持する。

    Args:
        rgb: HxWx3 の RGB 配列。
        depth: HxW の depth 配列。単位はメートル。
        intrinsics: カメラ内部パラメータ。
        min_depth: 採用する最小 depth[m]。
        max_depth: 採用する最大 depth[m]。
    """

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    min_depth: float
    max_depth: float


@dataclass(frozen=True)
class PipelineParameters:
    """点群処理パイプラインの調整パラメータを保持する。

    Args:
        voxel_size: ボクセルグリッドの一辺の長さ[m]。
        plane_distance_threshold: 平面 inlier とみなす距離しきい値[m]。
        plane_max_iterations: RANSAC の最大反復回数。
        min_inlier_ratio: 平面として採用する最小 inlier 比率。
        cluster_tolerance: Euclidean clustering の距離しきい値[m]。
        cluster_min_size: 出力するクラスタの最小点数。
        cluster_max_size: 出力するクラスタの最大点数。None の場合は上限なし。
        random_seed: RANSAC の乱数 seed。
    """

    voxel_size: float = 0.01
    plane_distance_threshold: float = 0.01
    plane_max_iterations: int = 1000
    min_inlier_ratio: float = 0.1
    cluster_tolerance: float = 0.03
    cluster_min_size: int = 30
    cluster_max_size: int | None = None
    random_seed: int = 0


@dataclass(frozen=True)
class PlaneSegmentationResult:
    """RANSAC 平面検出の結果を保持する。

    Args:
        coefficients: 正規化済み平面係数 `[a, b, c, d]`。
        inlier_indices: 平面に属する点の index。
        remaining_pointcloud: 平面点を除外した点群。
    """

    coefficients: np.ndarray
    inlier_indices: np.ndarray
    remaining_pointcloud: np.ndarray


@dataclass(frozen=True)
class PointcloudPipelineResult:
    """点群処理パイプラインの各段階の出力を保持する。

    Args:
        pointcloud: RGBD から生成した点群。
        downsampled_pointcloud: ボクセルダウンサンプリング後の点群。
        plane: 平面検出結果。
        clusters: 平面除去後点群に対するクラスタ index 一覧。
    """

    pointcloud: np.ndarray
    downsampled_pointcloud: np.ndarray
    plane: PlaneSegmentationResult
    clusters: list[np.ndarray]


@dataclass(frozen=True)
class PointcloudFileConfig:
    """点群入出力ファイル設定を保持する。

    Args:
        rgb_path: RGB 画像のパス。
        depth_path: depth 配列または depth 画像のパス。
        output_path: 点群 NPY の出力先。
        ply_path: PLY の出力先。指定しない場合は保存しない。
        intrinsics: カメラ内部パラメータ。
        depth_scale: depth 画像をメートルへ変換する除数。
        min_depth: 採用する最小 depth[m]。
        max_depth: 採用する最大 depth[m]。
        allow_8bit_depth: 8bit depth PNG の読み込みを許可するかどうか。
    """

    rgb_path: Path
    depth_path: Path
    output_path: Path
    ply_path: Path | None
    intrinsics: CameraIntrinsics
    depth_scale: float
    min_depth: float
    max_depth: float
    allow_8bit_depth: bool

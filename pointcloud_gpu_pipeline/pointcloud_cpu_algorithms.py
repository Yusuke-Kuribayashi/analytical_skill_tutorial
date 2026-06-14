"""NumPy による点群処理パイプラインの基準実装を提供する。"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from pointcloud_data_types import (
    CameraIntrinsics,
    PipelineParameters,
    PlaneSegmentationResult,
    PointcloudFileConfig,
    PointcloudPipelineResult,
    RgbdInput,
)


def load_rgb(path: Path) -> np.ndarray:
    """RGB 画像を読み込む。

    Args:
        path: RGB 画像のパス。

    Returns:
        uint8 の RGB 配列。
    """
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_depth(config: PointcloudFileConfig) -> np.ndarray:
    """depth 配列または depth 画像をメートル単位で読み込む。

    Args:
        config: 点群ファイル入出力設定。

    Returns:
        float32 の depth 配列。単位はメートル。

    Raises:
        ValueError: 8bit depth PNG が許可されていない場合。
    """
    if config.depth_path.suffix == ".npy":
        return np.load(config.depth_path).astype(np.float32)

    depth = np.asarray(Image.open(config.depth_path))
    if depth.dtype == np.uint8 and not config.allow_8bit_depth:
        msg = (
            "8bit depth PNG は多くの場合、可視化用で実 depth ではありません。"
            "--allow-8bit-depth を指定するか、depth_*.npy を使用してください。"
        )
        raise ValueError(msg)
    return depth.astype(np.float32) / config.depth_scale


def load_rgbd(config: PointcloudFileConfig) -> RgbdInput:
    """RGBD 入力を読み込む。

    Args:
        config: 点群ファイル入出力設定。

    Returns:
        RGBD 入力。

    Raises:
        ValueError: RGB と depth の画像サイズが一致しない場合。
    """
    rgb = load_rgb(config.rgb_path)
    depth = load_depth(config)
    if rgb.shape[:2] != depth.shape:
        msg = f"RGB {rgb.shape[:2]} と depth {depth.shape} のサイズが一致しません。"
        raise ValueError(msg)
    return RgbdInput(rgb, depth, config.intrinsics, config.min_depth, config.max_depth)


def create_pointcloud_from_rgbd(data: RgbdInput) -> np.ndarray:
    """RGBD 配列から標準形式の点群を作成する。

    Args:
        data: RGBD 入力。

    Returns:
        Nx6 の `[x, y, z, r, g, b]` 点群。

    Raises:
        ValueError: RGB と depth の画像サイズが一致しない場合。
    """
    if data.rgb.shape[:2] != data.depth.shape:
        msg = f"RGB {data.rgb.shape[:2]} と depth {data.depth.shape} のサイズが一致しません。"
        raise ValueError(msg)

    height, width = data.depth.shape
    u_grid, v_grid = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    depth = data.depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > data.min_depth) & (depth < data.max_depth)
    z = depth[valid]
    x = (u_grid[valid] - data.intrinsics.cx) * z / data.intrinsics.fx
    y = (v_grid[valid] - data.intrinsics.cy) * z / data.intrinsics.fy
    colors = data.rgb[valid].astype(np.float32)
    xyz = np.column_stack((x, y, z)).astype(np.float32)
    return np.column_stack((xyz, colors)).astype(np.float32)


def create_dense_pointcloud_from_rgbd(data: RgbdInput) -> np.ndarray:
    """RGBD 配列から dense 形式の点群を作成する。

    Args:
        data: RGBD 入力。

    Returns:
        HxWx6 の `[x, y, z, r, g, b]` 点群。無効点の xyz は NaN。

    Raises:
        ValueError: RGB と depth の画像サイズが一致しない場合。
    """
    if data.rgb.shape[:2] != data.depth.shape:
        msg = f"RGB {data.rgb.shape[:2]} と depth {data.depth.shape} のサイズが一致しません。"
        raise ValueError(msg)

    height, width = data.depth.shape
    u_grid, v_grid = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    depth = data.depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > data.min_depth) & (depth < data.max_depth)
    output = np.empty((height, width, 6), dtype=np.float32)
    output[..., :3] = np.nan
    output[..., 3:6] = data.rgb.astype(np.float32)
    output[..., 0] = np.where(
        valid, (u_grid - data.intrinsics.cx) * depth / data.intrinsics.fx, np.nan
    )
    output[..., 1] = np.where(
        valid, (v_grid - data.intrinsics.cy) * depth / data.intrinsics.fy, np.nan
    )
    output[..., 2] = np.where(valid, depth, np.nan)
    return output


def dense_to_standard_pointcloud(dense_pointcloud: np.ndarray) -> np.ndarray:
    """dense 点群から有効点だけを標準形式へ変換する。

    Args:
        dense_pointcloud: HxWx6 の dense 点群。

    Returns:
        Nx6 の `[x, y, z, r, g, b]` 点群。
    """
    valid = np.isfinite(dense_pointcloud[..., 2])
    return dense_pointcloud[valid].astype(np.float32, copy=False)


def validate_pointcloud(pointcloud: np.ndarray) -> None:
    """標準形式の点群であることを検証する。

    Args:
        pointcloud: 検証対象点群。

    Raises:
        ValueError: shape が Nx6 ではない場合。
    """
    if pointcloud.ndim != 2 or pointcloud.shape[1] != 6:
        msg = "点群は Nx6 の [x, y, z, r, g, b] 形式で指定してください。"
        raise ValueError(msg)


def voxel_downsample_pointcloud(
    pointcloud: np.ndarray, voxel_size: float
) -> np.ndarray:
    """ボクセルグリッド方式で点群をダウンサンプリングする。

    Args:
        pointcloud: Nx6 の `[x, y, z, r, g, b]` 点群。
        voxel_size: ボクセルの一辺の長さ[m]。

    Returns:
        ダウンサンプリング後の Nx6 点群。

    Raises:
        ValueError: 点群形式または voxel_size が不正な場合。
    """
    validate_pointcloud(pointcloud)
    if voxel_size <= 0.0:
        msg = "voxel_size は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if pointcloud.shape[0] == 0:
        return pointcloud.astype(np.float32, copy=True)

    voxel_indices = np.floor(pointcloud[:, :3] / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    sums = np.zeros((unique_voxels.shape[0], 6), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    np.add.at(sums, inverse, pointcloud.astype(np.float64, copy=False))
    return (sums / counts[:, None]).astype(np.float32)


def fit_plane_from_points(points_xyz: np.ndarray) -> np.ndarray:
    """3 点から平面係数を計算する。

    Args:
        points_xyz: 3x3 の xyz 点群。

    Returns:
        正規化済み平面係数 `[a, b, c, d]`。退化時は NaN を含む。
    """
    first, second, third = points_xyz
    normal = np.cross(second - first, third - first)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return np.full(4, np.nan, dtype=np.float32)
    normal = normal / norm
    distance = -float(np.dot(normal, first))
    return np.array([normal[0], normal[1], normal[2], distance], dtype=np.float32)


def point_to_plane_distances(
    points_xyz: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """点群から平面までの距離を計算する。

    Args:
        points_xyz: Nx3 の xyz 点群。
        coefficients: 平面係数 `[a, b, c, d]`。

    Returns:
        各点の平面距離。
    """
    normal_norm = float(np.linalg.norm(coefficients[:3]))
    if normal_norm == 0.0 or not np.isfinite(normal_norm):
        return np.full(points_xyz.shape[0], np.inf, dtype=np.float32)
    return np.abs(points_xyz @ coefficients[:3] + coefficients[3]) / normal_norm


def detect_plane_ransac(
    pointcloud: np.ndarray,
    distance_threshold: float,
    max_iterations: int,
    min_inlier_ratio: float,
    random_seed: int,
) -> PlaneSegmentationResult:
    """RANSAC で最大平面を検出して削除する。

    Args:
        pointcloud: Nx6 の `[x, y, z, r, g, b]` 点群。
        distance_threshold: 平面 inlier とみなす距離しきい値[m]。
        max_iterations: RANSAC の最大反復回数。
        min_inlier_ratio: 平面として採用する最小 inlier 比率。
        random_seed: 乱数 seed。

    Returns:
        平面係数、inlier index、平面削除後点群。

    Raises:
        ValueError: 点群またはパラメータが不正な場合。
    """
    validate_pointcloud(pointcloud)
    if pointcloud.shape[0] < 3:
        msg = "平面推定には 3 点以上の点群が必要です。"
        raise ValueError(msg)
    if distance_threshold <= 0.0:
        msg = "distance_threshold は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if max_iterations <= 0:
        msg = "max_iterations は 1 以上を指定してください。"
        raise ValueError(msg)
    if not 0.0 <= min_inlier_ratio <= 1.0:
        msg = "min_inlier_ratio は 0 以上 1 以下を指定してください。"
        raise ValueError(msg)

    rng = np.random.default_rng(random_seed)
    xyz = pointcloud[:, :3].astype(np.float64, copy=False)
    best_coefficients: np.ndarray | None = None
    best_inliers = np.array([], dtype=np.int64)

    for _ in range(max_iterations):
        sample_indices = rng.choice(pointcloud.shape[0], size=3, replace=False)
        coefficients = fit_plane_from_points(xyz[sample_indices])
        if not np.all(np.isfinite(coefficients)):
            continue
        distances = point_to_plane_distances(xyz, coefficients)
        inliers = np.flatnonzero(distances <= distance_threshold)
        if inliers.shape[0] > best_inliers.shape[0]:
            best_coefficients = coefficients
            best_inliers = inliers.astype(np.int64)

    min_inliers = int(np.ceil(pointcloud.shape[0] * min_inlier_ratio))
    if best_coefficients is None or best_inliers.shape[0] < max(3, min_inliers):
        coefficients = np.full(4, np.nan, dtype=np.float32)
        return PlaneSegmentationResult(
            coefficients=coefficients,
            inlier_indices=np.array([], dtype=np.int64),
            remaining_pointcloud=pointcloud.astype(np.float32, copy=True),
        )

    mask = np.ones(pointcloud.shape[0], dtype=bool)
    mask[best_inliers] = False
    return PlaneSegmentationResult(
        coefficients=best_coefficients.astype(np.float32),
        inlier_indices=best_inliers,
        remaining_pointcloud=pointcloud[mask].astype(np.float32, copy=False),
    )


def euclidean_cluster_pointcloud(
    pointcloud: np.ndarray,
    cluster_tolerance: float,
    cluster_min_size: int,
    cluster_max_size: int | None = None,
) -> list[np.ndarray]:
    """Euclidean clustering で点群をクラスタ分割する。

    Args:
        pointcloud: Nx6 の `[x, y, z, r, g, b]` 点群。
        cluster_tolerance: 同一クラスタとみなす距離しきい値[m]。
        cluster_min_size: 出力するクラスタの最小点数。
        cluster_max_size: 出力するクラスタの最大点数。None の場合は上限なし。

    Returns:
        クラスタごとの点群 index 配列一覧。

    Raises:
        ValueError: 点群またはパラメータが不正な場合。
    """
    validate_pointcloud(pointcloud)
    if cluster_tolerance <= 0.0:
        msg = "cluster_tolerance は 0 より大きい値を指定してください。"
        raise ValueError(msg)
    if cluster_min_size <= 0:
        msg = "cluster_min_size は 1 以上を指定してください。"
        raise ValueError(msg)
    if cluster_max_size is not None and cluster_max_size < cluster_min_size:
        msg = "cluster_max_size は cluster_min_size 以上を指定してください。"
        raise ValueError(msg)

    point_count = pointcloud.shape[0]
    if point_count == 0:
        return []

    xyz = pointcloud[:, :3].astype(np.float32, copy=False)
    visited = np.zeros(point_count, dtype=bool)
    clusters: list[np.ndarray] = []
    tolerance_sq = cluster_tolerance * cluster_tolerance

    for start_index in range(point_count):
        if visited[start_index]:
            continue
        queue: deque[int] = deque([start_index])
        visited[start_index] = True
        cluster: list[int] = []
        while queue:
            current = queue.popleft()
            cluster.append(current)
            diff = xyz - xyz[current]
            neighbor_indices = np.flatnonzero(
                np.einsum("ij,ij->i", diff, diff) <= tolerance_sq
            )
            for neighbor in neighbor_indices:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(int(neighbor))
        if len(cluster) < cluster_min_size:
            continue
        if cluster_max_size is not None and len(cluster) > cluster_max_size:
            continue
        clusters.append(np.array(sorted(cluster), dtype=np.int64))
    return clusters


def colorize_clusters(
    pointcloud: np.ndarray,
    clusters: list[np.ndarray],
    noise_color: tuple[float, float, float] = (128.0, 128.0, 128.0),
) -> np.ndarray:
    """クラスタごとに固定色を割り当てた点群を作成する。

    Args:
        pointcloud: Nx6 の `[x, y, z, r, g, b]` 点群。
        clusters: クラスタごとの点群 index 配列一覧。
        noise_color: クラスタに属さない点へ割り当てる RGB 色。

    Returns:
        xyz は入力を維持し、RGB をクラスタ色へ置き換えた Nx6 点群。

    Raises:
        ValueError: 点群形式またはクラスタ index が不正な場合。
    """
    validate_pointcloud(pointcloud)
    colored = pointcloud.astype(np.float32, copy=True)
    colored[:, 3:6] = np.asarray(noise_color, dtype=np.float32)
    palette = np.array(
        [
            [230.0, 25.0, 75.0],
            [60.0, 180.0, 75.0],
            [0.0, 130.0, 200.0],
            [245.0, 130.0, 48.0],
            [145.0, 30.0, 180.0],
            [70.0, 240.0, 240.0],
            [240.0, 50.0, 230.0],
            [210.0, 245.0, 60.0],
            [250.0, 190.0, 190.0],
            [0.0, 128.0, 128.0],
        ],
        dtype=np.float32,
    )
    for cluster_id, indices in enumerate(clusters):
        if indices.size == 0:
            continue
        if np.any(indices < 0) or np.any(indices >= pointcloud.shape[0]):
            msg = "クラスタ index が点群の範囲外です。"
            raise ValueError(msg)
        colored[indices, 3:6] = palette[cluster_id % palette.shape[0]]
    return colored


def run_pointcloud_pipeline(
    data: RgbdInput,
    parameters: PipelineParameters,
) -> PointcloudPipelineResult:
    """RGBD からクラスタリングまでの CPU パイプラインを実行する。

    Args:
        data: RGBD 入力。
        parameters: パイプラインパラメータ。

    Returns:
        各ステージの出力を含むパイプライン結果。
    """
    pointcloud = create_pointcloud_from_rgbd(data)
    downsampled = voxel_downsample_pointcloud(pointcloud, parameters.voxel_size)
    plane = detect_plane_ransac(
        downsampled,
        parameters.plane_distance_threshold,
        parameters.plane_max_iterations,
        parameters.min_inlier_ratio,
        parameters.random_seed,
    )
    clusters = euclidean_cluster_pointcloud(
        plane.remaining_pointcloud,
        parameters.cluster_tolerance,
        parameters.cluster_min_size,
        parameters.cluster_max_size,
    )
    return PointcloudPipelineResult(pointcloud, downsampled, plane, clusters)


def save_ply(path: Path, pointcloud: np.ndarray) -> None:
    """点群を ASCII PLY として保存する。

    Args:
        path: PLY の出力先。
        pointcloud: Nx6 の `[x, y, z, r, g, b]` 点群。

    Raises:
        ValueError: 点群形式が不正な場合。
    """
    validate_pointcloud(pointcloud)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = pointcloud[:, :3]
    rgb = np.clip(pointcloud[:, 3:6], 0, 255).astype(np.uint8)

    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {pointcloud.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(xyz, rgb, strict=True):
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def default_intrinsics(
    width: int, height: int, fx: float, fy: float
) -> CameraIntrinsics:
    """画像サイズと焦点距離からデフォルト内部パラメータを作成する。

    Args:
        width: 画像幅[pixel]。
        height: 画像高さ[pixel]。
        fx: x 方向焦点距離[pixel]。
        fy: y 方向焦点距離[pixel]。

    Returns:
        主点を画像中心に置いたカメラ内部パラメータ。
    """
    return CameraIntrinsics(fx=fx, fy=fy, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0)

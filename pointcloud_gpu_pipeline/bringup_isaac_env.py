"""Isaac Sim でテーブル上の基本形状を RGBD 撮影する。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

MODE_PRODUCTION = "production"
MODE_DEVELOPMENT = "development"
RUN_MODE = MODE_PRODUCTION


@dataclass(frozen=True)
class AppConfig:
    """実行設定を保持する。

    Args:
        mode: 実行モード。production は撮影して終了し、development は GUI を維持する。
        frames: 撮影するフレーム数。
        output_dir: RGBD データの出力先。
        width: カメラ画像の幅。
        height: カメラ画像の高さ。
        warmup_steps: 撮影前にレンダリングを安定させるステップ数。
        headless: GUI を表示せずに実行するかどうか。
        save_depth_png: 深度の可視化 PNG も保存するかどうか。
    """

    mode: str
    frames: int
    output_dir: Path
    width: int
    height: int
    warmup_steps: int
    headless: bool
    save_depth_png: bool


def parse_args() -> AppConfig:
    """CLI 引数を解析する。

    Returns:
        解析済みの実行設定。
    """
    parser = argparse.ArgumentParser(
        description="Isaac Sim でテーブル上の形状を RGBD 撮影します。"
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_PRODUCTION, MODE_DEVELOPMENT),
        default=RUN_MODE,
        help="実行モード。production は撮影後に終了し、development は GUI で編集可能な状態を維持する。",
    )
    parser.add_argument(
        "--frames", "-n", type=int, default=10, help="撮影するフレーム数。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rgbd"),
        help="RGBD データの出力先。",
    )
    parser.add_argument("--width", type=int, default=1280, help="カメラ画像の幅。")
    parser.add_argument("--height", type=int, default=720, help="カメラ画像の高さ。")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=60,
        help="撮影前に進めるレンダリングステップ数。",
    )
    parser.add_argument(
        "--show", action="store_true", help="production でも GUI を表示して実行する。"
    )
    parser.add_argument(
        "--save-depth-png", action="store_true", help="深度の可視化 PNG も保存する。"
    )
    args = parser.parse_args()

    if args.frames <= 0:
        msg = "--frames は 1 以上を指定してください。"
        raise ValueError(msg)
    if args.width <= 0 or args.height <= 0:
        msg = "--width と --height は 1 以上を指定してください。"
        raise ValueError(msg)
    if args.warmup_steps < 0:
        msg = "--warmup-steps は 0 以上を指定してください。"
        raise ValueError(msg)

    return AppConfig(
        mode=args.mode,
        frames=args.frames,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        warmup_steps=args.warmup_steps,
        headless=args.mode == MODE_PRODUCTION and not args.show,
        save_depth_png=args.save_depth_png,
    )


def main() -> None:
    """Isaac Sim を起動して RGBD 撮影または開発用 GUI 表示を実行する。"""
    config = parse_args()
    simulation_app = SimulationApp({"headless": config.headless})

    try:
        run_scene(config, simulation_app)
    finally:
        simulation_app.close()


def run_scene(config: AppConfig, simulation_app: SimulationApp) -> None:
    """シーンを構築して実行モードに応じた処理を行う。

    Args:
        config: 実行設定。
        simulation_app: 起動済みの Isaac Sim アプリケーション。
    """
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCapsule, FixedCuboid, FixedSphere
    from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
    from isaacsim.sensors.camera import Camera
    from PIL import Image
    from pxr import UsdLux

    if config.mode == MODE_PRODUCTION:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_lights(world, UsdLux)

    table_height = 0.72
    tabletop_thickness = 0.06
    tabletop_z = table_height + tabletop_thickness / 2.0
    object_z_base = table_height + tabletop_thickness

    world.scene.add(
        FixedCuboid(
            prim_path="/World/TableTop",
            name="table_top",
            position=np.array([0.0, 0.0, tabletop_z]),
            scale=np.array([1.2, 0.8, tabletop_thickness]),
            size=1.0,
            color=np.array([0.55, 0.42, 0.28]),
        )
    )
    add_table_legs(world, FixedCuboid, table_height)
    add_objects(world, FixedCapsule, FixedCuboid, FixedSphere, np, object_z_base)

    camera_prim_path = "/World/RGBD_Camera"
    camera_position = np.array([0.0, 0.8, 1.6])
    camera_target = np.array([0.0, 0.0, object_z_base])
    camera_orientation = look_at_quat_usd_camera(
        camera_position,
        camera_target,
        np.array([0.0, 0.0, 1.0]),
        rot_matrices_to_quats,
    )
    camera = Camera(
        prim_path=camera_prim_path,
        name="rgbd_camera",
        position=camera_position,
        frequency=30,
        resolution=(config.width, config.height),
    )

    world.reset()
    camera.initialize()
    camera.set_world_pose(
        position=camera_position, orientation=camera_orientation, camera_axes="usd"
    )
    camera.add_distance_to_image_plane_to_frame()
    camera.set_clipping_range(0.05, 2.0)
    camera.set_focal_length(1.8)  # cm
    camera.set_focus_distance(float(np.linalg.norm(camera_target - camera_position)))

    for _ in range(config.warmup_steps):
        world.step(render=True)

    if config.mode == MODE_DEVELOPMENT:
        run_development_loop(simulation_app, world, camera_prim_path)
        return

    for frame_index in range(config.frames):
        world.step(render=True)
        save_rgbd_frame(
            camera, config.output_dir, frame_index, Image, np, config.save_depth_png
        )

    print(f"Saved {config.frames} RGBD frames to {config.output_dir}", flush=True)


def run_development_loop(
    simulation_app: SimulationApp, world, camera_prim_path: str
) -> None:
    """GUI を開いたままシーンを編集できる状態で維持する。

    Args:
        simulation_app: 起動済みの Isaac Sim アプリケーション。
        world: Isaac Sim の World。
        camera_prim_path: viewport に設定するカメラ Prim パス。
    """
    set_active_viewport_camera(camera_prim_path)
    print(
        "Development mode: GUI 上で物体や /World/RGBD_Camera を調整できます。"
        "ウィンドウを閉じると終了します。",
        flush=True,
    )
    while simulation_app.is_running():
        world.step(render=True)


def set_active_viewport_camera(camera_prim_path: str) -> None:
    """アクティブ viewport を指定カメラの視点に切り替える。

    Args:
        camera_prim_path: viewport に設定するカメラ Prim パス。
    """
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_prim_path
    except Exception as exc:  # noqa: BLE001
        print(f"Viewport camera setup skipped: {exc}", flush=True)


def add_lights(world, usd_lux) -> None:
    """シーンに照明を追加する。

    Args:
        world: Isaac Sim の World。
        usd_lux: USD の照明モジュール。
    """
    stage = world.scene.stage
    dome_light = usd_lux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(300.0)

    key_light = usd_lux.DistantLight.Define(stage, "/World/KeyLight")
    key_light.CreateIntensityAttr(2500.0)
    key_light.CreateAngleAttr(0.35)


def add_table_legs(world, fixed_cuboid_cls, table_height: float) -> None:
    """テーブル脚を追加する。

    Args:
        world: Isaac Sim の World。
        fixed_cuboid_cls: FixedCuboid クラス。
        table_height: テーブルトップ下端の高さ。
    """
    import numpy as np

    leg_positions = [(-0.5, -0.3), (-0.5, 0.3), (0.5, -0.3), (0.5, 0.3)]
    for index, (x_pos, y_pos) in enumerate(leg_positions):
        world.scene.add(
            fixed_cuboid_cls(
                prim_path=f"/World/TableLeg_{index}",
                name=f"table_leg_{index}",
                position=np.array([x_pos, y_pos, table_height / 2.0]),
                scale=np.array([0.06, 0.06, table_height]),
                size=1.0,
                color=np.array([0.38, 0.28, 0.18]),
            )
        )


def add_objects(
    world,
    fixed_capsule_cls,
    fixed_cuboid_cls,
    fixed_sphere_cls,
    np_module,
    z_base: float,
) -> None:
    """テーブル上に立方体、球、カプセル、直方体を追加する。

    Args:
        world: Isaac Sim の World。
        fixed_capsule_cls: FixedCapsule クラス。
        fixed_cuboid_cls: FixedCuboid クラス。
        fixed_sphere_cls: FixedSphere クラス。
        np_module: NumPy モジュール。
        z_base: テーブルトップ上面の高さ。
    """
    world.scene.add(
        fixed_cuboid_cls(
            prim_path="/World/ObjectCube",
            name="object_cube",
            position=np_module.array([-0.33, -0.18, z_base + 0.08]),
            scale=np_module.array([0.16, 0.16, 0.16]),
            size=1.0,
            color=np_module.array([0.9, 0.18, 0.14]),
        )
    )
    world.scene.add(
        fixed_sphere_cls(
            prim_path="/World/ObjectSphere",
            name="object_sphere",
            position=np_module.array([-0.1, 0.2, z_base + 0.08]),
            radius=0.08,
            color=np_module.array([0.1, 0.45, 0.95]),
        )
    )
    world.scene.add(
        fixed_capsule_cls(
            prim_path="/World/ObjectCapsule",
            name="object_capsule",
            position=np_module.array([0.2, -0.16, z_base + 0.09]),
            radius=0.045,
            height=0.18,
            color=np_module.array([0.12, 0.72, 0.32]),
        )
    )
    world.scene.add(
        fixed_cuboid_cls(
            prim_path="/World/ObjectRectangularPrism",
            name="object_rectangular_prism",
            position=np_module.array([0.34, 0.18, z_base + 0.05]),
            scale=np_module.array([0.24, 0.11, 0.1]),
            size=1.0,
            color=np_module.array([0.95, 0.65, 0.08]),
        )
    )


def look_at_quat_usd_camera(position, target, up, quat_converter) -> "np.ndarray":
    """カメラの -Z 軸を注視点へ向けるクォータニオンを計算する。

    Args:
        position: カメラ位置。
        target: カメラが注視する位置。
        up: ワールド上方向。
        quat_converter: 回転行列をクォータニオンへ変換する関数。

    Returns:
        scalar-first 形式の NumPy クォータニオン。
    """
    import numpy as np

    forward = target - position
    forward = forward / np.linalg.norm(forward)
    camera_z = -forward
    camera_x = np.cross(up, camera_z)
    camera_x = camera_x / np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    return quat_converter(np.column_stack((camera_x, camera_y, camera_z)))


def save_rgbd_frame(
    camera,
    output_dir: Path,
    frame_index: int,
    image_cls,
    np_module,
    save_depth_png: bool,
) -> None:
    """カメラフレームを RGB PNG と depth NPY として保存する。

    Args:
        camera: Isaac Sim の Camera。
        output_dir: 出力先ディレクトリ。
        frame_index: 保存するフレーム番号。
        image_cls: PIL Image クラス。
        np_module: NumPy モジュール。
        save_depth_png: 深度の可視化 PNG も保存するかどうか。
    """
    rgb = camera.get_rgb()
    depth = camera.get_depth()
    if rgb is None or depth is None:
        msg = "RGB または depth フレームを取得できませんでした。"
        raise RuntimeError(msg)

    image_cls.fromarray(rgb.astype(np_module.uint8)).save(
        output_dir / f"rgb_{frame_index:04d}.png"
    )
    np_module.save(
        output_dir / f"depth_{frame_index:04d}.npy", depth.astype(np_module.float32)
    )

    if save_depth_png:
        depth_image = normalize_depth(depth, np_module)
        image_cls.fromarray(depth_image).save(
            output_dir / f"depth_{frame_index:04d}.png"
        )


def normalize_depth(depth, np_module) -> "np.ndarray":
    """深度配列を可視化用の 8bit 画像へ変換する。

    Args:
        depth: 深度配列。
        np_module: NumPy モジュール。

    Returns:
        0 から 255 に正規化した深度画像。
    """
    valid = np_module.isfinite(depth) & (depth > 0.0)
    if not np_module.any(valid):
        return np_module.zeros(depth.shape, dtype=np_module.uint8)

    min_depth = float(np_module.min(depth[valid]))
    max_depth = float(np_module.max(depth[valid]))
    if max_depth <= min_depth:
        return np_module.zeros(depth.shape, dtype=np_module.uint8)

    normalized = (depth - min_depth) / (max_depth - min_depth)
    normalized = np_module.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).astype(np_module.uint8)


if __name__ == "__main__":
    main()

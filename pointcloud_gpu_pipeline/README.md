# Pointcloud GPU Pipeline

## Isaac Sim RGBD 撮影

`bringup_isaac_env.py` は Isaac Sim standalone script として起動し、テーブル上に立方体、球、カプセル、直方体を配置します。

実行モードは [bringup_isaac_env.py](/home/yusuke/analytical_skill_tutorial/pointcloud_gpu_pipeline/bringup_isaac_env.py) 冒頭の `RUN_MODE` 変数、または CLI の `--mode` で切り替えます。

- `production`: RGBD カメラで指定枚数を撮影して終了します。
- `development`: GUI を開き、物体の位置や追加、`/World/RGBD_Camera` の位置・向きを調整できる状態で維持します。

### 実行コマンド

Isaac Sim の Python から実行してください。

```bash
/home/yusuke/isaac-sim/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh bringup_isaac_env.py \
  --mode production \
  --frames 10 \
  --output-dir outputs/rgbd
```

開発モードで GUI を開く場合:

```bash
/home/yusuke/isaac-sim/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh bringup_isaac_env.py \
  --mode development
```

`RUN_MODE = "development"` に変更しておけば、CLI で `--mode development` を指定せずに開発モードで起動できます。

### 必要な環境変数

通常は Isaac Sim の `python.sh` が必要な環境を設定するため、追加の環境変数は不要です。

### 入力ファイル・出力ファイル

入力ファイルは不要です。`production` の出力先は `--output-dir` で指定します。

- `rgb_0000.png`: RGB 画像。
- `depth_0000.npy`: depth 配列。単位はメートルです。
- `depth_0000.png`: `--save-depth-png` 指定時だけ保存される可視化用 depth 画像。

`development` では撮影ファイルは保存せず、GUI ウィンドウを閉じるまでシーン編集用に実行を継続します。

### 主要なオプション

- `--mode`: `production` または `development`。デフォルトは `RUN_MODE` 変数。
- `--frames`, `-n`: production で撮影する枚数。デフォルトは `10`。
- `--output-dir`: production の出力先ディレクトリ。デフォルトは `outputs/rgbd`。
- `--width`: RGBD 画像の幅。デフォルトは `1280`。
- `--height`: RGBD 画像の高さ。デフォルトは `720`。
- `--warmup-steps`: 撮影前または GUI 表示前のレンダリングステップ数。デフォルトは `60`。
- `--show`: production でも headless ではなく GUI 付きで実行する。
- `--save-depth-png`: production で depth の可視化 PNG も保存する。

### 外部サービス・ポート番号

外部サービスや固定ポートは使用しません。Isaac Sim がローカル GPU とレンダラを使用します。

## GPU ベンチマーク用の最小 conda 環境

Isaac Sim 一式を含めずに、点群ベンチマークだけを実行する最小環境は `environment-gpu-benchmark-cuda128.yml` で作成します。

```bash
conda env create -f environment-gpu-benchmark-cuda128.yml
```

既に環境がある場合は、次のように実行します。

```bash
conda run -n pc-gpu-bench-cuda128 python benchmark_pointcloud_pipeline.py \
  --rgb outputs/rgbd/rgb_0000.png \
  --depth outputs/rgbd/depth_0000.npy \
  --fx 1100 \
  --fy 1100 \
  --runs 21 \
  --warmup-runs 1 \
  --backends numpy_cpu cupy_gpu numba_cuda open3d_cuda \
  --output outputs/benchmark_pointcloud_pipeline.json
```

この環境は Python、NumPy、Pillow、Numba、CUDA 12.8、CuPy、Open3D だけを入れるため、Isaac Sim 環境の clone より軽量です。

## RGBD から NumPy 点群を作成

`create_pointcloud_from_rgbd.py` は、RGB 画像と depth から NumPy で `Nx6` の点群を作成します。出力配列は `[x, y, z, r, g, b]` で、座標はカメラ座標系、depth の単位はメートルです。点群作成処理だけを `time.perf_counter()` で計測し、処理時間を ms で表示します。

### 実行コマンド

depth は実測値が入っている `.npy` を使用してください。`depth_*.png` は可視化用に正規化されているため、通常の点群作成には使いません。

```bash
conda run -n isaac-sim python create_pointcloud_from_rgbd.py \
  --rgb outputs/rgbd/rgb_0000.png \
  --depth outputs/rgbd/depth_0000.npy \
  --output outputs/pointcloud_0000.npy \
  --ply outputs/pointcloud_0000.ply \
  --fx 1100 \
  --fy 1100
```

### 必要な環境変数

追加の環境変数は不要です。`numpy` と `Pillow` が入っている Python 環境で実行してください。

### 入力ファイル・出力ファイル

- 入力: `--rgb` に RGB PNG、`--depth` に depth NPY または 16bit depth PNG を指定します。
- 出力: `--output` に `Nx6` の `float32` NPY を保存します。RGB 値は 0 から 255 の値です。
- 任意出力: `--ply` を指定すると確認用の ASCII PLY を保存します。

### 主要なオプション

- `--fx`, `--fy`: カメラ焦点距離。単位は pixel です。
- `--cx`, `--cy`: カメラ主点。省略時は画像中心を使用します。
- `--min-depth`, `--max-depth`: 点群化する depth 範囲。単位はメートルです。
- `--depth-scale`: 整数 depth PNG をメートルへ変換する除数。mm 単位の 16bit PNG なら `1000` です。
- `--allow-8bit-depth`: 8bit depth PNG の読み込みを許可します。可視化 PNG から作る点群は実スケールにならないため、基本的には使用しません。

### 外部サービス・ポート番号

外部サービスや固定ポートは使用しません。

## 点群処理パイプラインを実行してクラスタ色付き点群を保存

`run_pointcloud_pipeline.py` は、RGBD から点群生成、VoxelGrid ダウンサンプリング、RANSAC 平面削除、Euclidean clustering を実行し、クラスタごとに色分けした点群を保存します。

### 実行コマンド

```bash
conda run -n isaac-sim python run_pointcloud_pipeline.py \
  --rgb outputs/rgbd/rgb_0000.png \
  --depth outputs/rgbd/depth_0000.npy \
  --fx 1100 \
  --fy 1100 \
  --clustered-output outputs/pipeline_clustered.npy \
  --clustered-ply outputs/pipeline_clustered.ply \
  --summary-output outputs/pipeline_summary.json
```

### 入力ファイル・出力ファイル

- 入力: `--rgb` に RGB PNG、`--depth` に depth NPY を指定します。
- 出力: `--pointcloud-output` に RGBD から生成した標準点群 NPY を保存します。
- 出力: `--downsampled-output` に VoxelGrid 後点群 NPY を保存します。
- 出力: `--remaining-output` に平面除去後点群 NPY を保存します。
- 出力: `--clustered-output` にクラスタ色付き点群 NPY を保存します。
- 出力: `--clustered-ply` にクラスタ色付き点群 PLY を保存します。
- 出力: `--summary-output` に平面係数、クラスタ数、クラスタ点数を JSON 保存します。

### 主要なオプション

- `--voxel-size`: VoxelGrid の一辺の長さ[m]。標準は `0.01`。
- `--plane-distance-threshold`: RANSAC 平面 inlier の距離しきい値[m]。標準は `0.01`。
- `--plane-max-iterations`: RANSAC の最大反復回数。標準は `1000`。
- `--cluster-tolerance`: Euclidean clustering の距離しきい値[m]。標準は `0.03`。
- `--cluster-min-size`: 出力するクラスタの最小点数。標準は `30`。
- `--random-seed`: RANSAC の乱数 seed。標準は `0`。

### 外部サービス・ポート番号

外部サービスや固定ポートは使用しません。

## 点群処理パイプラインの速度・精度比較

`benchmark_pointcloud_pipeline.py` は、同じ RGBD 入力から標準形式点群 `Nx6 [x, y, z, r, g, b]` を作成し、点群生成、VoxelGrid ダウンサンプリング、RANSAC 平面削除、Euclidean clustering、パイプライン全体の処理時間を比較します。`cupy_gpu`, `numba_cuda`, `open3d_cuda` は GPU backend として比較できます。`open3d_cuda` は Open3D tensor API で VoxelGrid、RANSAC 平面削除、DBSCAN clustering を計測します。

### 実行コマンド

```bash
conda run -n pc-gpu-bench-cuda128 python benchmark_pointcloud_pipeline.py \
  --rgb outputs/rgbd/rgb_0000.png \
  --depth outputs/rgbd/depth_0000.npy \
  --fx 1100 \
  --fy 1100 \
  --runs 21 \
  --warmup-runs 1 \
  --output outputs/benchmark_pointcloud_pipeline.json
```

特定バックエンドだけ比較する場合:

```bash
conda run -n isaac-sim python benchmark_pointcloud_pipeline.py \
  --rgb outputs/rgbd/rgb_0000.png \
  --depth outputs/rgbd/depth_0000.npy \
  --fx 1100 \
  --fy 1100 \
  --backends numpy_cpu numba_cuda
```

### 必要な環境変数

追加の環境変数は不要です。GPU バックエンドはインストール状況に応じて自動判定され、利用できない場合は結果 JSON に `unavailable` または `error` として記録されます。

### 入力ファイル・出力ファイル

- 入力: `--rgb` に RGB PNG、`--depth` に depth NPY を指定します。
- 出力: `--output` にステージごとの `first_run_ms`, `mean_ms`, `min_ms`, `max_ms`, `max_abs_error_xyz`, `mean_abs_error_xyz`, `count_mismatch` を JSON 保存します。

### 主要なオプション

- `--backends`: 比較するバックエンド名。標準は `numpy_cpu cupy_gpu numba_cuda open3d_cuda` です。
- `--runs`: 総実行回数。標準は `21`。
- `--warmup-runs`: 統計から除外する先頭実行回数。標準は `1`。
- `--voxel-size`: VoxelGrid の一辺の長さ[m]。標準は `0.01`。
- `--plane-distance-threshold`: RANSAC 平面 inlier の距離しきい値[m]。標準は `0.01`。
- `--plane-max-iterations`: RANSAC の最大反復回数。標準は `1000`。
- `--cluster-tolerance`: Euclidean clustering の距離しきい値[m]。標準は `0.03`。
- `--cluster-min-size`: 出力するクラスタの最小点数。標準は `30`。
- `--fx`, `--fy`, `--cx`, `--cy`: カメラ内部パラメータ。
- `--min-depth`, `--max-depth`: 点群化する depth 範囲。単位はメートルです。

### 拡張方法

点群生成バックエンド追加時は `pointcloud_generation_backends.py` に `PointcloudGenerationBackend` 互換クラスを追加し、`default_generation_backends()` に登録します。後段処理の CPU 基準実装は `pointcloud_cpu_algorithms.py`、ステージ別計測と JSON 出力は `pointcloud_pipeline_benchmark.py` に分離しています。

### 外部サービス・ポート番号

外部サービスや固定ポートは使用しません。


## ファイル構成

- `docs/specification.md`: 仕様駆動開発/TDD の基準仕様。入出力、受け入れ基準、ベンチマーク条件を記載します。
- `pointcloud_data_types.py`: `CameraIntrinsics`, `RgbdInput`, `PipelineParameters`, `PlaneSegmentationResult`, `PointcloudPipelineResult` などの共有 dataclass を定義します。
- `pointcloud_cpu_algorithms.py`: NumPy CPU の基準実装を定義します。主な関数は `create_pointcloud_from_rgbd`, `voxel_downsample_pointcloud`, `detect_plane_ransac`, `euclidean_cluster_pointcloud`, `colorize_clusters`, `run_pointcloud_pipeline`, `save_ply` です。
- `pointcloud_generation_backends.py`: RGBD から点群生成を行うバックエンド class を定義します。`NumpyPointcloudBackend`, `CupyPointcloudBackend`, `NumbaCudaPointcloudBackend`, `Open3DCudaPointcloudBackend` が入っています。
- `create_pointcloud_from_rgbd.py`: RGBD ファイルから `Nx6 [x, y, z, r, g, b]` 点群 NPY/PLY を作成する CLI です。
- `pointcloud_cupy_pipeline.py`: CuPy で点群生成、VoxelGrid、RANSAC 平面削除、Euclidean clustering の GPU 常駐処理を行います。
- `pointcloud_gpu_backend_pipelines.py`: Numba CUDA の点群生成結果を GPU 上で受け渡し、GPU 常駐パイプラインへ接続します。
- `pointcloud_pipeline_benchmark.py`: ステージ別・パイプライン全体の計測、統計計算、JSON 保存を行います。
- `run_pointcloud_pipeline.py`: パイプラインを実行してクラスタ色付き点群 NPY/PLY を保存する CLI です。
- `benchmark_pointcloud_pipeline.py`: パイプラインベンチマーク CLI です。
- `tests/test_pointcloud_cpu_algorithms.py`: 小さな合成データで仕様を固定する pytest です。

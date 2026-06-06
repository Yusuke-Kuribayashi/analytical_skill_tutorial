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

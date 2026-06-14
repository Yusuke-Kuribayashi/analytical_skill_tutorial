# 目的

NVIDIA Isaac Sim で取得した RGBD 画像を入力として、点群処理パイプラインを CPU / CuPy / Numba / Open3D で実装し、各処理単体およびパイプライン全体の処理時間と精度差分を比較する。

この課題を通して、GPU を用いた点群処理の高速化効果と、GPU 実装時に発生する初回実行コスト、データ転送、同期、実装方式の違いを学ぶ。

# 比較対象

- CPU: NumPy 実装を基準実装とする。
- CuPy: GPU 配列演算による実装。
- Numba: Numba CUDA による GPU 実装。
- Open3D: Tensor API CUDA による GPU 実装。

# 点群の標準形式

点群の標準出力形式は `Nx6 float32` の配列とし、各行は次の順序で保持する。

```text
[x, y, z, r, g, b]
```

- `x, y, z`: カメラ座標系の 3D 座標。単位はメートル。
- `r, g, b`: RGB 値。範囲は `0` から `255`。
- 無効点は標準出力形式には含めない。
- ベンチマーク内部で dense 表現を使う場合でも、保存・後段処理・テスト対象の標準形式は `Nx6 [x, y, z, r, g, b]` とする。

# 入力仕様

## RGB 画像

- 形式: PNG
- shape: `HxWx3`
- dtype: `uint8`
- 色順: RGB

## Depth

- 形式: NPY を標準とする。
- shape: `HxW`
- dtype: `float32` または `float64`。読み込み後は `float32` に変換する。
- 単位: メートル。
- RGB と depth の画像サイズは一致している必要がある。

## カメラ内部パラメータ

RGBD から点群を生成するときは、次の内部パラメータを使用する。

- `fx`: x 方向の焦点距離[pixel]。
- `fy`: y 方向の焦点距離[pixel]。
- `cx`: 主点 x 座標[pixel]。未指定時は画像中心を使用する。
- `cy`: 主点 y 座標[pixel]。未指定時は画像中心を使用する。

## Depth の有効条件

次の条件をすべて満たす depth のみ点群化する。

- `depth` が有限値である。
- `depth > min_depth`。
- `depth < max_depth`。

# 対象パイプライン

点群処理パイプラインは次の順序で実行する。

1. 点群生成
2. ダウンサンプリング
3. 平面検出および平面点群の削除
4. クラスタリング

# 1. 点群生成

RGBD 画像から `Nx6 [x, y, z, r, g, b]` の点群を生成する。

座標変換は次の式に従う。

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth[v, u]
```

## 受け入れ基準

- RGB と depth のサイズが一致しない場合は `ValueError` を送出する。
- 無効 depth は出力点群に含めない。
- CPU 実装を基準とし、GPU 実装の `xyz` 最大絶対誤差は `1e-5` 以下とする。
- GPU 実装と CPU 実装の有効点数は一致すること。

# 2. ダウンサンプリング

ダウンサンプリングは PCL の `VoxelGrid` と同様のボクセルグリッド方式とする。

- 3D 空間を `voxel_size` の立方体ボクセルへ分割する。
- 同一ボクセル内の点は 1 点に集約する。
- 集約点の `x, y, z` はボクセル内点群の重心とする。
- 集約点の `r, g, b` はボクセル内 RGB の平均値とする。

## 初期パラメータ

- `voxel_size = 0.01`

## 受け入れ基準

- `voxel_size <= 0` の場合は `ValueError` を送出する。
- 出力形式は `Nx6 [x, y, z, r, g, b]` を維持する。
- 同一入力・同一パラメータに対して決定的な出力を返す。
- GPU 実装は CPU 実装と同じボクセル分割結果を返す。
- GPU 実装の重心座標の最大絶対誤差は `1e-5` 以下とする。

# 3. 平面検出および削除

平面検出は PCL の `SACSegmentation` で一般的に使われる構成を参考にし、RANSAC による平面モデル推定を採用する。

- モデル: 平面 `ax + by + cz + d = 0`
- 推定手法: RANSAC
- 平面 inlier は、点から推定平面までの距離が `plane_distance_threshold` 以下の点とする。
- 平面削除後点群は、inlier を除外した点群とする。

## 初期パラメータ

- `plane_distance_threshold = 0.01`
- `plane_max_iterations = 1000`
- `min_inlier_ratio = 0.1`

## 出力

- 平面係数 `[a, b, c, d]`
- 平面 inlier indices または mask
- 平面削除後点群 `Nx6 [x, y, z, r, g, b]`

## 受け入れ基準

- 点数が平面推定に必要な 3 点未満の場合は `ValueError` を送出する。
- `plane_distance_threshold <= 0` の場合は `ValueError` を送出する。
- 推定された法線 `[a, b, c]` は正規化する。
- 明らかな平面テストデータに対して、期待される inlier を検出できる。
- 平面削除後点群には inlier が含まれない。
- GPU 実装の inlier 判定は CPU 実装と一致することを目標とする。ただし RANSAC の乱択差があるため、テストでは乱数 seed を固定する。

# 4. クラスタリング

クラスタリングは Euclidean clustering とする。

- 点間距離が `cluster_tolerance` 以下の点を同一クラスタ候補として連結する。
- Kd-tree、グリッド、または GPU 向け近傍探索構造を利用してよい。
- 出力クラスタは点群 index のリスト、またはクラスタ ID 配列とする。

## 初期パラメータ

- `cluster_tolerance = 0.03`
- `cluster_min_size = 30`
- `cluster_max_size` は任意指定とし、未指定時は上限なしとする。

## 受け入れ基準

- `cluster_tolerance <= 0` の場合は `ValueError` を送出する。
- `cluster_min_size <= 0` の場合は `ValueError` を送出する。
- 距離的に分離した小規模テストデータに対して、期待クラスタ数を返す。
- `cluster_min_size` 未満のクラスタは出力しない。
- 同一入力・同一パラメータに対して決定的な出力を返す。

# ベンチマーク仕様

## 計測対象

次の単位で処理時間を計測する。

- 点群生成
- ダウンサンプリング
- 平面検出および削除
- クラスタリング
- パイプライン全体

## 実行回数

各計測対象について 21 回実行する。

- 1 回目は cold start / warmup として統計から除外する。
- 2 回目から 21 回目までの 20 回を統計対象とする。
- 1 回目の処理時間は `first_run_ms` として保存する。

GPU 実装では、計測前後に必要な同期を行い、非同期実行時間だけを過小評価しないようにする。

## 出力統計

各処理・各バックエンドについて、少なくとも次の値を出力する。

- `backend`: バックエンド名。
- `stage`: 計測対象ステージ名。
- `runs`: 総実行回数。標準は `21`。
- `warmup_runs`: 統計から除外した回数。標準は `1`。
- `measured_runs`: 統計対象回数。標準は `20`。
- `first_run_ms`: 1 回目の処理時間。
- `mean_ms`: 2 回目以降の平均処理時間。
- `min_ms`: 2 回目以降の最小処理時間。
- `max_ms`: 2 回目以降の最大処理時間。
- `status`: `ok`, `unavailable`, `error` のいずれか。
- `error`: エラー時のメッセージ。

# 精度比較仕様

- CPU 実装を基準実装とする。
- GPU 実装は、CPU 実装と同じ入力・同じパラメータで比較する。
- 点群生成とダウンサンプリングでは、`xyz` 最大絶対誤差 `1e-5` 以下を目標とする。
- 平面検出とクラスタリングでは、乱択や探索順序の差を避けるため、テストでは乱数 seed と入力順序を固定する。
- 精度比較結果はベンチマーク JSON に含める。

# エラー処理仕様

次の場合は明示的なエラーまたは `unavailable` として扱う。

- 入力ファイルが存在しない。
- RGB と depth のサイズが一致しない。
- パラメータが不正である。
- GPU ライブラリがインストールされていない。
- CUDA device が利用できない。
- 対象バックエンドが未実装である。

GPU ライブラリや CUDA device が利用できない場合、ベンチマーク全体は停止せず、そのバックエンドの結果を `unavailable` として記録する。

# TDD 方針

1. NumPy CPU 実装の単体テストを先に作成し、仕様を固定する。
2. GPU 実装は CPU 実装の出力と比較するテストを追加する。
3. GPU が利用できない環境では、GPU テストは skip し、ベンチマークでは `unavailable` として扱う。
4. 小さな合成点群を使い、期待値が手計算できるテストを優先する。
5. 実 RGBD データを使うテストは integration test として分離する。

# 実行方法の記載

新しいスクリプト、ベンチマーク、パイプライン処理を追加または変更した場合は、README.md または該当ディレクトリの README に次を記載する。

- 実行コマンド
- 必要な環境変数
- 入力ファイル・出力ファイル
- 主要なオプション
- 依存する外部サービスやポート番号

# 参考

- PCL VoxelGrid: https://pointclouds.org/documentation/tutorials/voxel_grid.html
- PCL Plane model segmentation: https://pointclouds.org/documentation/tutorials/planar_segmentation.html
- PCL Euclidean Cluster Extraction: https://pointclouds.org/documentation/tutorials/cluster_extraction.html

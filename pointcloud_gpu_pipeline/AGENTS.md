# Docstring

Python の関数・メソッド・クラスには、簡潔な 日本語の Google 形式 docstring を記述すること。

例:
```
def load_config(path: Path) -> AppConfig:   
    """設定ファイルを読み込む。

    Args:
        path: 設定ファイルのパス。

    Returns:
        読み込んだアプリケーション設定。

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合。
    """
```

# インポート
ライブラリをインポートする際は、ファイルの上に固めること。関数やクラス内でライブラリを読み込むことは禁止。

悪い例
```
def test_function(image: np.ndarray):
    import matplotlib.pyplot as plt

    plt.imshow(image)
```


# Lint / Format / Type Check / Test

コード作成・修正後およびgit push前は、対象コンポーネントで以下を実行し、指摘があれば修正すること。

```
# backend または cdt
conda activate isaac-sim
ruff check .
ruff format .
mypy .
pytest -q

# conda 環境を activate できない非対話実行では以下を使う
conda run -n isaac-sim ruff check .
conda run -n isaac-sim ruff format .
conda run -n isaac-sim mypy .
conda run -n isaac-sim pytest -q

# frontend
npm run lint
npm run format
npm run build
```

Python コンポーネントでは、原則として conda 環境上で ruff, mypy, pytest を通過させること。
ただし、既存コード起因で全体チェックが失敗する場合は、変更対象ファイルに限定して実行し、失敗理由を作業報告に記載する。

# ファイル構成

実行ファイルは原則として 約300行前後 を目安とする。
300行を大きく超える場合は、責務ごとにモジュールを分割すること。

実装コードは、原則として以下の方針で構成する。

- トップレベルに長い処理を直接書かない
- 関数・クラスに責務を分割する
- 入出力、設定、ビジネスロジック、外部サービス連携を分離する
- 再利用・拡張しやすい構造にする
- 実行エントリポイントは main() や CLI 層に集約する

# コメント

処理の意図や実装の大まかな流れが分かりにくい箇所には、日本語コメントを記述すること。
ただし、コードを読めば明らかな処理に対する冗長なコメントは避ける。

良い例:
```
# 推論結果を後段の制御周期に合わせるため、最新フレームではなく遅延補正後のフレームを使用する。
```

悪い例:
```
# iに1を足す
i += 1
```

# 実行方法の記載

新しいスクリプト、バッチ処理、API、学習処理、推論処理を追加した場合は、実行方法を README.md または該当ディレクトリの README に記載すること。

記載内容には、少なくとも以下を含める。

- 実行コマンド
- 必要な環境変数
- 入力ファイル・出力ファイル
- 主要なオプション
- 依存する外部サービスやポート番号

# パラメータ管理

学習モデル、推論処理、実験スクリプトなど、パラメータ変更が頻繁に発生するコードでは、値をコード内に直接固定しないこと。

原則として、以下のいずれかで変更可能にする。

- bash スクリプトの引数
- 設定ファイル
- 環境変数
- CLI オプション

例:
```
bash scripts/train_model.sh \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --epochs 50
```

学習条件、モデル構成、データパス、出力先などは、再実行・比較実験がしやすい形で管理すること。
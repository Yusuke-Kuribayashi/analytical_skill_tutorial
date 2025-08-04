"""
PyTorch Tutorial Package
========================

機械学習とPyTorchのためのチュートリアルとライブラリ
libディレクトリには自作のライブラリファイルが含まれています
"""

# libディレクトリから主要な関数・クラスをインポート
from .lib import *

# バージョン情報
__version__ = "0.1.0"

# libディレクトリへの直接アクセスも提供
from . import lib
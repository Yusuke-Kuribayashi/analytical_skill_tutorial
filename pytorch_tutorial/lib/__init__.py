"""
PyTorch Tutorial Library Package
===============================

COCOデータセット処理、モデル訓練、評価のためのユーティリティ関数群を含むライブラリ
"""

# COCO評価関連
from .coco_eval import CocoEvaluator, convert_to_xywh, merge, create_common_coco_eval

# COCOユーティリティ関連  
from .coco_utils import (
    FilterAndRemapCocoCategories,
    ConvertCocoPolysToMask,
    CocoDetection,
    convert_coco_poly_to_mask,
    get_coco,
    get_coco_kp,
    get_coco_api_from_dataset
)

# 訓練・評価エンジン
from .engine import train_one_epoch, evaluate

# ユーティリティ関数
from .utils import (
    SmoothedValue,
    MetricLogger,
    collate_fn,
    setup_for_distributed,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    get_world_size,
    get_rank,
    is_main_process,
    save_on_master
)

# データ変換関連
from .transforms import (
    Compose,
    RandomHorizontalFlip,
    PILToTensor,
    ConvertImageDtype,
    RandomIoUCrop,
    RandomZoomOut,
    RandomPhotometricDistort,
    ScaleJitter,
    FixedSizeCrop,
    RandomShortestSize,
    SimpleCopyPaste
)

# 日本語フォント設定
from .japanese_font_config import setup_japanese_font

__version__ = "0.1.0"
__all__ = [
    # COCO評価
    "CocoEvaluator", "convert_to_xywh", "merge", "create_common_coco_eval",
    
    # COCOユーティリティ
    "FilterAndRemapCocoCategories", "ConvertCocoPolysToMask", "CocoDetection",
    "convert_coco_poly_to_mask", "get_coco", "get_coco_kp", "get_coco_api_from_dataset",
    
    # 訓練・評価
    "train_one_epoch", "evaluate",
    
    # ユーティリティ
    "SmoothedValue", "MetricLogger", "collate_fn", "setup_for_distributed",
    "init_distributed_mode", "is_dist_avail_and_initialized", "get_world_size",
    "get_rank", "is_main_process", "save_on_master",
    
    # 変換
    "Compose", "RandomHorizontalFlip", "PILToTensor", "ConvertImageDtype",
    "RandomIoUCrop", "RandomZoomOut", "RandomPhotometricDistort", "ScaleJitter",
    "FixedSizeCrop", "RandomShortestSize", "SimpleCopyPaste",

    # 日本語フォント設定
    "setup_japanese_font"
]
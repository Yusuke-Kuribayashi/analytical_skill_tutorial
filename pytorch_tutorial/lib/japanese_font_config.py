"""
日本語フォント設定用の共通モジュール
matplotlib で日本語を表示するための設定
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

def setup_japanese_font():
    """
    日本語フォントを設定する関数
    
    Returns:
        str: 設定されたフォント名（設定できなかった場合はNone）
    """
    try:
        # 利用可能な日本語フォントを探す
        all_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 優先順位順に日本語フォントを探す
        preferred_fonts = [
            'Noto Sans CJK JP',
            'Noto Serif CJK JP', 
            'Hiragino Sans',
            'Yu Gothic',
            'Meiryo',
            'Takao Gothic',
            'IPA Gothic',
            'DejaVu Sans'
        ]
        
        selected_font = None
        for font in preferred_fonts:
            if font in all_fonts:
                selected_font = font
                break
        
        if selected_font:
            # フォント設定を適用
            mpl.rcParams['font.family'] = selected_font
            mpl.rcParams['font.sans-serif'] = [selected_font] + mpl.rcParams['font.sans-serif']
            
            # 日本語を含むプロットでのマイナス記号問題を解決
            mpl.rcParams['axes.unicode_minus'] = False
            
            print(f"日本語フォントを設定しました: {selected_font}")
            return selected_font
        else:
            print("利用可能な日本語フォントが見つかりませんでした")
            return None
            
    except Exception as e:
        print(f"フォント設定中にエラーが発生しました: {e}")
        return None

def get_available_japanese_fonts():
    """
    利用可能な日本語フォントのリストを取得
    
    Returns:
        list: 利用可能な日本語フォント名のリスト
    """
    try:
        all_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 日本語フォントの候補
        japanese_keywords = ['noto', 'cjk', 'hiragino', 'yu gothic', 'meiryo', 'takao', 'ipa']
        
        japanese_fonts = []
        for font in all_fonts:
            for keyword in japanese_keywords:
                if keyword.lower() in font.lower():
                    japanese_fonts.append(font)
                    break
        
        return sorted(set(japanese_fonts))
        
    except Exception as e:
        print(f"フォント取得中にエラーが発生しました: {e}")
        return []

# モジュールをインポートした時点で自動的にフォント設定を適用
if __name__ != "__main__":
    setup_japanese_font() 
"""
PyTorch テンソル基礎チュートリアル
テンソル作成、操作、計算の基本的な例
"""

import torch
import numpy as np

def tensor_creation():
    """テンソル作成の様々な方法"""
    print("=== テンソル作成の例 ===")
    
    # 1. リストからテンソル作成
    data = [[1, 2], [3, 4]]
    tensor_from_list = torch.tensor(data)
    print(f"リストからテンソル: {tensor_from_list}")
    
    # 2. NumPy配列からテンソル作成
    np_array = np.array(data)
    tensor_from_numpy = torch.from_numpy(np_array)
    print(f"NumPyからテンソル: {tensor_from_numpy}")
    
    # 3. 特殊なテンソル作成
    zeros_tensor = torch.zeros(2, 3)
    ones_tensor = torch.ones(2, 3)
    random_tensor = torch.rand(2, 3)
    
    print(f"ゼロテンソル:\n{zeros_tensor}")
    print(f"1テンソル:\n{ones_tensor}")
    print(f"ランダムテンソル:\n{random_tensor}")
    
    return tensor_from_list

def tensor_operations(tensor):
    """テンソル操作の例"""
    print("\n=== テンソル操作の例 ===")
    
    # 形状の確認
    print(f"形状: {tensor.shape}")
    print(f"データ型: {tensor.dtype}")
    print(f"デバイス: {tensor.device}")
    
    # テンソル操作
    tensor_float = tensor.float()
    
    # 算術演算
    result_add = tensor_float + 10
    result_mul = tensor_float * 2
    result_pow = torch.pow(tensor_float, 2)
    
    print(f"元のテンソル: {tensor_float}")
    print(f"加算: {result_add}")
    print(f"乗算: {result_mul}")
    print(f"累乗: {result_pow}")
    
    # 行列演算
    if tensor_float.shape[0] == tensor_float.shape[1]:
        matrix_mul = torch.matmul(tensor_float, tensor_float)
        print(f"行列積: {matrix_mul}")
    
    return tensor_float

def tensor_indexing_slicing(tensor):
    """インデックスとスライシング"""
    print("\n=== インデックスとスライシング ===")
    
    # より大きなテンソルを作成
    large_tensor = torch.rand(4, 4)
    print(f"4x4テンソル:\n{large_tensor}")
    
    # インデックス
    print(f"[0, 0]要素: {large_tensor[0, 0]}")
    print(f"最初の行: {large_tensor[0, :]}")
    print(f"最初の列: {large_tensor[:, 0]}")
    
    # スライシング
    print(f"2x2サブテンソル:\n{large_tensor[:2, :2]}")
    
    return large_tensor

def tensor_reshape(tensor):
    """テンソルの形状変更"""
    print("\n=== テンソル形状変更 ===")
    
    original_shape = tensor.shape
    print(f"元の形状: {original_shape}")
    
    # reshape
    reshaped = tensor.reshape(-1)  # 1次元に変換
    print(f"1次元に変換: {reshaped}, 形状: {reshaped.shape}")
    
    # view (メモリを共有)
    viewed = tensor.view(1, -1)
    print(f"viewで変換: {viewed}, 形状: {viewed.shape}")
    
    # transpose
    if len(tensor.shape) == 2:
        transposed = tensor.transpose(0, 1)
        print(f"転置: {transposed}, 形状: {transposed.shape}")
    
    return reshaped

def gradient_example():
    """勾配計算の例"""
    print("\n=== 勾配計算の例 ===")
    
    # requires_grad=Trueで勾配計算を有効化
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    
    # 計算グラフを構築
    z = x * y + torch.sin(x)
    
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z = x * y + sin(x): {z}")
    
    # 逆伝播
    z.backward()
    
    print(f"dz/dx: {x.grad}")
    print(f"dz/dy: {y.grad}")

def main():
    """メイン関数"""
    print("PyTorchテンソル基礎チュートリアル")
    print("=" * 40)
    
    # テンソル作成
    tensor = tensor_creation()
    
    # テンソル操作
    tensor_ops = tensor_operations(tensor)
    
    # インデックスとスライシング
    large_tensor = tensor_indexing_slicing(tensor_ops)
    
    # 形状変更
    reshaped = tensor_reshape(large_tensor)
    
    # 勾配計算
    gradient_example()
    
    print("\nチュートリアル完了！")

if __name__ == "__main__":
    main() 
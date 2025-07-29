"""
PyTorch 線形回帰チュートリアル
簡単な線形回帰モデルの実装と訓練
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
try:
    from japanese_font_config import setup_japanese_font
except ImportError:
    setup_japanese_font = None

# 設定
if setup_japanese_font:
    setup_japanese_font()
else:
    print("日本語フォント設定をスキップします")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

class LinearRegressionModel(nn.Module):
    """線形回帰モデル"""
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def generate_data():
    """サンプルデータ生成"""
    print("=== データ生成 ===")
    
    # 真の係数
    true_weight = 2.0
    true_bias = 1.0
    
    # 入力データ
    x = torch.randn(100, 1)
    # 目標値（ノイズ付き）
    y = true_weight * x + true_bias + 0.1 * torch.randn(100, 1)
    
    print(f"データポイント数: {x.shape[0]}")
    print(f"真の重み: {true_weight}, 真のバイアス: {true_bias}")
    
    return x, y, true_weight, true_bias

def train_model(model, x_train, y_train, epochs=1000, learning_rate=0.01):
    """モデル訓練"""
    print(f"\n=== モデル訓練 (エポック数: {epochs}) ===")
    
    # 損失関数とオプティマイザー
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 訓練ループ
    losses = []
    
    for epoch in range(epochs):
        # 順伝播
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 損失記録
        losses.append(loss.item())
        
        # 進捗表示
        if (epoch + 1) % 100 == 0:
            print(f"エポック [{epoch+1}/{epochs}], 損失: {loss.item():.4f}")
    
    return losses

def evaluate_model(model, x_test, y_test):
    """モデル評価"""
    print("\n=== モデル評価 ===")
    
    with torch.no_grad():
        predictions = model(x_test)
        mse = nn.MSELoss()(predictions, y_test)
        print(f"テスト損失 (MSE): {mse.item():.4f}")
        
        # 学習した重みとバイアス
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()
        print(f"学習した重み: {weight:.4f}")
        print(f"学習したバイアス: {bias:.4f}")
        
        return predictions, weight, bias

def plot_results(x, y, predictions, losses, true_weight, true_bias, learned_weight, learned_bias):
    """結果の可視化"""
    print("\n=== 結果の可視化 ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # データと予測の可視化
    x_np = x.numpy()
    y_np = y.numpy()
    pred_np = predictions.numpy()
    
    ax1.scatter(x_np, y_np, alpha=0.5, label='実際のデータ')
    ax1.plot(x_np, pred_np, 'r-', label=f'予測線 (y = {learned_weight:.2f}x + {learned_bias:.2f})')
    
    # 真の線も表示
    x_line = np.linspace(x_np.min(), x_np.max(), 100)
    y_true_line = true_weight * x_line + true_bias
    ax1.plot(x_line, y_true_line, 'g--', label=f'真の線 (y = {true_weight}x + {true_bias})')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('線形回帰結果')
    ax1.legend()
    ax1.grid(True)
    
    # 損失の推移
    ax2.plot(losses)
    ax2.set_xlabel('エポック')
    ax2.set_ylabel('損失 (MSE)')
    ax2.set_title('訓練損失の推移')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', 
                dpi=150, bbox_inches='tight')
    print("結果を 'linear_regression_results.png' に保存しました")

def advanced_example():
    """より高度な例：多変量線形回帰"""
    print("\n=== 多変量線形回帰の例 ===")
    
    # 多変量データ生成
    n_features = 3
    n_samples = 200
    
    X = torch.randn(n_samples, n_features)
    true_weights = torch.tensor([1.5, -2.0, 0.5]).reshape(-1, 1)
    true_bias = 0.8
    
    y = X @ true_weights + true_bias + 0.1 * torch.randn(n_samples, 1)
    
    # モデル作成
    model = LinearRegressionModel(n_features, 1)
    
    # 訓練
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(500):
        predictions = model(X)
        loss = criterion(predictions, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"エポック [{epoch+1}/500], 損失: {loss.item():.4f}")
    
    # 結果表示
    learned_weights = model.linear.weight.data.numpy().flatten()
    learned_bias = model.linear.bias.data.item()
    
    print(f"真の重み: {true_weights.numpy().flatten()}")
    print(f"学習した重み: {learned_weights}")
    print(f"真のバイアス: {true_bias}")
    print(f"学習したバイアス: {learned_bias:.4f}")

def main():
    """メイン関数"""
    print("PyTorch線形回帰チュートリアル")
    print("=" * 40)
    
    # データ生成
    x, y, true_weight, true_bias = generate_data()
    
    # モデル作成
    model = LinearRegressionModel(input_dim=1, output_dim=1)
    print(f"モデル構造:\n{model}")
    
    # 訓練
    losses = train_model(model, x, y, epochs=1000, learning_rate=0.01)
    
    # 評価
    predictions, learned_weight, learned_bias = evaluate_model(model, x, y)
    
    # 可視化
    try:
        plot_results(x, y, predictions, losses, true_weight, true_bias, 
                    learned_weight, learned_bias)
    except ImportError:
        print("matplotlib が利用できないため、可視化をスキップします")
    
    # 高度な例
    advanced_example()
    
    print("\nチュートリアル完了！")

if __name__ == "__main__":
    main() 
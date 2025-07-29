"""
PyTorch カスタムデータセットチュートリアル
独自のデータセットクラスとデータローダーの実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from japanese_font_config import setup_japanese_font

# 日本語フォント設定
try:
    setup_japanese_font()
except ImportError:
    print("日本語フォント設定をスキップします")

class CSVDataset(Dataset):
    """CSVファイルからデータを読み込むカスタムデータセット"""
    
    def __init__(self, csv_file=None, features=None, target=None, transform=None):
        """
        Args:
            csv_file (str): CSVファイルのパス
            features (array): 特徴量データ
            target (array): ターゲットデータ
            transform (callable): データ変換処理
        """
        if csv_file is not None:
            # CSVファイルから読み込み
            self.data_frame = pd.read_csv(csv_file)
        else:
            # 直接データを使用
            self.features = torch.FloatTensor(features)
            self.target = torch.FloatTensor(target)
            self.data_frame = None
        
        self.transform = transform
    
    def __len__(self):
        if self.data_frame is not None:
            return len(self.data_frame)
        else:
            return len(self.features)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.data_frame is not None:
            # CSVファイルから取得
            sample = self.data_frame.iloc[idx]
            features = sample.iloc[:-1].values.astype('float32')
            target = sample.iloc[-1]
            
            sample = {'features': torch.FloatTensor(features),
                     'target': torch.FloatTensor([target])}
        else:
            # 直接データから取得
            sample = {'features': self.features[idx],
                     'target': self.target[idx]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ImageDataset(Dataset):
    """画像データ用のカスタムデータセット（シミュレーション）"""
    
    def __init__(self, num_samples=1000, img_size=(32, 32), num_classes=3, transform=None):
        """
        Args:
            num_samples (int): サンプル数
            img_size (tuple): 画像サイズ (height, width)
            num_classes (int): クラス数
            transform (callable): データ変換処理
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform
        
        # ダミー画像データ生成
        np.random.seed(42)
        self.images = np.random.rand(num_samples, 3, *img_size).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        sample = {'image': torch.FloatTensor(image),
                 'label': torch.LongTensor([label])}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class Normalize(object):
    """データ正規化変換"""
    
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        features = sample['features']
        normalized_features = (features - self.mean) / self.std
        sample['features'] = normalized_features
        return sample

class AddNoise(object):
    """ノイズ追加変換"""
    
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor
    
    def __call__(self, sample):
        if 'image' in sample:
            image = sample['image']
            noise = torch.randn_like(image) * self.noise_factor
            sample['image'] = image + noise
        return sample

def create_sample_data():
    """サンプルデータの生成"""
    print("=== サンプルデータ生成 ===")
    
    # 回帰問題用のサンプルデータ
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # 特徴量生成
    X = np.random.randn(n_samples, n_features)
    
    # ターゲット生成（線形関係 + ノイズ）
    true_weights = np.array([1.5, -2.0, 0.5, 1.0, -0.8])
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    print(f"データ形状: X={X.shape}, y={y.shape}")
    
    # CSVファイルに保存
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/sample_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"データを {csv_path} に保存しました")
    
    return X, y, csv_path

def demonstrate_csv_dataset():
    """CSVデータセットの使用例"""
    print("\n=== CSVデータセット使用例 ===")
    
    # サンプルデータ作成
    X, y, csv_path = create_sample_data()
    
    # カスタムデータセット作成
    dataset = CSVDataset(csv_file=csv_path)
    
    print(f"データセットサイズ: {len(dataset)}")
    
    # サンプル確認
    sample = dataset[0]
    print(f"サンプル特徴量形状: {sample['features'].shape}")
    print(f"サンプルターゲット: {sample['target'].item():.4f}")
    
    # データローダー作成
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # バッチ確認
    for batch_idx, batch in enumerate(dataloader):
        print(f"バッチ {batch_idx}: 特徴量形状={batch['features'].shape}, "
              f"ターゲット形状={batch['target'].shape}")
        if batch_idx >= 2:  # 最初の3バッチのみ表示
            break
    
    return dataset, dataloader

def demonstrate_array_dataset():
    """配列データセットの使用例"""
    print("\n=== 配列データセット使用例 ===")
    
    # サンプルデータ生成
    X, y, _ = create_sample_data()
    
    # 訓練・テスト分割（手動実装）
    np.random.seed(42)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # データ正規化（手動実装）
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    # カスタムデータセット作成（変換付き）
    transform = Normalize(mean=0.0, std=1.0)
    
    train_dataset = CSVDataset(
        features=X_train_scaled, 
        target=y_train.reshape(-1, 1),
        transform=transform
    )
    
    test_dataset = CSVDataset(
        features=X_test_scaled, 
        target=y_test.reshape(-1, 1),
        transform=transform
    )
    
    print(f"訓練データセットサイズ: {len(train_dataset)}")
    print(f"テストデータセットサイズ: {len(test_dataset)}")
    
    # データローダー作成
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def demonstrate_image_dataset():
    """画像データセットの使用例"""
    print("\n=== 画像データセット使用例 ===")
    
    # 変換処理
    transform = AddNoise(noise_factor=0.05)
    
    # 画像データセット作成
    image_dataset = ImageDataset(
        num_samples=500, 
        img_size=(32, 32), 
        num_classes=3,
        transform=transform
    )
    
    print(f"画像データセットサイズ: {len(image_dataset)}")
    
    # サンプル確認
    sample = image_dataset[0]
    print(f"画像形状: {sample['image'].shape}")
    print(f"ラベル: {sample['label'].item()}")
    
    # データローダー作成
    image_loader = DataLoader(image_dataset, batch_size=16, shuffle=True)
    
    # バッチ確認
    for batch_idx, batch in enumerate(image_loader):
        print(f"バッチ {batch_idx}: 画像形状={batch['image'].shape}, "
              f"ラベル形状={batch['label'].shape}")
        if batch_idx >= 1:
            break
    
    return image_dataset, image_loader

class SimpleRegressor(nn.Module):
    """回帰用の簡単なモデル"""
    
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_with_custom_dataset(train_loader, test_loader):
    """カスタムデータセットを使った訓練例"""
    print("\n=== カスタムデータセットでの訓練 ===")
    
    # モデル作成
    model = SimpleRegressor(input_dim=5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 訓練
    epochs = 50
    train_losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in train_loader:
            features = batch['features']
            targets = batch['target']
            
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"エポック [{epoch+1}/{epochs}], 損失: {avg_loss:.4f}")
    
    # テスト
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            targets = batch['target']
            predictions = model(features)
            test_loss += criterion(predictions, targets).item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"テスト損失: {avg_test_loss:.4f}")
    
    # 損失の可視化
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses)
        plt.title('訓練損失の推移')
        plt.xlabel('エポック')
        plt.ylabel('損失')
        plt.grid(True)
        plt.savefig('custom_dataset_training.png', 
                    dpi=150, bbox_inches='tight')
        print("訓練損失を 'custom_dataset_training.png' に保存しました")
    except ImportError:
        print("matplotlib が利用できないため、可視化をスキップします")

def data_loading_tips():
    """データローディングのヒントとベストプラクティス"""
    print("\n=== データローディングのヒント ===")
    
    tips = [
        "1. num_workers を適切に設定してデータ読み込みを並列化",
        "2. pin_memory=True でGPU転送を高速化（GPU使用時）",
        "3. データセットが大きい場合は__getitem__で遅延読み込み",
        "4. データ変換は軽量に保ち、重い処理は事前に実行",
        "5. バッチサイズはメモリ使用量とのバランスを考慮",
        "6. シャッフルは訓練時のみ有効化",
        "7. 検証用データは固定順序で評価の再現性を確保"
    ]
    
    for tip in tips:
        print(tip)

def main():
    """メイン関数"""
    print("PyTorch カスタムデータセットチュートリアル")
    print("=" * 50)
    
    # CSVデータセット例
    csv_dataset, csv_loader = demonstrate_csv_dataset()
    
    # 配列データセット例
    train_loader, test_loader = demonstrate_array_dataset()
    
    # 画像データセット例
    image_dataset, image_loader = demonstrate_image_dataset()
    
    # カスタムデータセットでの訓練
    train_with_custom_dataset(train_loader, test_loader)
    
    # ヒント表示
    data_loading_tips()
    
    print("\nチュートリアル完了！")

if __name__ == "__main__":
    main() 
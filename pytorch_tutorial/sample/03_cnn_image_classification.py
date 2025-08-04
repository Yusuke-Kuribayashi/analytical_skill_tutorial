"""
PyTorch CNN画像分類チュートリアル
畳み込みニューラルネットワークを使った画像分類の実装
CIFAR-10データセットを使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from japanese_font_config import setup_japanese_font

# 日本語フォント設定
try:
    setup_japanese_font()
except ImportError:
    print("日本語フォント設定をスキップします")

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

class SimpleCNN(nn.Module):
    """シンプルなCNNモデル"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 畳み込み層
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # プーリング層
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全結合層
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # ドロップアウト
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 畳み込み + ReLU + プーリング
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # 平坦化
        x = x.view(-1, 128 * 4 * 4)
        
        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_data():
    """CIFAR-10データセットの読み込み"""
    print("=== データセット読み込み ===")
    
    # データ前処理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # データセット読み込み
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    
    # クラス名
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"訓練データ数: {len(trainset)}")
    print(f"テストデータ数: {len(testset)}")
    print(f"クラス数: {len(classes)}")
    print(f"クラス名: {classes}")
    
    return trainloader, testloader, classes

def visualize_samples(dataloader, classes, num_samples=8):
    """サンプル画像の可視化"""
    print("\n=== サンプル画像表示 ===")
    
    # バッチからサンプル取得
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # 正規化を元に戻す
    def denormalize(tensor):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        return tensor * std + mean
    
    try:
        # 画像表示
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = denormalize(images[i])
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0)
            
            axes[i].imshow(img)
            axes[i].set_title(f'クラス: {classes[labels[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', 
                    dpi=150, bbox_inches='tight')
        print("サンプル画像を 'sample_images.png' に保存しました")
    except ImportError:
        print("matplotlib が利用できないため、画像表示をスキップします")

def train_model(model, trainloader, epochs=10):
    """モデル訓練"""
    print(f"\n=== モデル訓練 (エポック数: {epochs}) ===")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 勾配をゼロにリセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 逆伝播
            loss.backward()
            optimizer.step()
            
            # 統計
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 進捗表示
            if (i + 1) % 100 == 0:
                print(f'エポック [{epoch+1}/{epochs}], '
                      f'ステップ [{i+1}/{len(trainloader)}], '
                      f'損失: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # エポック終了時の精度
        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(trainloader)
        
        train_accuracies.append(epoch_accuracy)
        train_losses.append(epoch_loss)
        
        print(f'エポック [{epoch+1}/{epochs}] 完了, 訓練精度: {epoch_accuracy:.2f}%')
    
    return train_losses, train_accuracies

def evaluate_model(model, testloader, classes):
    """モデル評価"""
    print("\n=== モデル評価 ===")
    
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # クラス別精度計算
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 全体精度
    overall_accuracy = 100 * correct / total
    print(f'テスト精度: {overall_accuracy:.2f}%')
    
    # クラス別精度
    print("\nクラス別精度:")
    for i in range(len(classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{classes[i]}: {accuracy:.2f}%')
    
    return overall_accuracy

def plot_training_history(losses, accuracies):
    """訓練履歴の可視化"""
    print("\n=== 訓練履歴の可視化 ===")
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失の推移
        ax1.plot(losses)
        ax1.set_title('訓練損失の推移')
        ax1.set_xlabel('エポック')
        ax1.set_ylabel('損失')
        ax1.grid(True)
        
        # 精度の推移
        ax2.plot(accuracies)
        ax2.set_title('訓練精度の推移')
        ax2.set_xlabel('エポック')
        ax2.set_ylabel('精度 (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', 
                    dpi=150, bbox_inches='tight')
        print("訓練履歴を 'training_history.png' に保存しました")
    except ImportError:
        print("matplotlib が利用できないため、可視化をスキップします")

def model_summary(model):
    """モデル構造の表示"""
    print("\n=== モデル構造 ===")
    print(model)
    
    # パラメータ数計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")

def main():
    """メイン関数"""
    print("PyTorch CNN画像分類チュートリアル")
    print("=" * 50)
    
    # データ読み込み
    trainloader, testloader, classes = load_data()
    
    # サンプル画像表示
    visualize_samples(trainloader, classes)
    
    # モデル作成
    model = SimpleCNN(num_classes=10).to(device)
    model_summary(model)
    
    # 訓練
    train_losses, train_accuracies = train_model(model, trainloader, epochs=5)
    
    # 評価
    test_accuracy = evaluate_model(model, testloader, classes)
    
    # 訓練履歴可視化
    plot_training_history(train_losses, train_accuracies)
    
    # モデル保存
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("訓練済みモデルを 'cnn_model.pth' に保存しました")
    
    print(f"\n最終テスト精度: {test_accuracy:.2f}%")
    print("チュートリアル完了！")

if __name__ == "__main__":
    main() 
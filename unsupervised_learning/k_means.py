'''
k-means法をnumpyで実装する例
'''
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. 学習を行うデータを生成
# -------------------------------------------------------------
X, y = make_blobs(n_samples=500, centers=3, cluster_std=3, random_state=10)
print("学習データ: ", X.shape)  # (500, 2)
print("正解データ: ", y.shape)  # (500,)

# -------------------------------------------------------------
# 2. k-means法のアルゴリズムを実装
# -------------------------------------------------------------
# k-means法のパラメータ
K = 3  # クラスタ数
max_iters = 100  # 最大イテレーション数
tol = 1e-4  # 収束判定の閾値

np.random.seed(42)  # 再現性のためのシード設定
centroids = X[np.random.choice(X.shape[0], K, replace=False)]
print("初期重心:\n", centroids)

for i in range(max_iters):
    # 各データポイントと3つの重心までの距離を計算
    distnces = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
    # 設定した３つの重心のうち最も近い重心を選択する
    labels = np.argmin(distnces, axis=1)
    # それぞれ同じクラスで重心を再計算
    new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
    # 重心の変化が小さい場合は終了
    if np.linalg.norm(new_centroids - centroids) < tol:
        break
    centroids = new_centroids
print("最終重心:\n", centroids)

# -------------------------------------------------------------
# 3. 結果を可視化
# -------------------------------------------------------------
# グラフの描画
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=30, label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()
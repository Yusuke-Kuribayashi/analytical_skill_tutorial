"""
- 決定木をnumpyのみで実装
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# -------------------------------------------------------------
# 1. 学習を行うデータを生成
# -------------------------------------------------------------
X, y = make_blobs(n_samples=500, centers=3, cluster_std=3, random_state=10)

# -------------------------------------------------------------
# 2. CARTアルゴリズムを実装
# -------------------------------------------------------------
# ノード内の情報を保持するクラス
# is_leaf: ノードが葉ノードかどうか(末端か否か)
# pred: ノードの予測値(ノード内にいるサンプル)
# feat: 分割に使用する特徴量のインデックス
# thr: 分割に使用する閾値
# left: 左の子ノード
# right: 右の子ノード
class Node:
    # タイポミス・メモリ使用量を防ぐ
    __slots__ = ("is_leaf", "pred", "feat", "thr", "left", "right")
    def __init__(self):
        self.is_leaf = False
        self.pred = None
        self.feat = None
        self.thr = None
        self.left = None
        self.right = None

# ジニ不純度を計算する関数
# ジニ不純度は、クラスの分布の不均一性を測る指標で、値が小さいほど純度が高い
def gini(counts) -> float:
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return 1.0 - np.sum(p * p)

# 最適な分割を見つける関数
# X: 特徴量行列
# y: ラベル
# idxs: インデックス
# n_classes: クラス数
# min_leaf: ノードが保持すべき最小サンプル数
def find_best_split(X, y, idxs, n_classes, min_leaf):
    best_gain, best_feat, best_thr = 0.0, None, None
    # 親ノードの情報を取得
    # 現在のノードの中に、n_classes数のクラスがある場合、各クラスのサンプル数をカウント
    parent_counts = np.bincount(y[idxs], minlength=n_classes) # (n_classes,)
    # 親ノードのジニ不純度を計算
    parent_imp = gini(parent_counts)
    # 親ノードの中にあるサンプル数, 特徴量の数を取得
    m, n_feats = idxs.size, X.shape[1]

    # 特徴量ごとに最適な分割を探す
    # その中で、一番ジニ不純度が小さい分割を選択する
    for feat in range(n_feats):
        # 特徴量を取得
        vals = X[idxs, feat]
        # 今回検証する特徴量を昇順にソートしたときのインデックスを取得
        order = vals.argsort()
        # 元データのインデックスでの値
        sorted_idx = idxs[order]
        # ラベルを昇順にソート(yはサンプル全体で保持されているので、sorted_idxに基づいてソート)
        sorted_y = y[sorted_idx]
        # 特徴量の値を昇順にソート
        sorted_vals = vals[order]

        left_counts = np.zeros(n_classes, dtype=int)
        right_counts = parent_counts.copy()

        for i in range(1, m):
            c = sorted_y[i-1]
            left_counts[c] += 1
            right_counts[c] -= 1

            # 分割点を決定するために、隣接する値が異なる場合のみ考慮
            if sorted_vals[i] == sorted_vals[i-1]:
                continue
            # 最小葉サイズの条件を満たさない場合はスキップ
            if left_counts.sum() < min_leaf or right_counts.sum() < min_leaf:
                continue

            # 親ノードのジニ不純度から、分割後のジニ不純度を引いて情報利得を計算
            gain = parent_imp - (
                left_counts.sum() * gini(left_counts) +
                right_counts.sum() * gini(right_counts)
            ) / m

            # 情報利得が最大の分割を選択
            # gainが最大の分割を見つけた場合、best_gain, best_feat, best_thrを更新
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thr = (sorted_vals[i] + sorted_vals[i-1]) * 0.5

    return best_feat, best_thr, best_gain

# X: 特徴量行列
# y: ラベル
# idxs: インデックス
# depth: 現在の深さ
# max_depth: 木の最大深さ
# min_leaf: ノードが保持すべき最小サンプル数
# n_classes: クラス数
def build_tree(X, y, idxs, depth, max_depth, min_leaf, n_classes):
    # 新たなノードを作成
    node = Node()
    counts = np.bincount(y[idxs], minlength=n_classes)
    # 現在のノードの中で一番多いクラスを予測値として設定
    node.pred = counts.argmax()

    # 停止条件を確認(深さが最大深さに達した、またはノード内のサンプル数が最小葉サイズを満たしている場合)
    if depth >= max_depth or counts.max() == idxs.size:
        node.is_leaf = True
        return node

    # 最適な分割を見つける
    feat, thr, gain = find_best_split(X, y, idxs, n_classes, min_leaf)

    # featがNoneまたはgainが0の場合、ノードを葉ノードとして設定(これ以上分割できない)
    if feat is None or gain == 0.0:
        node.is_leaf = True
        return node

    # どの特徴で分割するか、どの値で分割するかを設定
    node.feat, node.thr = feat, thr

    # 左: その特徴の値が閾値(thr)より小さいサンプル
    # 右: その特徴の値が閾値(thr)以上のサンプル
    left_mask = X[idxs, feat] < thr
    left_idxs = idxs[left_mask]
    right_idxs = idxs[~left_mask]

    # 左右のノードに再帰的に分割を行う
    node.left = build_tree(X, y, left_idxs, depth+1, max_depth, min_leaf, n_classes)
    node.right = build_tree(X, y, right_idxs, depth+1, max_depth, min_leaf, n_classes)
    return node

# 一サンプルデータを予測する関数
def predict_one(x, node):
    # 葉ノード(末端)まで再帰的に処理を実行
    while not node.is_leaf:
        # 特徴量の値が閾値(thr)より小さい場合は左の子ノードへ、そうでない場合は右の子ノードへ移動
        node = node.left if x[node.feat] < node.thr else node.right
    # 葉ノードに到達したら、そのノードの予測値(クラス)を返す
    return node.pred

def predict(X, root):
    return np.array([predict_one(x, root) for x in X])

# 木を作成
n_classes = len(np.unique(y))
idxs = np.arange(y.shape[0])
root = build_tree(X, y, idxs, depth=0, max_depth=4, min_leaf=5, n_classes=n_classes)

# 評価
y_pred = predict(X, root)
accuracy = (y_pred == y).mean()
print(f"Training accuracy: {accuracy:.3f}")

# -------------------------------------------------------------
# 3. 決定機の分割領域を確認
# -------------------------------------------------------------
# Create a mesh grid
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = predict(grid, root).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes+1)-0.5)
scatter = plt.scatter(X[:,0], X[:,1], c=y, cmap='tab10', edgecolor='k', s=30)
plt.title("Decision Tree - Training accuracy {:.2f}".format(accuracy))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# -------------------------------------------------------------
# 1. Generate data (same settings as user)
# -------------------------------------------------------------
X, y = make_blobs(n_samples=500, centers=3, cluster_std=3, random_state=10)

# -------------------------------------------------------------
# 2. Implement a minimal CART decision tree (NumPy only)
# -------------------------------------------------------------
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

def gini(counts):
    p = counts / counts.sum()
    return 1.0 - np.sum(p * p)

def find_best_split(X, y, idxs, n_classes, min_leaf):
    best_gain, best_feat, best_thr = 0.0, None, None
    parent_counts = np.bincount(y[idxs], minlength=n_classes)
    parent_imp = gini(parent_counts)
    m, n_feats = idxs.size, X.shape[1]

    for feat in range(n_feats):
        # sort by feature
        vals = X[idxs, feat]
        order = vals.argsort()
        sorted_idx = idxs[order]
        sorted_y = y[sorted_idx]
        sorted_vals = vals[order]

        left_counts = np.zeros(n_classes, dtype=int)
        right_counts = parent_counts.copy()

        for i in range(1, m):
            c = sorted_y[i-1]
            left_counts[c] += 1
            right_counts[c] -= 1

            if sorted_vals[i] == sorted_vals[i-1]:
                continue
            if left_counts.sum() < min_leaf or right_counts.sum() < min_leaf:
                continue

            gain = parent_imp - (
                left_counts.sum() * gini(left_counts) +
                right_counts.sum() * gini(right_counts)
            ) / m

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thr = (sorted_vals[i] + sorted_vals[i-1]) * 0.5

    return best_feat, best_thr, best_gain

def build_tree(X, y, idxs, depth, max_depth, min_leaf, n_classes):
    node = Node()
    counts = np.bincount(y[idxs], minlength=n_classes)
    node.pred = counts.argmax()

    # stop conditions
    if depth >= max_depth or counts.max() == idxs.size:
        node.is_leaf = True
        return node

    feat, thr, gain = find_best_split(X, y, idxs, n_classes, min_leaf)
    if feat is None or gain == 0.0:
        node.is_leaf = True
        return node

    node.feat, node.thr = feat, thr
    left_mask = X[idxs, feat] < thr
    left_idxs = idxs[left_mask]
    right_idxs = idxs[~left_mask]

    node.left = build_tree(X, y, left_idxs, depth+1, max_depth, min_leaf, n_classes)
    node.right = build_tree(X, y, right_idxs, depth+1, max_depth, min_leaf, n_classes)
    return node

def predict_one(x, node):
    while not node.is_leaf:
        node = node.left if x[node.feat] < node.thr else node.right
    return node.pred

def predict(X, root):
    return np.array([predict_one(x, root) for x in X])

# Build tree
n_classes = len(np.unique(y))
idxs = np.arange(y.shape[0])
root = build_tree(X, y, idxs, depth=0, max_depth=4, min_leaf=5, n_classes=n_classes)

# Evaluate
y_pred = predict(X, root)
accuracy = (y_pred == y).mean()
print(f"Training accuracy: {accuracy:.3f}")

# -------------------------------------------------------------
# 3. Visualize decision regions
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
plt.title("Decision Tree (NumPy only) - Training accuracy {:.2f}".format(accuracy))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

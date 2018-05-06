from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=10000,
                  n_features=3,
                  centers=15)

print(X, y)


tree = KDTree(X, leaf_size=40, metric='minkowski')
dist, ind = tree.query([X[0]], k=3)

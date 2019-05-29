from sklearn.neighbors import KDTree
import numpy as np
import pandas
from matplotlib import pyplot as plt


class DBSCAN:
    """
    Implementation of dbscan clustering algorithm.
    """
    NOISE = -1
    UNDEFINED = -2

    def __init__(self, min_samples=4, eps=0.1, metric="euclidean"):
        """
        saves parameters for clustering
        :param min_samples: minimal number of samples in neighborhood to consider it dense
        :param eps: neighborhood size
        :param metric: distance metric to use
        """
        self.min_samples = min_samples
        self.eps = eps
        self.metric = metric

    def fit_predict(self, X):
        """
        Runs clustering.
        :param X: Data to cluster.
        :return: Cluster indices for each sample (-1 for noise)
        """
        c = -1
        n = X.shape[0]
        labels = np.ones(n, dtype=np.int32) * self.UNDEFINED
        tree = KDTree(X, metric=self.metric)
        for point in range(n):
            if labels[point] == self.UNDEFINED:
                neigh_idxs = tree.query_radius(X[np.newaxis, point, :], self.eps)
                if neigh_idxs[0].shape[0] < self.min_samples:
                    labels[point] = self.NOISE
                else:
                    c += 1
                    labels[point] = c
                    neigh_idxs = list(neigh_idxs[0])
                    all_neighs = set(neigh_idxs)
                    for neighbor in neigh_idxs:
                        if labels[neighbor] == self.NOISE:
                            labels[neighbor] = c
                        elif labels[neighbor] == self.UNDEFINED:
                            labels[neighbor] = c
                            to_add = tree.query_radius(X[np.newaxis, neighbor, :], self.eps)
                            if to_add[0].shape[0] >= self.min_samples:
                                to_add = set(to_add[0]) - all_neighs
                                all_neighs.update(to_add)
                                neigh_idxs.extend(to_add)
        return labels


def k_dist(X, metric="euclidean", k=4):
    """
    Calculates sorted vector of distances to k-th closest neighbor for each input sample
    :param X: Input data
    :param metric: Distance metric to use
    :param k: distance to which closest neighbor to calculate
    :return: sorted distances (largest first)
    """
    tree = KDTree(X, metric=metric)
    dist, _ = tree.query(X, k + 1)
    return np.array(sorted([i[-1] for i in dist], reverse=True))


def vizualization():
    """
    Vizualization of k-dist and clustering on some data.
    """
    X = np.array(pandas.read_csv("dbscan-paintedData.csv", sep="\t"))
    plt.figure()
    plt.subplot(2, 1, 1)
    for k in [1, 3, 15]:
        dists = k_dist(X, k=k)
        plt.plot(dists, label="k=%d" % k)
    plt.legend()
    plt.xlabel("i-ti primer")
    plt.ylabel("razdalja")
    # plt.show()
    plt.subplot(2, 1, 2)
    dbs = DBSCAN(3, 0.07)
    clusters = dbs.fit_predict(X)
    classes = np.unique(clusters)
    for cls in classes:
        mask = clusters == cls
        plt.scatter(X[mask, 0], X[mask, 1], 10, label="Noise" if cls == -1 else cls)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    vizualization()

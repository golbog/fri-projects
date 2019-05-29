import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from matplotlib import cm
from sklearn.decomposition import PCA


class DBSCAN:
    """
    Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors for DBSCAN.
    """

    def __init__(self, min_samples=5, eps=4, metric='euclidean'):
        self.min_samples = min_samples;
        self.eps = eps;
        self.metric = metric;

    def fit_predict(self, data):
        """
        Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels
        """
        cluster_counter = 0
        NOISE = -1
        UNDEFINED = -9

        labels = np.full(len(data), UNDEFINED, dtype=int)
        tree = KDTree(data, metric=self.metric, leaf_size=10)

        for index_row, row in enumerate(data):
            if labels[index_row] != UNDEFINED:
                continue

            neighbors = list(tree.query_radius([row], r=self.eps)[0])

            if len(neighbors) < self.min_samples:
                labels[index_row] = NOISE
                continue

            labels[index_row] = cluster_counter

            for neighbor_index in neighbors:
                if labels[neighbor_index] == NOISE:
                    labels[neighbor_index] = cluster_counter

                if labels[neighbor_index] != UNDEFINED:
                    continue

                labels[neighbor_index] = cluster_counter

                add_neighbors = list(tree.query_radius([data[neighbor_index]], r=self.eps)[0])
                if len(add_neighbors) >= self.min_samples:
                    new_add_neighbors = [x for x in add_neighbors if x not in neighbors]
                    neighbors.extend(new_add_neighbors)

            cluster_counter += 1

        return labels


def k_dist(X, metric='euclidean', k=5):
    """
    For each point calculates distances to its 'k' nearest neighbor.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors for DBSCAN.
    k : int
        Which nearest neighbor we calculate distance

    Returns
    -------
    dist : ndarray, of shape (n_samples,)
        array of distances to k-th neighbor
    """
    tree = KDTree(X, metric=metric, leaf_size=2)
    res = list()
    for x in X:
        distance, neighbors = tree.query([x], k=k + 1)
        zipped = sorted(zip(neighbors[0], distance[0]), key=lambda a: a[1])
        [_, kth_distance] = zipped[-1]
        res.append(kth_distance)
    return np.array(sorted(res))


def plot_kdist(X, k):
    """
    Plot k-dist graph.

    Parameters
    ----------
    X : array of distances to k-th neighboor (n_samples, )
    k : int
        Which nearest neighbor we used to calculate distances
    """
    plt.plot(X)
    plt.title('k = ' + str(k))
    plt.ylabel('Distance')
    plt.xlabel('Point')

    plt.savefig('kdist' + str(k) + '.pdf')
    plt.show()


def plot_DBSCAN(X, clusters, min_samples, eps):
    """
    Plot scatter plot of DBSCAN clustering.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    clusters : array of shape (n_samples, )
        Labels of clusters for each point
    minsamples : int
        Parameter min_samples that was used for clustering.
    eps : float
        Parameter eps that was used for clustering.
    """
    colors = cm.rainbow(np.linspace(0, 1, 1 + len(np.unique(clusters))))
    for cluster in np.unique(clusters):
        filtered = X[clusters == cluster, :]
        plt.scatter(filtered[:, 0], filtered[:, 1], s=3, c=colors[cluster], alpha=1, label=str(cluster))

    plt.title('DBSCAN clustering [eps=' + str(eps) + ', min_samples=' + str(min_samples) + ']')
    plt.ylabel('Y')
    plt.xlabel('X')

    legend = plt.legend()
    for handle in legend.legendHandles:
        handle.set_sizes([7])

    plt.savefig('dbscan.pdf')
    plt.show()


if __name__ == '__main__':
    # Example of usage
    digits = sklearn.datasets.load_digits().data[:500]
    pca = PCA(n_components=2)
    data = pca.fit_transform(digits)

    data = sklearn.datasets.load_iris().data[:, (2, 3)]

    k1 = k_dist(data, k=3)
    k2 = k_dist(data, k=5)
    k3 = k_dist(data, k=7)

    plot_kdist(k1, 3)
    plot_kdist(k2, 5)
    plot_kdist(k3, 7)

    samples = 4
    eps = 0.2

    dbscan = DBSCAN(min_samples=samples, eps=eps)
    clusters = dbscan.fit_predict(data)

    plot_DBSCAN(data, clusters, samples, eps)

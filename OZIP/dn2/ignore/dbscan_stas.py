import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from matplotlib import cm
from sklearn.decomposition import PCA


def k_dist(X, k, metric="euclidean"):
    """For each point calculates distances to its 'k' nearest neighbor.

    Parameters
    ----------
    X : array of shape (n_samples, n_features), or array of shape (n_samples, n_samples)
    k : int
        For which nearest neighbor to calculate distance.
    metric : string, optional
        The metric to use when calculating distance between instances in a
        feature array.

    Returns
    -------
    dist : array of shape (n_samples,)
        array of distances to 'k' neighbor
    """
    tree = KDTree(X, metric=metric)
    dist, ind = tree.query(X, k=k, return_distance=True)
    return sorted(dist[:, :1])


class DBSCAN:
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Implementation is based on the article "A Density-Based Algorithm
    for Discovering Clusters in Large Spatial Databases with Noise".

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    metric : string, optional
        The metric to use when calculating distance between instances in a
        feature array.
    """

    NOISE = -1
    UNDEFINED = -2

    def __init__(self, eps=0.1, min_samples=4, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._tree = None

    def fit_predict(self, X):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array of shape (n_samples, n_features),
            or array of shape (n_samples, n_samples)

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            cluster labels
        """
        tree = KDTree(X, metric=self.metric)

        c = 0  # cluster counter
        labels = np.full((X.shape[0], 1), self.UNDEFINED)
        
        for p_ind, p in enumerate(X):
            if labels[p_ind] != self.UNDEFINED:
                continue

            p_neighbor_inds = list(tree.query_radius([p], r=self.eps)[0])
            if len(p_neighbor_inds) < self.min_samples:
                labels[p_ind] = self.NOISE
                continue

            labels[p_ind] = c

            for q_ind in p_neighbor_inds:
                if q_ind == p_ind:  # skip p itself
                    continue

                if labels[q_ind] == self.NOISE:
                    labels[q_ind] = c
                if labels[q_ind] != self.UNDEFINED:
                    continue

                labels[q_ind] = c

                q = X[q_ind]
                q_neighbor_inds = list(tree.query_radius([q], r=self.eps)[0])

                if len(q_neighbor_inds) >= self.min_samples:
                    q_neighbor_inds = [x for x in q_neighbor_inds if x not in p_neighbor_inds]
                    p_neighbor_inds.extend(q_neighbor_inds)

            c += 1

        return labels.flatten()

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
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(clusters))))
    #colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c','#fabebe', '#008080', '#aa6e28']
    for cluster in np.unique(clusters):
        filtered = X[clusters == cluster, :]
        print(len(filtered))
        plt.scatter(filtered[:, 0], filtered[:, 1], s=3, c=colors[cluster], alpha=1, label=str(cluster))

    plt.title('stas DBSCAN clustering [eps=' + str(eps) + ', min_samples=' + str(min_samples) + ']')
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
    eps = .1

    dbscan = DBSCAN(min_samples=samples, eps=eps)
    clusters = dbscan.fit_predict(data)

    print(clusters)

    plot_DBSCAN(data, clusters, samples, eps)
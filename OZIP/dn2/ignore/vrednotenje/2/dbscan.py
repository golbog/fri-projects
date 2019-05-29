import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfs
from sklearn import datasets
from scipy.spatial.distance import cdist
from sklearn.neighbors.kd_tree import KDTree


def k_dist_plot(data, k):
    """
    Plots values of k_dist function according to different values of k.

    :param data: data to plot
    :param k: list of different ks
    :return: instance of matplotlib.pyplot.figure and subplots
    """
    colors = ['r', 'g', 'b']
    fig, (plt_1, plt_2) = plt.subplots(nrows=2, figsize=(15, 20))
    for i in range(len(k)):
        vec = k_dist(data, k[i])
        plt_1.plot(vec, colors[i], label='k = ' + str(k[i]))
    plt_1.set_xlabel('data point')
    plt_1.set_ylabel('distance to k-th nearest neighbour')
    plt_1.set_title('k_dist function')
    plt_1.legend()
    return fig, (plt_1, plt_2)


def dbscan_plot(data, plt_2, eps):
    """
    Scatter plot of two-attribute data with DBSCAN algorithm.

    :param data: data to plot
    :param plt_2: bottom part of the plot
    :return: None
    """
    dbscan = DBSCAN(eps=eps)
    group_indices = dbscan.fit_predict(data)
    number_of_clusters = len(np.unique(group_indices[group_indices >= 0]))
    noise = np.where(group_indices == -1)[0]
    noise_mask = np.array([(x in noise) for x in range(len(group_indices))])
    group_points = group_indices[~noise_mask]

    distinct_colors = ["#0082c8", "#3cb44b", "#e6194b", "#ffe119", "#f58231",
                       "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#008080"]
    colors = list(map(lambda x: distinct_colors[x], group_points))

    plt_2.scatter(data[~noise_mask, 0], data[~noise_mask, 1], marker='.', c=colors)
    plt_2.scatter(data[noise_mask, 0], data[noise_mask, 1], marker='x', s=12, color='black')
    plt_2.set_xlabel('x1')
    plt_2.set_ylabel('x2')
    plt_2.set_title('DBSCAN (min_samples=' + str(dbscan.min_samples) + ', eps=' + str(eps) +
                    ') --- Number of clusters found: ' + str(number_of_clusters))
    # plt.show()
    return None


def k_dist(X, k, metric='euclidean'):
    """
    Returns ordered vector of distances to k-th neighbour, where the example itself
    (for which we observe neighbourhood) is excluded from the list of neighbours.

    :param X: data
    :param k: k-th closest neighbour
    :param metric: distance measure
    :return: vector of distances
    """
    distances = cdist(X, X, metric)
    vector = np.zeros(X.shape[0])
    for i, point_distances in enumerate(distances):
        point_distances.sort()
        vector[i] = point_distances[k]
    return sorted(vector, reverse=True)


class DBSCAN:
    def __init__(self, min_samples=4, eps=0.1, metric='euclidean'):
        """
        :param min_samples: DBSCAN algorithm parameter
        :param eps: DBSCAN algorithm parameter
        :param metric: distance measure
        """
        self.min_samples = min_samples
        self.eps = eps
        self.metric = metric

    def fit_predict(self, X):
        """
        Runs DBSCAN algorithm on data.
        Returns vector with indices of groups where examples belong (starting with index 0).
        Element value is -1, when example represents noise or does not belong to any group.

        :param X: data
        :return: vector with group indices
        """
        labels = np.full(X.shape[0], -99)      # -99 means undefined
        c = -1
        for index, _ in enumerate(X):
            if labels[index] != -99:
                continue

            neighbours = self.eps_neighbourhood(X, index, self.eps, self.metric)
            if len(neighbours) < self.min_samples:
                labels[index] = -1
                continue

            c += 1
            labels[index] = c
            S = list(set(neighbours).difference({index}))
            while S:
                q = S.pop()
                if labels[q] == -1:
                    labels[q] = c
                if labels[q] != -99:
                    continue
                labels[q] = c
                neighbours = self.eps_neighbourhood(X, q, self.eps, self.metric)
                if len(neighbours) >= self.min_samples:
                    S.extend(neighbours)

        return labels

    @staticmethod
    def eps_neighbourhood(X, index, eps, metric):
        """
        Query for neighbors within a given radius.

        :param X: data
        :param index: index position of point in data
        :param eps: looking for points inside radius eps
        :param metric: distance metric
        :return: vector of indices
        """
        tree = KDTree(X, leaf_size=2, metric=metric)
        indices = tree.query_radius([X[index]], r=eps)
        return indices[0]


if __name__ == "__main__":
    dataset, _ = datasets.make_moons(n_samples=2000, noise=0.07)

    fig, (_, plt_2) = k_dist_plot(dataset, [3, 8, 15])   # k_dist_graph
    dbscan_plot(dataset, plt_2, 0.07)   # scatter plot

    plt.subplots_adjust(hspace=0.3)
    pdf = pdfs.PdfPages("plots.pdf")
    pdf.savefig(fig)
    pdf.close()

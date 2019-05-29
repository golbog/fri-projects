from sklearn.neighbors import KDTree
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA


class DBSCAN:
    def __init__(self, min_samples=5, eps=4, metric='euclidean'):
        self.min_samples = min_samples;
        self.eps = eps;
        self.metric = metric;

    def fit_predict(self, data):
        cluster_counter = 0
        labels = dict()
        tree = KDTree(data, metric=self.metric, leaf_size=10)

        for index_row, row in enumerate(data):

            print(row[0], row[1])

            if row[0] == 5.2 and row[1] == 2.3:
                print()

            if index_row in labels:
                continue

            neighbors = list(tree.query_radius([row], r=self.eps)[0])

            if len(neighbors) < self.min_samples:
                labels[index_row] = -1
                continue

            labels[index_row] = cluster_counter

            for neighbor_index in neighbors:
                if neighbor_index in labels and labels[neighbor_index] == -1:
                    labels[neighbor_index] = cluster_counter

                if neighbor_index in labels and not :
                    continue

                labels[neighbor_index] = cluster_counter

                add_neighbors = list(tree.query_radius([data[neighbor_index]], r=self.eps)[0])
                if len(add_neighbors) >= self.min_samples:
                    new_add_neighbors = [x for x in add_neighbors if x not in neighbors]
                    neighbors.extend(new_add_neighbors)

            cluster_counter += 1

        res = np.array([labels[key] for key in sorted(labels)])
        return res

    def _range_query(self, data, main):
        neighbors = list()
        for row in data:
            if self.metric(main, row) < self.eps:
                neighbors.append(row)
        return neighbors


def k_dist(X, metric='euclidean', k=5):
    tree = KDTree(X, metric=metric, leaf_size=2)
    res = list()
    for x in X:
        distance, neighbors = tree.query([x], k=k + 1)
        zipped = sorted(zip(neighbors[0], distance[0]), key=lambda a: a[1])
        [_, kth_distance] = zipped[-1]
        res.append(kth_distance)
    return np.array(sorted(res))


def plot_kdist(X, k):
    plt.plot(X)
    plt.title('k = ' + str(k))
    plt.ylabel('Distance')
    plt.xlabel('Point')

    plt.savefig('kdist' + str(k) + '.pdf')
    plt.show()


def plot_DBSCAN(X, clusters, minsamples, eps):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = cm.rainbow(np.linspace(0, 1, 1 + len(np.unique(clusters))))
    # for line, cluster in zip(X, clusters):
    #    ax.scatter(line[0], line[1], color=colors[cluster], alpha=1, s=3, label=str(cluster))

    for cluster in np.unique(clusters):
        filtered = X[clusters == cluster, :]
        plt.scatter(filtered[:, 0], filtered[:, 1], s=3, c=colors[cluster], alpha=1, label=str(cluster))

    plt.title('DBSCAN clustering [eps=' + str(eps) + ', min_samples=' + str(minsamples) + ']')
    plt.ylabel('Y')
    plt.xlabel('X')

    legend = plt.legend()
    for handle in legend.legendHandles:
        handle.set_sizes([7])

    plt.savefig('dbscan.pdf')
    plt.show()


if __name__ == '__main__':
    digits = sklearn.datasets.load_digits().data[:500]
    pca = PCA(n_components=2)
    data = pca.fit_transform(digits)

    data = sklearn.datasets.load_iris().data[:, (2, 3)]

    k1 = k_dist(data, k=3)
    k2 = k_dist(data, k=5)
    k3 = k_dist(data, k=7)

    # plot_kdist(k1, 3)
    # plot_kdist(k2, 5)
    # plot_kdist(k3, 7)

    samples = 4
    eps = 0.2

    dbscan = DBSCAN(min_samples=samples, eps=eps)
    clusters = dbscan.fit_predict(data)
    print(clusters)

    plot_DBSCAN(data, clusters, samples, eps)
    print(k1)

    pass

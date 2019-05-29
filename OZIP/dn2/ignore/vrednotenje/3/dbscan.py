import sklearn.neighbors
import numpy as np
from matplotlib import pyplot


def k_dist(X, metric='euclidean', k=4):
    tree = sklearn.neighbors.KDTree(X, metric=metric)
    k_distances = [tree.query([X[i]], k=k + 1)[0][0][k] for i in range(len(X))]
    return np.sort(k_distances)[::-1]


class DBSCAN(object):
    min_samples = 4
    eps = 0.1
    metric = 'euclidean'

    def __init__(self, min_samples=4, eps=0.1, metric='euclidean'):
        self.min_samples = min_samples
        self.eps = eps
        self.metric = metric

    def fit_predict(self, X):
        c = 0
        tree = sklearn.neighbors.KDTree(X, metric=self.metric)
        labels = np.repeat(-2, len(X))
        for i in range(len(X)):
            if labels[i] != -2:
                continue
            indices = tree.query_radius([X[i]], r=self.eps)[0]
            if len(indices) < self.min_samples:
                labels[i] = -1
                continue
            c += 1
            labels[i] = c
            s = [x for x in indices if x != i]
            while len(s):
                q = s.pop(0)
                if labels[q] == -1:
                    labels[q] = c
                if labels[q] != -2:
                    continue
                labels[q] = c
                new_indices = tree.query_radius([X[q]], r=self.eps)[0]
                if len(new_indices) >= self.min_samples:
                    s.extend(new_indices)
        return labels


if __name__ == '__main__':
    data = np.loadtxt('Aggregation.txt', delimiter='\t', usecols=(0, 1))
    for idx, k in enumerate([3, 4, 5]):
        ax = pyplot.subplot(2, 3, idx + 1)
        pyplot.plot(k_dist(data, k=k), 'o')
        ax.set_title('K-dist, k=%d' % k)
    pyplot.subplot(212)
    labels = DBSCAN(min_samples=4, eps=1.1).fit_predict(data)
    pyplot.scatter(data[:, 0], data[:, 1], c=labels)
    pyplot.show()

import Orange
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import time
import numpy as np
import sklearn.neighbors as sknei
from matplotlib.backends.backend_pdf import PdfPages


class DBSCAN:
    def __init__(self, min_samples=4, eps=0.1, metric='euclidean'):
        """
        Initialize DBSCAN
        :param min_samples: (integer) minimal number of neighbours
                            within distance eps for point to become a core
        :param eps: float (min distance)
        :param metric: string (type of metric used for measuring distance)
        """
        self.min_samples = min_samples
        self.eps = eps
        self.metric = metric

    def fit_predict(self, X):
        """
        Performs clustering on data X and returns cluster labels
        :param X: array with data
        :return: array of cluster labels
        """
        print("# Fit predict ( eps = ", self.eps, ", min_samples = ", self.min_samples, ") ...")
        start_time = time.time()

        clusters = 0
        labels = [0] * len(X)
        tree = sknei.KDTree(X)
        for i in range(len(X)):
            if labels[i] > 0:  # point is already marked
                continue

            # get indexes of X[i] neighbours within the distance eps
            if self.metric == 'euclidean':
                N = tree.query_radius([X[i]], r=self.eps)[0]
            if len(N) < self.min_samples:  # mark as a noise
                labels[i] = -1
                continue
            clusters += 1
            labels[i] = clusters
            index = 0
            while index < len(N):  # over all neighbours of point X[i]
                if labels[N[index]] > 0:  # neighbour is already in the cluster
                    index += 1
                    continue
                labels[N[index]] = clusters

                # get indexes of X[N[index]] neighbours within the distance eps
                if self.metric == 'euclidean':
                    Nq = tree.query_radius([X[N[index]]], r=self.eps)[0]
                if len(Nq) >= self.min_samples:
                    for n in Nq:
                        if n not in N:
                            N = np.append(N, n)
                index += 1

        print("#", clusters, " clusters found.")
        print("# Fit predict finished in ", (time.time() - start_time), " sec.")
        return np.array(labels)

    def k_dist(self, X, metric='euclidean', k=2):
        """
        Calculates distances to k-th closest neighbour for all cases
        :param X: array with data
        :param metric: type of metric used for measuring distances
        :param k: number of neighbours
        :return: sorted array of distances to k-th closest neighbour
        """
        print("# Calculating distances to k-th closest neighbour ...")
        tree = sknei.KDTree(X)
        dists = []
        for i in range(len(X)):
            if metric == 'euclidean':
                dist, ind = tree.query([X[i]], k=(k + 1), return_distance=True)
                dists.append(dist[0][k])

        print("# Distances successfully calculated.")
        return np.array(sorted(dists, reverse=True))


def load_data(file_name):
    """
    Loads data from the file with name 'filename'
    :param file_name (string)
    :return: data (array)
    """
    data = Orange.data.Table(file_name)
    return data


def plot_clusters(X, labels, title, name):
    """
    Plots data with points colored according to their cluster label
    :param X: array of data
    :param labels: array of labels
    :param title: plot title
    :param name: name of input data
    """
    print("# Plotting data ...")
    pdf = PdfPages(name + '_clusters.pdf')
    fig = plt.figure()

    colours = get_cmap(len(set(labels)) + 1)    # get colors
    for i in range(len(X)):
        if labels[i] > 0:
            plt.scatter(X[i][0], X[i][1], 5, color=colours(labels[i]))
        elif labels[i] == -1:  # noise
            plt.scatter(X[i][0], X[i][1], 5, color='black')
    plt.title(title)

    # add legend
    patches = [mpatches.Patch(color='black', label='-1')]
    for i in range(1, len(set(labels)) + 1):
        patches.append(mpatches.Patch(color=colours(i), label=str(i)))
    plt.legend(handles=patches)

    plt.xlabel('x')
    plt.ylabel('y')
    pdf.savefig(fig)
    plt.clf()
    pdf.close()
    print("# Data successfully plotted")


def plot_k_dist(k_dist, k):
    """
    Displays plot with distances
    :param k_dist: array of distances to k-th closest vector
    :param k: cluster id
    """
    x = list(range(len(k_dist)))
    plt.plot(x, k_dist, label="k = " + str(k))


def calculate_k_dists(X, name):
    """
    Plots k_distances for different k-s
    :param X: array of data
    :param name: name of input data
    """
    pdf = PdfPages(name + '_graphs.pdf')
    fig = plt.figure()

    for i in [4, 10, 25]:   # for different k-s
        k_dists = dbscan.k_dist(X, 'euclidean', i)
        plot_k_dist(k_dists, i)
    plt.title('k_dist')
    plt.xlabel('n')
    plt.ylabel('dist')
    plt.legend()
    pdf.savefig(fig)
    plt.clf()
    pdf.close()


def plot_data(X):
    """
    Plots data
    :param X: array of data
    """
    X = np.array(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def get_cmap(n, name='rainbow'):
    """
    Returns n different colours
    src: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    :param n: number of colours
    :param name: type
    :return: n different colours
    """
    return plt.cm.get_cmap(name, n)


if __name__ == "__main__":
    data_name = 'sample_data'
    data = load_data(data_name + '.tab')
    # data_name = 'iris'
    # data = sklearn.datasets.load_iris().data[:, (2, 3)]

    eps_ = 0.04
    min_samples_ = 4

    # perform dbscan
    dbscan = DBSCAN(eps=eps_, min_samples=min_samples_)
    labels = dbscan.fit_predict(data)

    # plot k_dists for different k-s
    calculate_k_dists(data, data_name)

    # plot clusters
    title = "Clusters = " + str(len(set(labels) - {-1})) + ", eps = " + str(eps_) \
            + ", min_samples = " + str(min_samples_) + ")"
    plot_clusters(data, labels, title, data_name)


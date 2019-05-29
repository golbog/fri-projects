import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


class PCA:
    """
    Principal component analysis (PCA)

    Parameters
    ----------
    n_components : int
        Number of components to keep.

    Attribues
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        'explained_variance_'.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If 'n_components' is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0).

    n_components_ : int
        The estimated number of components.

    """

    def __init__(self, n_components=2):
        self.n_components_ = n_components
        self.components_ = list()  # lastni vektorji
        self.explained_variance_ = np.empty(n_components)  # lastne vrednosti
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def covariance(self, X):
        """
        Returns covariance matrix of the matrix X.

        Parameters
        ----------
        X : Data matrix of shape [n_examples, n_features]

        Returns
        -------
        X_new : (np.ndarray): Array of shape [n_features, n_features]
        """
        self.mean_ = X.mean(axis=0)
        X_centered = np.array(X - X.mean(axis=0))
        return X_centered.T.dot(X_centered) / X.shape[0]

    def transform(self, X):
        """
        Apply dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X = X - self.mean_
        return X.dot(self.components_.T)


class EigenPCA(PCA):
    def fit(self, X):
        """
        Fit the model with X.

        Parameters
        ----------
        X : arrays (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        eighval, eighvec = np.linalg.eig(self.covariance(X))
        self.components_ = eighvec.T[:self.n_components_]
        self.explained_variance_ = eighval[:self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        return self


class PowerPCA(PCA):
    def fit(self, X, eps=1e-10, iter_limit=1000):
        """
        Fit the model with X.

        Parameters
        ----------
        X : arrays (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        M = self.covariance(X)

        for n in range(self.n_components_):
            x = np.random.randn(M.shape[0])

            for _ in range(iter_limit):
                x_new = M.dot(x) / np.linalg.norm(M.dot(x))

                if np.linalg.norm(x_new - x) < eps:
                    break

                x = x_new

            eigval = x_new.dot(M).dot(x_new)
            eigvec = np.array([x_new])
            M = M - (eigvec.T.dot(eigvec)) * eigval

            self.explained_variance_[n] = eigval
            self.components_.append(x_new)

        self.components_ = np.array(self.components_)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        return self


class OrtoPCA(PCA):
    def fit(self, X, eps=1e-10, iter_limit=1000):
        """
        Fit the model with X.

        Parameters
        ----------
        X : arrays (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        M = self.covariance(X)
        b = np.random.randn(X.shape[1], self.n_components_)
        b = b / np.linalg.norm(b, axis=0)

        for i in range(iter_limit):
            b_new = self.gram_schmidt_orthogonalize(M.dot(b))

            if np.linalg.norm(b - b_new) < eps:
                self.components_ = b_new.T
                self.explained_variance_ = np.array([(M.dot(eigvec) / eigvec)[0] for eigvec in self.components_])
                self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
                return self.components_

            b = b_new

        return self

    def gram_schmidt_orthogonalize(self, vecs):
        """
        Gram-Schmidt orthogonalization of column vectors.

        Parameters
        ----------
        vecs (np.adarray): Array of shape [n_features, k] with column
            vectors to orthogonalize.

        Returns
        ----------
        Orthonormalized vectors of the same shape as on input.
        """
        U = np.array(vecs, copy=True)
        k = vecs.shape[1]

        for i in range(1, k):
            U[:, i] -= sum([self.project(U[:, j], vecs[:, i]) for j in range(i)])

        return U / np.linalg.norm(U, axis=0)

    def project(self, u, v):
        """
        Projection operator.

        Parameters
        ----------
        u : Vector being projected on.
        v : Projecting vector.

        Returns
        ----------
        v_new : projection of v on u.
        """
        return v.dot(u) / u.dot(u) * u


if __name__ == '__main__':
    # Read data
    file = "data/train.csv"
    X = list()
    indexes = list()
    with open(file, "rt") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        for line in reader:
            X.append(list(map(int, line[1:])))
            indexes.append(int(line[0]))
    X = np.array(X)
    indexes = np.array(indexes)
    print("read")

    # fit a model an transform
    P = PowerPCA(2)
    P.fit(X)
    trans = P.transform(X)
    print("fitted and transformed")

    # visualize
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # dots
    colors = cm.rainbow(np.linspace(0, 1, len(set(indexes))))
    for line, index in zip(trans, indexes):
        ax.scatter(line[0], line[1], color=colors[index], alpha=0.7, s=0.5)

    # numbers
    for i in set(indexes):
        x = np.array(trans[np.where(indexes == i)])
        x = x.mean(axis=0)
        ax.text(x[0], x[1], str(i), color=colors[i], fontsize=25)

    plt.savefig('pca.pdf')
    plt.show()

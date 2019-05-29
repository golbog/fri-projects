from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, triu
from scipy.io import mmread
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import NMF
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD


def mtx_to_npz(mtx_path, npz_path):
    """
    Save .mtx file in .npz format ( faster reading).
    """
    data_mat = mmread(mtx_path)
    sparse_data = sparse.csr_matrix(data_mat)
    sparse.save_npz(npz_path, sparse_data)


class ConsensusClustering:
    """
    Consensus clustering algorithm.

    Parameters
    ----------
    min_k : minimum number of clusters
    max_k : maximum number of clusters
    resample_rate : how much data we take in resampling phase
    H : number of resamplings for each cluster number
    """

    def __init__(self, min_k=30, max_k=51, resample_rate=0.5, H=5):
        self.min_k = min_k
        self.max_k = max_k
        self.resample_rate = resample_rate
        self.H = H

    def resample(self, data):
        """
        Resample data.
        """
        rows = random.sample(list(range(data.shape[0])), int(data.shape[0] * self.resample_rate))
        return sorted(rows), data[rows]  # mogoce [rows,:]

    def calculate_consensus_matrix(self, M, rows, sz):
        """
        Calculate consensus matrix.
        """
        consM = lil_matrix((sz, sz), dtype=np.float32)
        for h, Mh in enumerate(M):
            print(h)
            for i in range(len(Mh)):
                for j in range(i + 1, len(Mh)):
                    x = rows[h][i]
                    y = rows[h][j]
                    if Mh[i] == Mh[j]:
                        consM[x, y] += 1
                    consM[y, x] += 1

        for i in range(sz):
            for j in range(i + 1, sz):
                if consM[i, j] > 0:
                    consM[i, j] /= consM[j, i]

        return csr_matrix(triu(consM, k=1))

    def fit_transform(self, data, remaining_data=None):
        """
        Run consensus clustering algorithem for data and apply it to both data and remaining_data.
        """
        Mc = list()
        for k in range(self.min_k, self.max_k + 1):
            print(k)
            cluster = MiniBatchKMeans(n_clusters=k, random_state=0)
            M = list()
            R = list()

            for h in range(self.H):
                [rows, D] = self.resample(data)
                Mh = cluster.fit_predict(D)
                M.append(Mh)
                R.append(rows)
            # compute consensus matrix
            Mc.append(self.calculate_consensus_matrix(M, R, data.shape[0]))

        # Select best index of k
        k_index = self.find_best_k(Mc)
        best_n = k_index + self.min_k
        best_Mc = Mc[k_index]

        # partition D into k clusters based on Mc(k)
        distance_matrix = 1 - best_Mc.toarray()
        y = AgglomerativeClustering(n_clusters=best_n, affinity='precomputed', linkage='average').fit_predict(
            distance_matrix)

        # cluster remaining data
        if remaining_data is not None:
            clf = RandomForestClassifier()
            clf.fit(data, y)
            predicted = clf.predict(remaining_data)
            return y, predicted

        return y

    def find_best_k(self, Mc):
        """
        Select best number of clusters according to consensus matrices.
        """
        utri_indeces = np.triu_indices(Mc[0].shape[0], 1)
        As = list()
        for M in Mc:
            M_utri = np.sort(M[utri_indeces]).tolist()[0]
            quantiles, cumprob, counts = self.ecdf(M_utri)
            probs = np.concatenate(
                np.array([list(cumprob[i] for _ in range(count)) for i, count in enumerate(counts)])).ravel()

            zipped = list(zip(M_utri, probs))
            A = sum([(M_utri[i] - M_utri[i - 1]) * zipped[i][1] for i in range(1, len(M_utri))])

            As.append(A)

        delta_kKs = [((As[i + 1] - As[i]) / As[i]) for i in range(len(As) - 1)]
        print(delta_kKs)
        print(max(enumerate(delta_kKs), key=(lambda x: x[1])))
        return max(enumerate(delta_kKs), key=(lambda x: x[1]))[0]

    def ecdf(self, sample):
        """
        Calculate empirical CDF.
        """
        sample = np.array(sample)
        quantiles, counts = np.unique(sample, return_counts=True)
        cumprob = np.cumsum(counts).astype(np.double) / sample.size
        return quantiles, cumprob, counts


if __name__ == '__main__':
    # example usage
    COLS = 2000
    ROWS = 6100

    data = sparse.load_npz('data/train.npz')
    data = data > 0

    rows = np.array(random.sample(list(range(data.shape[0])), ROWS))
    sum_cols = np.sum(data, axis=0).tolist()[0]
    columns = np.argsort(sum_cols)[-COLS:]
    best_data = data[:, columns]
    learn_data = best_data[rows, :]
    remaining_data = np.delete(best_data.toarray(), rows, axis=0)

    consensus = True
    if consensus:
        cs = ConsensusClustering()
        y1, y2 = cs.fit_transform(learn_data, remaining_data)

        res = np.zeros(len(y1) + len(y2), dtype='int8')
        res[rows] = y1
        rows_remaining = [i for i in range(len(sum_cols)) if i not in rows]
        res[rows_remaining] = y2
        np.savetxt('res-cons(45-51).txt', res, delimiter='\n', fmt='%d')

    else:
        nmf = NMF(n_components=50, init='random', random_state=0)
        kmeans = MiniBatchKMeans(n_clusters=50, random_state=0)

        W = nmf.fit_transform(data[:, columns])
        res = kmeans.fit_predict(W)
        np.savetxt('res-cons-most.txt', res, delimiter='\n', fmt='%d')

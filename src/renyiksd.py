import numpy as np

from goftest import NystroemKSD
from kgof.goftest import bootstrapper_rademacher


def select_renyi_landmarks(X, k, m):
    """
    Greedy Rényi-inspired landmark selection that minimizes the quadratic
    energy 1^T K 1 over the chosen subset. Returns an array of indices of
    length m (m is clipped to n).
    """
    n = len(X)
    m = min(m, n)
    K = k.eval(X, X)
    diag = np.diag(K)

    # start with the point of lowest total kernel mass
    row_sums = K.sum(axis=1)
    first = int(np.argmin(row_sums))
    selected = [first]

    # cross_sums[i] = sum_{j in selected} K_ij
    cross_sums = K[:, first].copy()

    while len(selected) < m:
        # incremental energy if we add candidate i: 2 * sum_{j in S} K_ij + K_ii
        scores = 2 * cross_sums + diag
        scores[selected] = np.inf  # avoid re-picking
        nxt = int(np.argmin(scores))
        selected.append(nxt)
        cross_sums += K[:, nxt]

    return np.array(selected, dtype=int)


class RenyiNystroemKSD(NystroemKSD):
    """
    Nyström KSD variant that uses a Rényi-inspired landmark selector to choose
    informative Nyström points instead of uniform sampling. Interface matches
    NystroemKSD so it can replace it in downstream training (e.g., GAN loss).
    """

    def __init__(self, p, k, m=lambda n: int(4 * np.sqrt(n)), bootstrapper=bootstrapper_rademacher,
                 alpha=0.01, n_simulate=500, seed=11,
                 landmark_selector=select_renyi_landmarks):
        super().__init__(p, k, m, bootstrapper, alpha, n_simulate, seed)
        self.landmark_selector = landmark_selector

    def compute_stat(self, dat, return_ustat_gram=False):
        """
        Select landmarks with the Rényi criterion, then compute the Nyström KSD
        statistic. If `return_ustat_gram` is True, returns (stat, (H_mn, H_mm_inv)).
        """
        X = dat.data()
        n = len(X)
        m = min(self.m(n), n)

        idx = self.landmark_selector(X, self.k, m)

        H_mn = self.h_p(X[idx], X)
        H_mm_inv = np.linalg.pinv(H_mn[:, idx], hermitian=True)
        beta = H_mn @ np.ones((n, 1)) / n

        stat = (beta.T @ H_mm_inv @ beta)[0][0]
        if return_ustat_gram:
            return stat, (H_mn, H_mm_inv)
        return stat


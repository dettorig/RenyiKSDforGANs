import numpy as np

import kgof.data as data
import kgof.kernel as kernel
import kgof.density as density
from kgof.goftest import KernelSteinTest

from renyiksd import RenyiNystroemKSD


def test_renyi_h_p_matches_reference_gauss():
    n, m, d = 100, 10, 5
    rng = np.random.default_rng(1234)
    X = rng.standard_normal(size=(n, d))
    k = kernel.KGauss(sigma2=0.5)
    p = density.IsotropicNormal(0, 1)
    _, ref_gram = KernelSteinTest(p=p, k=k).compute_stat(
        data.Data(X), return_ustat_gram=True
    )
    actual = RenyiNystroemKSD(p=p, k=k).h_p(X=X, Y=X)
    np.testing.assert_allclose(ref_gram, actual)


def test_renyi_stat_close_to_quadratic():
    n, d = 300, 2
    rng = np.random.default_rng(1234)
    X = data.Data(rng.standard_normal(size=(n, d)))
    k = kernel.KGauss(sigma2=0.5)
    p = density.IsotropicNormal(0, 1)
    ref_stat = KernelSteinTest(p=p, k=k).compute_stat(X, return_ustat_gram=False)
    renyi_stat = RenyiNystroemKSD(p=p, k=k).compute_stat(X)
    np.testing.assert_allclose(ref_stat / n, renyi_stat, rtol=5e-2, atol=5e-3)


def test_renyi_h0_and_h1():
    n, d = 600, 2
    rng = np.random.default_rng(1234)
    k = kernel.KGauss(sigma2=0.5)
    p = density.IsotropicNormal(0, 1)

    X_h0 = data.Data(rng.standard_normal(size=(n, d)))
    res_h0 = RenyiNystroemKSD(p=p, k=k, alpha=0.01, n_simulate=800, seed=123).perform_test(X_h0)
    assert res_h0["pvalue"] > 0.2  # should not reject under H0

    X_h1 = data.Data(rng.normal(loc=3.0, size=(n, d)))  # bigger shift for power
    res_h1 = RenyiNystroemKSD(p=p, k=k, alpha=0.01, n_simulate=800, seed=123).perform_test(X_h1)
    assert res_h1["pvalue"] < 0.05  # should reject under H1

if __name__ == "__main__":
    test_renyi_h_p_matches_reference_gauss()
    test_renyi_stat_close_to_quadratic()
    test_renyi_h0_and_h1()
    print("done")

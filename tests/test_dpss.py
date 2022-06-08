from croissant import dpss
import numpy as np
import pytest


def test_dpss_op():

    x = np.linspace(1, 50, 50)  # target frequencies
    with pytest.raises(ValueError):  # didn't specify any kwargs
        _ = dpss.dpss_op(x)

    with pytest.raises(ValueError):
        _ = dpss.dpss_op(x, eignval_cutoff=1e-12, nterms=10)  # too many kwargs

    nterms = 10
    design_matrix = dpss_op(x, nterms=nterms)
    assert design_matrix.shape == (x.size, nterms)
    assert design_matrix == design_matrix.real


def test_freq2dpss2freq():
    # generate 3 sets of frequencies that should share dpss modes
    # croissant does this with beam, sky, and simulation frequencies
    f1 = np.linspace(10, 60, 50)
    f2 = np.linspace(3, 70, 100)
    f3 = np.linspace(1, 50, 50)

    x = np.unique(np.concatenate((f1, f2, f3)))
    nterms = 10
    A = dpss.dpss_op(x, nterms=nterms)  # design matrix

    data = [[f**2, np.sin(f)] for f in [f1, f2, f3]]  # mock data

    # compute dpss coeffs
    d1 = freq2dpss(data[0].T, x, f1, A)
    d2 = freq2dpss(data[1].T, x, f2, A)
    d3 = freq2dpss(data[2].T, x, f3, A)

    assert d1.shape == d2.shape == d3.shape == (nterms, 2)

    f1_ = dpss2freq(d1)
    assert f1.shape == f1_
    f2_ = dpss2freq(d2)
    assert f2.shape == f2_
    f3_ = dpss2freq(d3)
    assert f3.shape == f3_

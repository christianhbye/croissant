from croissant import dpss
import numpy as np
import pytest


def test_dpss_op():

    x = np.linspace(1, 50, 50)  # target frequencies
    with pytest.raises(ValueError):  # didn't specify any kwargs
        _ = dpss.dpss_op(x)

    with pytest.raises(ValueError):
        _ = dpss.dpss_op(
            x, eigenval_cutoff=1e-12, nterms=10
        )  # too many kwargs

    nterms = 10
    design_matrix = dpss.dpss_op(x, nterms=nterms)
    assert design_matrix.shape == (x.size, nterms)
    assert np.allclose(design_matrix, design_matrix.real)


def test_freq2dpss2freq():
    # generate 2 sets of frequencies that should share dpss modes
    # croissant does this with beam and simulation frequencies
    f1 = np.linspace(10, 60, 50)
    f2 = np.linspace(1, 50, 50)

    x = np.unique(np.concatenate((f1, f2)))
    nterms = 10
    A = dpss.dpss_op(x, nterms=nterms)  # design matrix

    # mock data
    data = [np.array([f**2, np.sin(f)]) for f in [f1, f2]]

    # compute dpss coeffs
    d1 = dpss.freq2dpss(data[0].T, f1, x, A)
    d2 = dpss.freq2dpss(data[1].T, f2, x, A)

    assert d1.shape == d2.shape == (nterms, 2)

    d1_ = dpss.dpss2freq(d1, A)
    assert data[0].T.shape == d1_[np.isin(x, f1)].shape
    d2_ = dpss.dpss2freq(d2, A)
    assert data[1].T.shape == d2_[np.isin(x, f2)].shape

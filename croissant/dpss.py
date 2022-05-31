from hera_filters.dspec import dpss_operator, fit_solution_matrix
import numpy as np


def dpss_coeffs(
    data,
    freq_in,
    freq_out=None,
    filter_center=0,
    filter_half_width=20e-9,
    eigenval_cutoff=None,
    edge_suppression=None,
    nterms=None,
    avg_suppression=None,
):
    """
    Transform data from frequency space to a basis of Discrete Prolate
    Spheroidal Sequences (DPSS). Thin wrapper around
    hera_filters.dspec.dpss_operator suitable for a single antenna
    autocorrelation experiment.

    Specify one and only one of eigenval_cutoff (min eigenvalue of sinc matrix),
    edge_suppression (degree of tone suppression at edge of filter window),
    nterms (number of DPSS terms to include), and
    avg_suppression (the avg suppression of tones within filter edges) to
    determine the cutoff for number of DPSS modes to use.

    See Ewall-Wice et al (2021) and https://github.com/HERA-Team/hera_filters
    for discussion of DPSS.

    Parameters
    ----------
    data : array-like
        The data to transform. The frequency axis must be the first axis.
    freq_in : array-like
        The frequencies in MHz that the data is sampled at.
    freq_out : array-like, optional
        The frequencies in MHz to map the inverse transform to. The DPSS modes
        will be used to interpolate to freq_out if necessary. Default: freq_in.
    filter_center : float (optional)
        The center of the delay filter window in seconds.
    filter_half_width : float, optional
        The half-width of the the delay filter window in seconds.
    eigenval_cutoff : float (optional)
        The threshold for eigenvalues of sinc matrix to include DPSS modes for.
    edge_suppression : float (optional)
        The minimum suppression of a tone at the edge of the filter window
        needed to include a DPSS mode.
    nterms : int (optional)
        The number of DPSS modes to include.
    avg_suppression : float (optional)
        The average suppression within a filter window needed to include a
        DPSS mode.

    Returns
    -------
    dpss_coeffs : np.ndarray
        The data transformed to the DPSS basis.
    B : np.ndarray
        The design matrix specifying the inverse transform. That is,
        B @ dpss_coeffs will return the data in frequency domain (interpolated
        to freq_out).

    Raises
    ------
    ValueError :
        If not one and only one of eigenval_cutoff, edge_suppression, nterms,
        and avg_suppression is specified and the rest are None.

    """
    freq_in = np.array(freq_in) * 1e6  # Hz

    if freq_out is None:
        freq_out = freq_in.copy()
        x = freq_in.copy()
    else:
        freq_out = np.array(freq_out) * 1e6
        if freq_out.max() > freq_in.max() or freq_out.min() < freq_in.min():
            raise ValueError(
                "Some of the values of freq_out are outside the range of "
                "freq_in."
            )

        x = np.unique(np.sort(np.append(freq_in, freq_out)))

    kwargs = {
        "eigenval_cutoff": eigenval_cutoff,
        "edge_suppression": edge_suppression,
        "nterms": nterms,
        "avg_suppression": avg_suppression,
    }

    # remove Nones and put the specified value in a list
    kwarg = {key: [val] for key, val in kwargs.items() if val is not None}
    if not len(kwarg) == 1:
        raise ValueError(f"Specify one and only of {list[kwargs.keys()]}.")

    W = np.eye(freq_in.size)
    B = dpss_operator(
        x,
        filter_centers=[filter_center],
        filter_half_widths=[filter_half_width],
        cache=None,
        **kwarg,
    )[0]
    A = B[np.isin(x, freq_in)]
    coeffs = fit_solution_matrix(W, A) @ np.array(data)
    return coeffs, B

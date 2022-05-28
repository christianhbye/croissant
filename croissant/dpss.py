from hera_filters.dpsec import dpss_operator

def dpss_interpolator(target_frequencies, input_freqs, **kwargs):
    """
    Compute linear interpolator in frequency space using the Discrete Prolate
    Spheroidal Sequences (DPSS) basis.
    """
    if input_freqs is None:
        raise ValueError("No input frequencies are provided.")
    input_freqs = np.copy(input_freqs) * 1e6  # convert to Hz
    target_frequencies = np.array(target_frequencies) * 1e6  # Hz
    if np.max(target_frequencies) > np.max(input_freqs) or np.min(
        target_frequencies
    ) < np.min(input_freqs):
        raise ValueError(
            "Some of the target frequencies are outside the range of the "
            "input frequencies."
        )
    target_frequencies = np.unique(np.append(target_frequencies, input_freqs))
    target_frequencies.sort()

    fc = kwargs.pop("filter_centers", [0])
    fhw = kwargs.pop("filter_half_widths", [20e-9])
    ev_cut = kwargs.pop("eigenval_cutoff", [1e-12])
    B = dpss_operator(
        target_frequencies,
        filter_centers=fc,
        filter_half_widths=fhw,
        eigenval_cutoff=ev_cut,
        **kwargs,
    )
    A = B[np.isin(target_frequencies, input_freqs)]
    interp = B @ np.linalg.inv(A.T @ A) @ A.T
    return interp

def coord_rep(coord):
    """
    Shorthand notation for coordinate systems.

    Parameters
    ----------
    coord : str
        The name of the coordinate system.

    Returns
    -------
    rep : str
        The one-letter shorthand notation for the coordinate system.

    """
    coord = coord.upper()
    if coord[0] == "E" and coord[1] == "Q":
        rep = "C"
    else:
        rep = coord[0]
    return rep

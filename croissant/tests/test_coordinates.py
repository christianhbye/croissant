from croissant.utils import coord_rep


def test_coord_rep():
    coords = ["galactic", "equatorial", "ecliptic", "mcmf", "topocentric"]
    short = ["G", "C", "E", "M", "T"]
    for i in range(len(coords)):
        assert coord_rep(coords[i]) == short[i]

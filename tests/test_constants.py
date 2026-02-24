import pytest

from croissant import constants


def test_future_warning():
    with pytest.warns(FutureWarning):
        wgts = constants.PIX_WEIGHTS_NSIDE
    assert wgts == constants._PIX_WEIGHTS_NSIDE


def test_missing_constant():
    with pytest.raises(AttributeError):
        _ = constants.NON_EXISTENT_CONSTANT

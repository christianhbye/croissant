import jax.numpy as jnp
import pytest
from croissant.jax import rotations

pytestmark = pytest.mark.parametrize("lmax", [8, 16, 64, 128])


def test_generate_euler_dl(lmax):
    pass

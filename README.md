# CROISSANT: spheriCal haRmOnics vISibility SimulAtor iN pyThon

[![codecov](https://codecov.io/gh/christianhbye/croissant/branch/main/graph/badge.svg?token=pj1hkgcazd)](https://codecov.io/gh/christianhbye/croissant)

CROISSANT is a rapid visiblity simulator in python based on spherical harmonics. Given an antenna design and a sky model, CROISSANT simulates the visbilities - that is, the perceived sky temperature.

CROISSANT uses spherical harmonics to decompose the sky and antenna beam to a set of coefficients. Since the spherical harmonics represents a complete, orthormal basis on the sphere, the visibility computation reduces nicely from a convolution to a dot product.

In frequency domain, CROISSANT uses Discrete Prolate Spheroidal Sequences as a rapid linear interpolation scheme. Being linear, this interpolation can be done directly on the spherical harmonics coefficients, avoiding redoing the most expensive part of the computation.

Moreover, the time evolution of the simulation is very natural in this representation. In the antenna reference frame, the sky rotates overhead with time. To account for this rotation, it is enough to rotate the spherical harmonics coefficients. In the right choice of coordinates (that is, one where the z-axis is aligned with the rotation axis of the earth or the moon), this rotation is simply achieved by multiplying the spherical coefficient by a phase.


> **New in version 4.0.0:** CROISSANT is now fully compatible with JAX, provided in the interfeace croissant.jax. Spherical harmonics transforms (built on [s2ftt](https://github.com/astro-informatics/s2fft/)), coordinate system transforms, rotations, and the simulator itself can now all be differentiated using JAX autograd.

Overall, this makes CROISSANT a very fast visibility simulator. CROISSANT can therefore be used to simulate a large combination of antenna models and sky models - allowing for the exploration of a range of propsed designs before choosing an antenna for an experiment.

## Installation
For the latest release, do `pip install croissant-sim` (see https://pypi.org/project/croissant-sim). Git clone this repository for the newest changes (this is under activate development, do so at your own risk!).

To access the JAX features, JAX must also be installed. See the [installation guide](https://github.com/google/jax#installation).

## Demo
Jupyter Notebook: https://nbviewer.org/github/christianhbye/croissant/blob/main/notebooks/example_sim.ipynb

## Contributing
Contributions are welcome - please see the [contribution guidelines](https://github.com/christianhbye/croissant/blob/add_contributing/CONTRIBUTING.md).

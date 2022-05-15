# CROISSANT: spheriCal haRmOnics vISibility SimulAtor iN pyThon

CROISSANT is a rapid visiblity simulator in python based on spherical harmonics. Given an antenna design and a sky model, CROISSANT simulates the visbilities - that is, the perceived sky temperature.

CROISSANT uses spherical harmonics to decompose the sky and antenna beam to a set of coefficients. Since the spherical harmonics represents a complete, orthormal basis on the sphere, the visibility computation reduces nicely from a convolution to a dot product.

In frequency domain, CROISSANT uses Discrete Prolate Spheroidal Sequences as a rapid linear interpolation scheme. Being linear, this interpolation can be done directly on the spherical harmonics coefficients, avoiding redoing the most expensive part of the computation.

Moreover, the time evolution of the simulation is very natural in this representation. In the antenna reference frame, the sky rotates overhead with time. To account for this rotation, it is enough to rotate the spherical harmonics coefficients. In the right choice of coordinates (that is, one where the z-axis is aligned with the rotation axis of the earth), this rotation is simply achieved by multiplying the spherical coefficient by a phase.

Overall, this makes CROISSANT a very fast visibility simulator. CROISSANT can therefore be used to simulate a large combination of antenna models and sky models - allowing for the exploration of a range of propsed designs before choosing an antenna for an experiment.

Finally, CROISSANT is parallelizable as the time domain of the simulation easily can be run in parallell, 


## Installation
`pip install croissant-sim` (see https://pypi.org/project/croissant-sim/0.1.0/)

## Demo
YouTube: https://youtu.be/P1wzTp5QlY0 \
Jupyter Notebook: https://nbviewer.org/github/christianhbye/croissant/blob/main/notebooks/example_sim.ipynb

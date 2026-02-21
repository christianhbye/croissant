# CROISSANT: spheriCal haRmOnics vISibility SimulAtor iN pyThon

[![codecov](https://codecov.io/gh/christianhbye/croissant/branch/main/graph/badge.svg?token=pj1hkgcazd)](https://codecov.io/gh/christianhbye/croissant)

CROISSANT is a rapid visiblity simulator in python based on spherical harmonics. Given an antenna design and a sky model, CROISSANT simulates the visbilities - that is, the perceived sky temperature.

CROISSANT uses spherical harmonics to decompose the sky and antenna beam to a set of coefficients. Since the spherical harmonics represents a complete, orthormal basis on the sphere, the visibility computation reduces nicely from a convolution to a dot product.

Moreover, the time evolution of the simulation is very natural in this representation. In the antenna reference frame, the sky rotates overhead with time. To account for this rotation, it is enough to rotate the spherical harmonics coefficients. In the right choice of coordinates (that is, one where the z-axis is aligned with the rotation axis of the earth or the moon), this rotation is simply achieved by multiplying the spherical coefficient by a phase.


> **New in version 5.0.0:** CROISSANT is now fully based on JAX and legacy support for numpy/healpy code is dropped. Spherical harmonics transforms (built on [s2ftt](https://github.com/astro-informatics/s2fft/)), coordinate system transforms, rotations, and the simulator itself can now all be differentiated using JAX autograd.

Overall, this makes CROISSANT a very fast visibility simulator. CROISSANT can therefore be used to simulate a large combination of antenna models and sky models - allowing for the exploration of a range of propsed designs before choosing an antenna for an experiment.

## Installation
To install the package for standard use, you can use your preferred Python package manager:

**Using `uv` (Recommended)**
```bash
uv pip install croissant-sim

```

**Using `pip`**

```bash
pip install croissant-sim

```

Note that croissant is only tested up to Python 3.12. Python 3.13 and newer versions are experimental.

## Development

We recommend using [`uv`](https://github.com/astral-sh/uv) to manage the development environment. It is exceptionally fast and handles virtual environments, dependencies, and lockfiles automatically. However, standard `pip` workflows are also fully supported.

### 1. Set Up the Environment

**Option A: Using `uv` (Recommended)**

`uv` will automatically read the `pyproject.toml`, create a virtual environment (`.venv`), and install all core and development dependencies.

```bash
# Clone the repository
git clone git@github.com:christianhbye/croissant.git
cd croissant

# Sync the project and install all dependencies
uv sync

```

**Option B: Using `pip`**

If you prefer standard Python tools, you will need to manually create the environment and install the package in editable mode.

```bash
# Clone the repository
git clone git@github.com:christianhbye/croissant.git
cd croissant

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with development dependencies
pip install -e .
pip install -r requirements-dev.txt

```

### 2. Install Pre-commit Hooks

We use `pre-commit` to automatically format and lint code before every commit. This ensures all code follows our style guidelines (enforced by `ruff`).

**With `uv`:**

```bash
uv run pre-commit install

```

**With `pip`:**

```bash
pre-commit install

```

### 3. Running Tests and Linters

We use `pytest` for testing and `ruff` for linting and formatting. Prepend these commands with `uv run` if using `uv`.

```bash
pytest                         # Run the test suite
ruff format                    # Auto-format code
ruff check --fix               # Run linter and fix auto-fixable errors

```

## Demo
Jupyter Notebook: https://nbviewer.org/github/christianhbye/croissant/blob/main/notebooks/example_sim.ipynb

## Contributing
Contributions are welcome - please see the [contribution guidelines](https://github.com/christianhbye/croissant/blob/add_contributing/CONTRIBUTING.md).

# Changelog

## [5.1.0](https://github.com/christianhbye/croissant/compare/croissant-sim-v5.0.0...croissant-sim-v5.1.0) (2026-03-11)


### ⚠ BREAKING CHANGES

* make s2fft euler angle convetions default
* remove numpy/healpy beam module, create JAX beam
* delete numpy/healpy backend + use uv

### Features

* add moon/earth options, make sky an object ([a57cf46](https://github.com/christianhbye/croissant/commit/a57cf465129cf301e767b81d3e1fb835d17860fd))
* add n_iter keyword and set defaults ([a8a012e](https://github.com/christianhbye/croissant/commit/a8a012e8e82f2bc44c8b56ace87c0dd8313047ad))
* create SphBase and Sky classes ([7e43cec](https://github.com/christianhbye/croissant/commit/7e43cecacd86578b690e9ed37d7f68c5f605636a))
* delete numpy/healpy backend + use uv ([482387d](https://github.com/christianhbye/croissant/commit/482387d7a4ce229b854181047fdbd51dba32e8d2))
* make s2fft euler angle convetions default ([d44afd3](https://github.com/christianhbye/croissant/commit/d44afd35c825137a4e43e8768bc5e5e989c8e43c))
* remove numpy/healpy beam module, create JAX beam ([efd1d55](https://github.com/christianhbye/croissant/commit/efd1d5561952aa1efd2503f5f63b96f3b0436bf2))


### Bug Fixes

* change import from alm to utils ([e367097](https://github.com/christianhbye/croissant/commit/e367097ac0c2cbb674b6ba5b3cdebe786438bb02))
* docs and small fixes to address comments in IMPROVEMENTS.md ([#89](https://github.com/christianhbye/croissant/issues/89)) ([463027b](https://github.com/christianhbye/croissant/commit/463027bd2f248faed87bed5bb308aaa428d4bc80))
* ensure constant.PIX_WEIGHTS_NSIDE raises the correct FutureWarning ([cf046c9](https://github.com/christianhbye/croissant/commit/cf046c9b73610bd8c71e74d38294bf0aba1b94fd))
* fix circular dependency rotations/utils for moved functions ([3aeb6bf](https://github.com/christianhbye/croissant/commit/3aeb6bf3132c0d33da5644302de0e7aaae2cfd31))
* fix imports ([0617d25](https://github.com/christianhbye/croissant/commit/0617d2575b36cae46733269b54effee28ff1e2e8))
* fix imports and ruff rules ([71fb368](https://github.com/christianhbye/croissant/commit/71fb36827cfe953383a3d84d1dd91870945bff79))
* maintain consistent function signatures in croissant top level for backwards compat ([53541a9](https://github.com/christianhbye/croissant/commit/53541a94ecde5c740d8d0e48bf2e57f7df1c3bbe))
* make sure futurewarning is called when util function is called, not imported ([42ad3f2](https://github.com/christianhbye/croissant/commit/42ad3f27f50cab99a08b6b6037eee48413ef5ede))
* remove old project files ([48799d9](https://github.com/christianhbye/croissant/commit/48799d9c52d222df96ea46b4e22e02af5e6bfb3d))
* update multipair example notebook to croissant v5 api ([9fb5567](https://github.com/christianhbye/croissant/commit/9fb5567e9581b8b559c508da0d3488409da1d9cc))


### Documentation

* clarify docstrings and warnings ([0b22b96](https://github.com/christianhbye/croissant/commit/0b22b96004fe53bca5070e23915801d05efd277d))
* fix bug in demo notebook ([2fe4c40](https://github.com/christianhbye/croissant/commit/2fe4c407621ea258c9e44507f3ca3d8e1246e893))
* fix docstring grammar ([f2a6d99](https://github.com/christianhbye/croissant/commit/f2a6d995ada7eb4ddc13114ae2c39e8cdf2b36a7))


### Miscellaneous Chores

* reset release baseline ([98625f3](https://github.com/christianhbye/croissant/commit/98625f3ef0121d02a338b7db36dd05972f00cef7))

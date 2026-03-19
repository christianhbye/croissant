# Changelog

## [5.1.1](https://github.com/christianhbye/croissant/compare/croissant-sim-v5.1.0...croissant-sim-v5.1.1) (2026-03-19)


### Bug Fixes

* add mepa et parameter ([7bb4651](https://github.com/christianhbye/croissant/commit/7bb46518ab85462448199971865cc27c5d5c70c5))
* cap cache maxsize ([0661eb2](https://github.com/christianhbye/croissant/commit/0661eb2724abe2499269c1a10f9db3873cb17fdb))


### Documentation

* fix docstring quote ([311c57b](https://github.com/christianhbye/croissant/commit/311c57bf9ac700c2185ea968aa8e73318192330b))
* fix notebooks with new code ([58611bf](https://github.com/christianhbye/croissant/commit/58611bf839f0a1c2f932eb3d0aa0c8b8d6450020))
* replace mcmf with mepa in docstring ([277208e](https://github.com/christianhbye/croissant/commit/277208e336d8ded6ae49217ebcbfaed99f1ca212))
* update claude.md ([e653aaa](https://github.com/christianhbye/croissant/commit/e653aaad1bc5628647500cd3075ca0ee39983969))
* update example notebooks with mepa frame code ([3d9e0e8](https://github.com/christianhbye/croissant/commit/3d9e0e8b6b5406e9426e34c980f3d8f6d615b6d0))
* update notebooks ([ae674d6](https://github.com/christianhbye/croissant/commit/ae674d6ec4460a0aa7e018370ba0aa3690f5eb36))

## [5.1.0](https://github.com/christianhbye/croissant/compare/croissant-sim-v5.0.0...croissant-sim-v5.1.0) (2026-03-16)


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

* cache rotation matrix calculation ([43a8dda](https://github.com/christianhbye/croissant/commit/43a8ddaf9bcb7a5da226c471d8d5283c3c70d800))
* change import from alm to utils ([e367097](https://github.com/christianhbye/croissant/commit/e367097ac0c2cbb674b6ba5b3cdebe786438bb02))
* derive obstime from topo_frame in topo_to_mepa_euler_dl ([d4c4f31](https://github.com/christianhbye/croissant/commit/d4c4f31aa24af2eb49081c0fe1d8209b68bc5a88))
* docs and small fixes to address comments in IMPROVEMENTS.md ([#89](https://github.com/christianhbye/croissant/issues/89)) ([463027b](https://github.com/christianhbye/croissant/commit/463027bd2f248faed87bed5bb308aaa428d4bc80))
* ensure constant.PIX_WEIGHTS_NSIDE raises the correct FutureWarning ([cf046c9](https://github.com/christianhbye/croissant/commit/cf046c9b73610bd8c71e74d38294bf0aba1b94fd))
* fix circular dependency rotations/utils for moved functions ([3aeb6bf](https://github.com/christianhbye/croissant/commit/3aeb6bf3132c0d33da5644302de0e7aaae2cfd31))
* fix imports ([0617d25](https://github.com/christianhbye/croissant/commit/0617d2575b36cae46733269b54effee28ff1e2e8))
* fix imports and ruff rules ([71fb368](https://github.com/christianhbye/croissant/commit/71fb36827cfe953383a3d84d1dd91870945bff79))
* maintain consistent function signatures in croissant top level for backwards compat ([53541a9](https://github.com/christianhbye/croissant/commit/53541a94ecde5c740d8d0e48bf2e57f7df1c3bbe))
* make sure futurewarning is called when util function is called, not imported ([42ad3f2](https://github.com/christianhbye/croissant/commit/42ad3f27f50cab99a08b6b6037eee48413ef5ede))
* place imports on top of module ([64b4994](https://github.com/christianhbye/croissant/commit/64b4994f8a2d8059c7b8c55a06562ca620ad4b31))
* remove old project files ([48799d9](https://github.com/christianhbye/croissant/commit/48799d9c52d222df96ea46b4e22e02af5e6bfb3d))
* replace MCMF with MEPA frame for Moon simulations ([9488cbe](https://github.com/christianhbye/croissant/commit/9488cbee3179dc9ad207a804fadec9402082a2b9))
* update multipair example notebook to croissant v5 api ([9fb5567](https://github.com/christianhbye/croissant/commit/9fb5567e9581b8b559c508da0d3488409da1d9cc))
* use standalone functions for MEPA test in notebook ([45822fe](https://github.com/christianhbye/croissant/commit/45822fe3055df8b4e3d9ff017eead4cc6141610e))
* use time.tdb.jd for accurate phase ([33f4d50](https://github.com/christianhbye/croissant/commit/33f4d50fe09afab99d6346e167b3ca00980d4193))


### Documentation

* add claude.md ([dd990cd](https://github.com/christianhbye/croissant/commit/dd990cdea7a155d10e6cd03b77c235fc584c28d4))
* clarify docstrings and warnings ([0b22b96](https://github.com/christianhbye/croissant/commit/0b22b96004fe53bca5070e23915801d05efd277d))
* fix bug in demo notebook ([2fe4c40](https://github.com/christianhbye/croissant/commit/2fe4c407621ea258c9e44507f3ca3d8e1246e893))
* fix docstring grammar ([f2a6d99](https://github.com/christianhbye/croissant/commit/f2a6d995ada7eb4ddc13114ae2c39e8cdf2b36a7))
* fix name change mcmf to mepa ([977b067](https://github.com/christianhbye/croissant/commit/977b067fbecc2b1c8ccea42574611f51844a58dd))
* rename and run notebook ([9e2223d](https://github.com/christianhbye/croissant/commit/9e2223d60662ff5ff5d2893fee6765bc79c74305))


### Miscellaneous Chores

* reset release baseline ([98625f3](https://github.com/christianhbye/croissant/commit/98625f3ef0121d02a338b7db36dd05972f00cef7))

CROISSANT decmposes the spatial dimensions of the sky model and antenna beams to [spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics). CROISSANT follows the [HEALPix]((https://en.wikipedia.org/wiki/Spherical_harmonics)) convention for spherical harmonics. These are orthonormal, that is
$\int \mathrm{d}\Omega Y_\ell^m \left(Y_{\ell'}^{m'}\right)^* = \delta_{\ell\ell'} \delta_{mm'}$. Moreover, spherical harmonics have the property that their complex conjugate is given by $Y_\ell^{-m} = (-1)^m \left(Y_\ell^m\right)^* $. 
It follows that for real-valued functions, the spherical harmomnics coefficients have the property $a_\ell^{-m} = (-1)^m \left(a_\ell^m \right)^* $. Hence, the coefficients with $m<0$ can always be reconstructed from the coefficients with $m>0$.

CROISSANT simulates visibilities V given by $V(\nu, t) = \int \mathrm{d}\Omega A(\theta, \phi, \nu) T_{\rm sky} (\theta, \phi, \nu, t)$. Decomposing the beam and the sky to spherical harmonics and using the above properties yield a simple expression for the visibility:

$$A(\theta, \phi) = \sum_{\ell, m} a_\ell^m Y_{\ell}^m, T_{\rm sky}(\theta, \phi) = \sum_{\ell, m} b_\ell^m Y_{\ell}^m$$ 

$$V = \sum_\ell a_l^0 b_l^0 + \sum_{\ell, m>0} a_\ell^m \left(b_\ell^m\right)^* + \sum_{\ell, m<0} a_\ell^m \left(b_\ell^m\right)^*$$

$$V = \sum_\ell a_l^0 b_l^0 + \sum_{\ell, m>0} \left(a_\ell^m \left(b_\ell^m\right)^* + \left(a_\ell^m\right)^* b_\ell^m\right)$$

$$V = \sum_\ell a_l^0 b_l^0 + 2\sum_{\ell, m>0}\Re \left(a_\ell^m \left(b_\ell^m\right)^*\right)$$


### DPSS
CROISSANT decomposes the spectral axis of the sky model and antenna beams to Discrete Prolate Shperoidal Sequences (see [Slepian 1978](https://ui.adsabs.harvard.edu/abs/1978ATTTJ..57.1371S/abstract) and [Ewall-Wice et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.5195E/abstract)), using the [hera_filters package](https://github.com/HERA-Team/hera_filters). The transformation from DPSS space to frequency space is described a design matrix $\mathbf{B}$ 
(A is sometimes used but we reserve that for the antenna beam). The formula for the convolution in spherical harmonics-DPSS space is derived below (denoting the design matrix by $\mathbf{B}$,
the sky by S, the antenna beam by A, and DPSS modes by $\eta$). Here, S and A are matrices of the spherical harmonics coefficients with different rows corresponding to different frequencies and different columns corresponding to different harmonics.

$$V_i = \sum_j S_{ij} (\nu, \ell, m) A_{ij} (\nu, \ell, m) = (\mathbf{S} (\nu, \ell, m) \mathbf{A}^T (\nu, \ell, m))_{ii}$$

$$V = diag(\mathbf{S} (\nu, \ell, m) \left(\mathbf{B} \mathbf{A} (\eta, \ell, m) \right)^T)$$

$$V = diag(\mathbf{S} (\nu, \ell, m) \mathbf{A} (\eta, \ell, m)^T \mathbf{B}^T)$$

Denote the product $\mathbf{S} (\nu, \ell, m) \mathbf{A} (\eta, \ell, m)^T $
by $R$.

$$V_{i} = (\mathbf{R} \mathbf{B}^T) \_{ii} = \sum_j R_{ij} B_{ij} $$

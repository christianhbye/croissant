CROISSANT decmposes the spatial dimensions of the sky model and antenna beams to [spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics). CROISSANT follows the [HEALPix]((https://en.wikipedia.org/wiki/Spherical_harmonics)) convention for spherical harmonics. These are orthonormal, that is
$\int \mathrm{d}\Omega Y_\ell^m \left(Y_{\ell'}^{m'}\right)^* = \delta_{\ell\ell'} \delta_{mm'}$. Moreover, spherical harmonics have the property that their complex conjugate is given by $\left(Y_\ell^m\right)^* = (-1)^m Y_\ell^m$. It follows that for real-valued functions, the spherical harmomnics coefficients have the property $\left(a_\ell^m\right)^*=(-1)^m a_\ell^m$. Hence, the coefficients with $m<0$ can always be reconstructed from the coefficients with $m>0$.

CROISSANT simulates visibilities V given by $V(\nu, t) = \int \mathrm{d}\Omega A(\theta, \phi, \nu) T_{\rm sky} (\theta, \phi, \nu, t)$. Decomposing the beam and the sky to spherical harmonics and using the above properties yield a simple expression for the visibility:

$$A(\theta, \phi) = \sum_{\ell, m} a_\ell^m Y_{\ell}^m, T_{\rm sky}(\theta, \phi) = \sum_{\ell, m} b_\ell^m Y_{\ell}^m$$ 

$$V = \sum_\ell a_l^0 b_l^0 + \sum_{\ell, m>0} a_\ell^m \left(b_\ell^m\right)^* + \sum_{\ell, m<0} a_\ell^m \left(b_\ell^m\right)^*$$

$$V = \sum_\ell a_l^0 b_l^0 + \sum_{\ell, m>0} \left(a_\ell^m \left(b_\ell^m\right)^* + \left(a_\ell^m\right)^* b_\ell^m\right)$$

$$V = \sum_\ell a_l^0 b_l^0 + 2\sum_{\ell, m>0}\Re \left(a_\ell^m \left(b_\ell^m\right)^*\right)$$

$$V = \sum_\ell a_l^0 b_l^0 + 2\sum_{\ell, m>0}\left(\Re(a_\ell^m) \Re(b_\ell^m) + \Im(a_\ell^m) \Im(b_\ell^m)\right)$$

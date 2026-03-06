def compute_Y_ll(l, theta, phi):
    """Compute sectoral harmonics Y_l^l with Condon-Shortley phase convention."""
    log_factor = jax.scipy.special.gammaln(2*l + 1) - 2*l * jnp.log(2) - 2 * jax.scipy.special.gammaln(l + 1)
    # Full normalization with Condon-Shortley phase (-1)^l
    condonshortley = 1 - 2 * (l % 2)
    norm = condonshortley * jnp.sqrt((2*l + 1) / (4 * jnp.pi) * jnp.exp(log_factor))
    Y_ll = norm * (jnp.sin(theta) ** l) * jnp.exp(1j * l * phi)
    return Y_ll

def compute_Y_l_minusl(l, theta, phi):
    """Compute sectoral harmonics Y_l^-l with Condon-Shortley phase convention."""
    log_factor = jax.scipy.special.gammaln(2*l + 1) - 2*l * jnp.log(2) - 2 * jax.scipy.special.gammaln(l + 1)
    norm = jnp.sqrt((2*l + 1) / (4 * jnp.pi) * jnp.exp(log_factor))
    Y_l_minusl = norm * (jnp.sin(theta) ** l) * jnp.exp(-1j * l * phi)
    return Y_l_minusl

@partial(jax.jit, static_argnames=['l'])
def accumulate_sh_down(a_lm, l, theta, phi):
    y0 = jnp.zeros_like(theta, dtype=jnp.complex128)
    result = jnp.zeros_like(y0)
    m0 = l
    y_lm = compute_Y_ll(l, theta, phi)
    y_lmp1 = 0 * y_lm
    carry = (m0, y_lm, y_lmp1, result)

    def accumulate_sh_step(carry, x):
        """Accumulates the current element to the carry."""
        m, y_lm, y_lmp1, acc = carry
        acc = acc + y_lm * x
        a = jnp.exp(-1j * phi) / jnp.sqrt(l * (l + 1) - m * (m - 1)) 
        b = 2 * m / jnp.tan(theta) 
        c = jnp.sqrt(l * (l+1) - m * (m + 1)) * jnp.exp(-1j * phi)
        y_lmm1 = -a * (b * y_lm + c * y_lmp1)
        return (m - 1, y_lmm1, y_lm, acc), None
    
    final_carry, _ = jax.lax.scan(accumulate_sh_step, carry, a_lm)
    return final_carry[-1]

@partial(jax.jit, static_argnames=['l'])
def accumulate_sh_up(a_lm, l, theta, phi):
    y0 = jnp.zeros_like(theta, dtype=jnp.complex128)
    result = jnp.zeros_like(y0)
    m0 = -l
    y_lm = compute_Y_l_minusl(l, theta, phi)
    y_lmm1 = 0 * y_lm
    carry = (m0, y_lm, y_lmm1, result)

    def accumulate_sh_step(carry, x):
        m, y_lm, y_lmm1, acc = carry
        acc = acc + y_lm * x
        a = jnp.exp(1j * phi) / jnp.sqrt((l - m) * (l + m + 1)) 
        b = 2 * m / jnp.tan(theta) 
        c = jnp.sqrt((l + m) * (l - m + 1)) * jnp.exp(1j * phi)
        y_lmp1 = -a * (b * y_lm + c * y_lmm1)
        return (m + 1, y_lmp1, y_lm, acc), None
    
    final_carry, _ = jax.lax.scan(accumulate_sh_step, carry, a_lm)
    return final_carry[-1]

@partial(jax.jit, static_argnames=['L_max'])
def alm2points(a_lm, theta, phi, L_max):
    """Evaluate sum of spherical harmonics at given points with coefficients a_lm."""
    final_sum = jnp.zeros_like(theta, dtype=jnp.complex128)
    for l in range(L_max + 1):
        minus_l_to_0 = accumulate_sh_up(a_lm[l, L_max-l:L_max+1], l, theta, phi)
        l_to_1 = accumulate_sh_down(a_lm[l, L_max+l:L_max:-1], l, theta, phi)
        final_sum += minus_l_to_0 + l_to_1
    return final_sum
  

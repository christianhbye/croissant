import jax

# enable 64-bit precision globally for all tests
jax.config.update("jax_enable_x64", True)

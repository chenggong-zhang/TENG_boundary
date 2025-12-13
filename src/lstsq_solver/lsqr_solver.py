import jax
from jax import lax
from jax import tree_util
from functools import partial
import jax.numpy as jnp
from src.lstsq_solver.utils import normsq, tree_normsq

def lsqr_solve(A, At, b, maxiter=100, use_jax_scan=False):
    """
    A: jvp
    At: vjp
    b: reward
    """
    # Initial setup
    # b = jax.device_put_sharded(b, jax.local_devices())
    x0 = tree_util.tree_map(jnp.zeros_like, At(b))
    b_length = b.size
    x_length = tree_util.tree_reduce(jnp.add, tree_util.tree_map(jnp.size, x0))
    if maxiter is None:
        maxiter = min(x_length*2, b_length*2)

    beta0 = jnp.linalg.norm(b)
    u0 = b / beta0
    v0 = At(u0)
    alpha0 = jnp.linalg.norm(v0)
    v0 = v0 / alpha0
    w0 = v0
    phi_bar0 = beta0
    rho_bar0 = alpha0


    if use_jax_scan:
        def loop_body(val, k):
            x, beta, u, v, alpha, w, phi_bar, rho_bar, _ = val # the last one should be c but we don't need it
            u = A(v) - alpha * u
            beta = jnp.linalg.norm(u)
            u = u / beta
            v = At(u) - beta * v
            alpha = jnp.linalg.norm(v)
            v = v / alpha
            rho = jnp.sqrt(rho_bar**2 + beta**2)
            c = rho_bar / rho
            s = beta / rho
            theta = s * alpha
            rho_bar = -c * alpha
            phi = c * phi_bar
            phi_bar = s * phi_bar
            x = x + (phi / rho) * w
            w = v - (theta / rho) * w
            return (x, beta, u, v, alpha, w, phi_bar, rho_bar, c), None
        (x, _, _, _, alpha, _, phi_bar, _, c), _ = lax.scan(loop_body, (x0, beta0, u0, v0, alpha0, w0, phi_bar0, rho_bar0, 0), jnp.arange(maxiter))

    else:
        # setup parameters
        x, beta, u, v, alpha, w, phi_bar, rho_bar, c = x0, beta0, u0, v0, alpha0, w0, phi_bar0, rho_bar0, 0

        for k in range(maxiter):
            u = A(v) - alpha * u
            beta = jnp.linalg.norm(u)
            u = u / beta
            v = At(u) - beta * v
            alpha = jnp.linalg.norm(v)
            v = v / alpha
            rho = jnp.sqrt(rho_bar ** 2 + beta ** 2)
            c = rho_bar / rho
            s = beta / rho
            theta = s * alpha
            rho_bar = -c * alpha
            phi = c * phi_bar
            phi_bar = s * phi_bar
            x = x + (phi / rho) * w
            w = v - (theta / rho) * w

    return x, (maxiter, phi_bar, phi_bar * alpha * jnp.abs(c))  # None for 'info', can be extended later
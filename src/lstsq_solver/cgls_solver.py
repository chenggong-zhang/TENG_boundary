import jax
from jax import lax
from jax import tree_util
from functools import partial
import jax.numpy as jnp
from src.lstsq_solver.utils import normsq, tree_normsq


# normsq = jax.jit(normsq)
# tree_normsq = jax.jit(tree_normsq)

def cgls_solve(A, At, b, x0=None, tol=1e-7, atol=0.0, maxiter=None):
    """
    A: jvp
    At: vjp
    b: reward
    """
    # Initial setup
    # b = jax.device_put_sharded(b, jax.local_devices())
    if x0 is None:
        x0 = tree_util.tree_map(jnp.zeros_like, At(b))
    b_length = b.size
    x_length = tree_util.tree_reduce(jnp.add, tree_util.tree_map(jnp.size, x0))
    if maxiter is None:
        maxiter = min(x_length*2, b_length*2)

    # Tolerance and initial residual
    r0 = b - A(x0)
    s0 = p0 = At(r0)
    gamma0 = tree_normsq(s0) # somehow this is the same as cg No need to modify later
    atol2 = jnp.maximum(tol**2 * gamma0, atol**2)


    # Core CGLS loop conditions and body
    def cond_fun(value):
        _, _, gamma, _, k = value
        # print(k, gamma)
        return (gamma > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        q = A(p)
        alpha = gamma / normsq(q)
        x_new = x + alpha * p
        r_new = r - alpha * q
        s_new = At(r_new)
        gamma_new = tree_normsq(s_new)
        beta = gamma_new / gamma
        p_new = s_new + beta * p
        return x_new, r_new, gamma_new, p_new, k + 1

    # Run the loop
    initial_value = (x0, r0, gamma0, p0, 0)
    x_final, _, gamma_final, _, nb_iters = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final, (nb_iters, jnp.sqrt(gamma_final))  # None for 'info', can be extended later



def cgls_solve2(A, At, b, x0=None, maxiter=100, use_jax_scan=False):
    """
    A: jvp
    At: vjp
    b: reward
    """
    # Initial setup
    # b = jax.device_put_sharded(b, jax.local_devices())
    if x0 is None:
        x0 = tree_util.tree_map(jnp.zeros_like, At(b))
    b_length = b.size
    x_length = tree_util.tree_reduce(jnp.add, tree_util.tree_map(jnp.size, x0))
    if maxiter is None:
        maxiter = min(x_length*2, b_length*2)

    # Tolerance and initial residual
    r0 = b - A(x0)
    s0 = p0 = At(r0)
    gamma0 = tree_normsq(s0) # somehow this is the same as cg No need to modify later

    # if use_jax_fori_loop:
    #     def loop_body(k, val):
    #         x, r, gamma, p = val
    #         q = A(p)
    #         alpha = gamma / normsq(q)
    #         x = x + alpha * p
    #         r = r - alpha * q
    #         s = At(r)
    #         gamma_new = tree_normsq(s)
    #         beta = gamma_new / gamma
    #         gamma = gamma_new
    #         p = s + beta * p
    #         return x, r, gamma, p
    #     x, r, gamma, p = lax.fori_loop(0, maxiter, loop_body, (x0, r0, gamma0, p0))

    if use_jax_scan:
        def loop_body(val, k):
            x, r, gamma, p = val
            q = A(p)
            alpha = gamma / normsq(q)
            x = x + alpha * p
            r = r - alpha * q
            s = At(r)
            gamma_new = tree_normsq(s)
            beta = gamma_new / gamma
            gamma = gamma_new
            p = s + beta * p
            return (x, r, gamma, p), None
        (x, r, gamma, p), _ = lax.scan(loop_body, (x0, r0, gamma0, p0), jnp.arange(maxiter))

    else:
        # setup parameters
        x, r, gamma, p = x0, r0, gamma0, p0

        for k in range(maxiter):
            q = A(p)
            alpha = gamma / normsq(q)
            x = x + alpha * p
            r = r - alpha * q
            s = At(r)
            gamma_new = tree_normsq(s)
            beta = gamma_new / gamma
            gamma = gamma_new
            p = s + beta * p

    return x, (maxiter, jnp.sqrt(gamma))  # None for 'info', can be extended later
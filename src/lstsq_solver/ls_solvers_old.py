from jax import lax, device_put, tree_map, tree_leaves, tree_structure
# from jax import partial
import jax.numpy as jnp
# import numpy as np

def cg(A, b, x0=None, tol=1e-5, atol=0.0, maxiter=None, M=None):
    # Initial setup
    if x0 is None:
        x0 = tree_map(jnp.zeros_like, b)
    b, x0 = device_put((b, x0))
    if maxiter is None:
        size = sum(bi.size for bi in tree_leaves(b))
        maxiter = 10 * size
    if M is None:
        M = lambda x: x  # Identity function as default preconditioner

    # Check tree structures and shapes
    if tree_structure(x0) != tree_structure(b):
        raise ValueError('x0 and b must have matching tree structure.')
    # Add more checks if needed

    # Tolerance and initial residual
    bs = jnp.vdot(b, b).real
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))
    r0 = b - A(x0)
    p0 = z0 = M(r0)
    gamma0 = jnp.vdot(r0, z0).real

    # Core CG loop conditions and body
    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma if M is None else jnp.vdot(r, r).real
        # print(k, rs)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / jnp.vdot(p, Ap).real
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M(r_new)
        gamma_new = jnp.vdot(r_new, z_new).real
        beta = gamma_new / gamma
        p_new = z_new + beta * p
        return x_new, r_new, gamma_new, p_new, k + 1

    # Run the loop
    initial_value = (x0, r0, gamma0, p0, 0)
    x_final, *_, nb_iters = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final, nb_iters  # None for 'info', can be extended later



def cgls(A, At, b, x0=None, tol=1e-5, atol=0.0, maxiter=None):
    # Initial setup
    if x0 is None:
        x0 = jnp.zeros(At(b).shape, dtype=b.dtype)  # x0 has the shape of columns of A
    b, x0 = device_put((b, x0))
    if maxiter is None:
        maxiter = 10 * b.size

    # Tolerance and initial residual
    r0 = b - A(x0)
    s0 = p0 = At(r0)
    gamma0 = jnp.vdot(s0, s0).real # somehow this is the same as cg No need to modify later
    atol2 = jnp.maximum(tol**2 * gamma0, atol**2)

    # # use similar stop creteria as cg on normal equation
    # Atb = At(b)
    # atol2 = jnp.maximum(tol ** 2 * jnp.vdot(Atb, Atb).real, atol ** 2)


    # Core CGLS loop conditions and body
    def cond_fun(value):
        _, _, gamma, _, k = value
        # print(k, gamma)
        return (gamma > atol2) & (k < maxiter)

    # use similar stop creteria as cg on normal equation
    # def cond_fun(value):
    #     x, r, gamma, _, k = value
    #     residual_normal_eq = At(r)  # This is A^T (b - Ax)
    #     gamma_normal_eq = jnp.vdot(residual_normal_eq, residual_normal_eq).real
    #     print(k, gamma_normal_eq, gamma)
    #     return (gamma_normal_eq > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        q = A(p)
        alpha = gamma / jnp.vdot(q, q).real
        x_new = x + alpha * p
        r_new = r - alpha * q
        s_new = At(r_new)
        gamma_new = jnp.vdot(s_new, s_new).real
        beta = gamma_new / gamma
        p_new = s_new + beta * p
        return x_new, r_new, gamma_new, p_new, k + 1

    # Run the loop
    initial_value = (x0, r0, gamma0, p0, 0)
    x_final, *_, nb_iters = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final, nb_iters  # None for 'info', can be extended later
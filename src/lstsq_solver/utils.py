import jax.numpy as jnp
from jax import tree_util

def normsq(x):
    x = x.ravel()
    return jnp.sum(jnp.square(x.real) + jnp.square(x.imag))

def tree_normsq(x):
    return tree_util.tree_reduce(jnp.add, tree_util.tree_map(normsq, x))
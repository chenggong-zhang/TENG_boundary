from collections import namedtuple
import jax

# AutoJittedTuple = namedtuple('AutoJittedTuple', 'nonjitted, jitted')
#
# def autojit(func, *args, **kwargs):
#     """
#     return a named tuple containing both the original (nonjitted) and jitted version of the input function
#
#     """
#     return AutoJittedTuple(func, jax.jit(func, *args, **kwargs))
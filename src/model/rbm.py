from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[
    None,
    str,
    lax.Precision,
    Tuple[str, str],
    Tuple[lax.Precision, lax.Precision],
]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]

def logcosh(x, clip_max=1e20): # have to clip this to avoid infinity, somehow infinity divide 2 results in a nan in jax
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return jnp.clip(x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0), -clip_max, clip_max)

class SqrtRBM(nn.Module):
    """
    sqrt rbm with log_psi(x) = a dot x + sum(logcosh(W @ x + b) / 2)
    """
    nb_sites: int
    hidden_dim: int
    use_bias: bool = True # do we use b
    use_aias: bool = True # do we use a
    param_dtype: Dtype = complex
    weight_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    aias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # setup parameters
        weight = self.param(
            'weight',
            self.weight_init,
            (self.nb_sites, self.hidden_dim),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (self.hidden_dim,), self.param_dtype
            )
        else:
            bias = None
        if self.use_aias:
            aias = self.param(
                'aias', self.aias_init, (self.nb_sites,), self.param_dtype
            )
        else:
            aias = None

        # compute result
        x = 2 * x - 1
        y = x @ weight
        if bias is not None:
            y = y + bias
        y = logcosh(y).sum(-1) / 2 # devide by 2 for sqrt, we may not need it
        if aias is not None:
            y = y + (x @ aias)
        return y
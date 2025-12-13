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
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
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



def init_normal(stddev=1, normalize=True, dtype=complex):
    """
    initialize psi with normal random numbers
    Note: returned log_psi is not normal random numbers, but log_normal random numbers
    """
    init_func = initializers.normal(stddev=stddev, dtype=dtype)
    def new_init_func(*args, **kwargs):
        psi = init_func(*args, **kwargs)
        assert psi.ndim == 1
        if normalize:
            psi /= jnp.linalg.norm(psi)
        return jnp.log(psi)
    return new_init_func

def init_value(psi=None, log_psi=None):
    assert (psi is None) != (log_psi is None)  # this is xor, trust me
    if psi is not None:
        log_psi = jnp.log(psi)
    def init_func(*args, **kwargs):
        return log_psi
    return init_func

class FullParam(nn.Module):
    nb_sites: int
    local_dim: int
    normalize: bool = False
    state_init: Callable[[PRNGKey, Shape, Dtype], Array] = init_normal()

    @nn.compact
    def __call__(self, s):
        # params should be log psi
        # log_psi = self.param('log_psi', nn.initializers.normal(stddev=1, dtype=jVMC.global_defs.tCpx), (2**self.nb_sites, ))
        # log_psi = self.param('log_psi', lambda a, b: self.init_state, (2 ** self.nb_sites,))
        log_psi = self.param('log_psi', self.state_init, (self.local_dim ** self.nb_sites,)) # hope I don't need to do this, but I didn't find a better way
        if self.normalize:
            log_psi -= jsp.special.logsumexp(log_psi.real * 2) / 2
        powers_of_d = jnp.flip(jnp.array([self.local_dim ** i for i in range(self.nb_sites)]))
        inds = (s.astype(int) * powers_of_d).sum(-1)
        return log_psi[inds]


if __name__ == '__main__':
    net = FullParam(nb_sites=4)
    from src.sampler import SimpleExactSampler
    sampler = SimpleExactSampler(nb_sites=4)
    from src.var_state import SimpleVarState
    var_state = SimpleVarState(net, 4, sampler)
    print()
import timeit
from functools import partial

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
import numpy as np
import jax
from jax import lax
from jax import random
from jax import tree_util
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn
from flax.linen import initializers
from simple_var_state import SimpleVarState
from src.sampler import SimpleMCMCSampler, SimpleExactSampler
from src.utils import quantum_ng_ls, quantum_ng_minsr, quantum_ng_tdvp
from src.lstsq_solver import cgls_solve

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

jax.config.update("jax_enable_x64", True)


class FullParametrization(nn.Module):
    nb_sites: int
    # init_state: jax.Array
    normalize: bool = False

    @nn.compact
    def __call__(self, s):
        # params should be log psi
        # log_psi = self.param('log_psi', nn.initializers.normal(stddev=1, dtype=jVMC.global_defs.tCpx), (2**self.nb_sites, ))
        # log_psi = self.param('log_psi', lambda a, b: self.init_state, (2 ** self.nb_sites,))
        log_psi = self.param('log_psi', lambda a, b: jnp.log(psi0), (2 ** self.nb_sites,)) # hope I don't need to do this, but I didn't find a better way
        if self.normalize:
            log_psi -= jsp.special.logsumexp(log_psi.real * 2) / 2
        powers_of_two = jnp.flip(jnp.array([2 ** i for i in range(self.nb_sites)]))
        inds = (s.astype(int) * powers_of_two).sum(-1)
        return log_psi[inds].real * 2 * log_psi[(inds + 3) % (2 ** self.nb_sites)].imag / log_psi[(inds//2) % (2 ** self.nb_sites)].real + 1j * log_psi[inds].imag / 3 / log_psi[(inds*5) % (2 ** self.nb_sites)].imag * log_psi[(inds-1) % (2 ** self.nb_sites)].real

# copied from jvmc
def logcosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

class RBM(nn.Module):
    """
    sqrt rbm with log_psi(x) = a dot x + sum(logcosh(W @ x + b) / 2)
    """
    nb_sites: int
    hidden_dim: int
    use_bias: bool = True # do we use b
    use_aias: bool = True # do we use a
    param_dtype: Dtype = float
    weight_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal() #initializers.zeros_init()
    aias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal() #initializers.zeros_init()

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


def ints_to_binary(inds, nb_sites):
    """Convert a batch of integers to their binary representations using JAX."""
    # Create a matrix where each row is the powers of two for bitwise AND operation
    powers_of_two = jnp.array([1 << i for i in range(nb_sites-1, -1, -1)])

    # Compute the binary representation using bitwise AND
    binary_matrix = jnp.bitwise_and(inds[:, jnp.newaxis], powers_of_two) > 0
    return binary_matrix.astype(jnp.int32)


def binary_to_decimal(binary_number):
    positions = jnp.flip(jnp.array([2**i for i in range(binary_number.shape[-1])]))
    decimal_number = jnp.dot(binary_number, positions)
    return decimal_number


def gen_full_basis(nb_sites):
    return ints_to_binary(jnp.arange(2**nb_sites), nb_sites)


def test_sampling(var_state):
    basis = gen_full_basis(var_state.nb_sites)[None, :, :]
    log_psi = var_state.log_psi(basis)
    probs = jnp.exp(2 * log_psi.real).squeeze(0)
    probs /= jnp.sum(probs)
    for nb_samples in [1000, 4000, 16000, 64000, 256000, 1024000]:
        var_state.sampler.nb_samples = nb_samples
        var_state.sampler.nb_sample_steps = nb_samples // (var_state.sampler.nb_chains_per_device * var_state.sampler.nb_devices)
        samples, log_psi, weights = var_state.sample()
        ind = binary_to_decimal(samples).squeeze(0)
        freqs = jnp.bincount(ind)
        freqs /= jnp.sum(freqs)
        print(jnp.linalg.norm(freqs - probs, 1))
    return

def test_jacobian(var_state, jvp, vjp, jacobian):
    jacobian = jacobian.squeeze(0)
    cotangent = np.random.randn(16) + 1j * np.random.randn(16)
    jacobian_split = jnp.concatenate([jacobian.real, jacobian.imag])
    cotangent_split = jnp.concatenate([cotangent.real, cotangent.imag])
    tangent = np.random.randn(29)  + 1j * np.random.randn(29)
    tangent_split = jnp.concatenate([tangent.real, tangent.imag])
    print()


def test_jacobian2(var_state, samples, sqrt_weights):
    jvp_raw, vjp_raw, log_psi = var_state.jvp_vjp_log_psi_func(samples, jit=False)
    weights = jnp.square(sqrt_weights)
    vjp_raw_weights_real = vjp_raw(weights + 0j)
    vjp_raw_weights_imag = vjp_raw(weights * 1j)

    def jvp(tangents):
        pushforwards = jvp_raw(tangents)
        pushforwards = pushforwards - (weights * pushforwards).sum()
        return pushforwards * sqrt_weights

    def vjp(cotangents):
        cotangents = cotangents * sqrt_weights
        pullbacks = vjp_raw(cotangents)
        cotangents_sum = cotangents.sum()
        pullbacks_mean = cotangents_sum.real * vjp_raw_weights_real + cotangents_sum.imag * vjp_raw_weights_imag
        # pullbacks_mean = tree_util.tree_map(lambda xr, xi: xr * cotangents_sum.real + xi * cotangents_sum.imag, vjp_raw_weights_real, vjp_raw_weights_imag)
        # pullbacks_mean = tree_util.tree_map(lambda x: x * cotangents_sum,
        #                                     tree_util.tree_map(jnp.add, vjp_raw(weights + 0j), vjp_raw(-1j * weights)))  # vjp_raw(weights) * cotangents.sum()
        return pullbacks - pullbacks_mean
        # return tree_util.tree_map(jnp.subtract, pullbacks, pullbacks_mean)

    jacobian_raw = var_state.jac_log_psi(samples).squeeze(0)
    jacobian = jacobian_raw - (jacobian_raw * weights[0, :, None]).sum(0)
    jacobian = jacobian * sqrt_weights[0, :, None]
    cotangent = np.random.randn(1000) + 1j * np.random.randn(1000)
    jacobian_split = jnp.concatenate([jacobian.real, jacobian.imag])
    cotangent_split = jnp.concatenate([cotangent.real, cotangent.imag])
    tangent = np.random.randn(29) + 1j * np.random.randn(29)
    tangent_split = jnp.concatenate([tangent.real, tangent.imag])
    jacobian_raw_weighted = jacobian_raw * sqrt_weights[0, :, None]
    jacobian_raw_weighted_split = jnp.concatenate([jacobian_raw_weighted.real, jacobian_raw_weighted.imag])
    term1 = cotangent_split @ jacobian_raw_weighted_split
    jacobian_mean = (jacobian_raw * weights[0, :, None]).sum(0, keepdims=True)
    jacobian_mean_weighted = jacobian_mean * sqrt_weights[0, :, None]
    jacobian_mean_weighted_split = jnp.concatenate([jacobian_mean_weighted.real, jacobian_mean_weighted.imag])
    term2 = cotangent_split @ jacobian_mean_weighted_split
    cotangent_weighted_sum = cotangent @ sqrt_weights[0]
    print(jnp.allclose(jvp(tangent_split), jacobian @ tangent_split))
    # print(jnp.allclose(jvp(tangent), jacobian @ tangent))
    print(jnp.allclose(vjp(cotangent[None, :]), cotangent_split @ jacobian_split))
    print()

def test_ng(var_state, samples, sqrt_weights):
    ls_solver = partial(cgls_solve, x0=None, tol=1e-14, atol=0.0, maxiter=None)
    rewards = np.random.randn(*sqrt_weights.shape) + 1j * np.random.randn(*sqrt_weights.shape)
    rewards = jnp.array(rewards)
    quantum_ng_ls(var_state, samples, sqrt_weights, rewards, ls_solver, joint_jit=True)
    quantum_ng_minsr(var_state, samples, sqrt_weights, rewards)
    print()






def test():
    global psi0
    nb_sites = 4
    # psi0 = jnp.ones(2**nb_sites, dtype=complex) / (2**(nb_sites/2))
    psi0 = np.random.randn(2**nb_sites) + 1j * np.random.randn(2**nb_sites)
    # psi0 /= np.linalg.norm(psi0)
    net = RBM(nb_sites=nb_sites, hidden_dim=5, param_dtype=complex)
    # sampler = SimpleExactSampler(nb_sites, init_sample=None, nb_chains_per_device=1000, nb_samples=1000, chain_length=50,
    #              rand_seed=1234)
    sampler = SimpleMCMCSampler(nb_sites, init_sample=None, nb_chains_per_device=1000, nb_samples=1000, chain_length=50,
                 rand_seed=1234)
    # var_state = Simplevar_state(FullParametrization(nb_sites, normalize=False), nb_sites, sampler=sampler,
    #                 init_seed=1234)
    var_state = SimpleVarState(net, nb_sites, sampler=sampler, init_seed=1234)
    sampler.thermalize(chain_length=200)
    samples = jnp.zeros((1, 1, 4))
    basis = gen_full_basis(nb_sites)[None, :, :]
    jvp, vjp, log_psi = var_state.jvp_vjp_log_psi_func(basis, return_value=True, jit=False)
    jacobian = var_state.jac_log_psi(basis)
    samples, log_psi, sqrt_weights = var_state.sample()
    test_ng(var_state, samples, sqrt_weights)
    # test_jacobian(var_state, jvp, vjp, jacobian)
    # test_jacobian2(var_state, samples, sqrt_weights)
    # print(var_state.jac_log_psi(basis))
    # print(jvp(var_state.get_parameters()))
    # print(vjp(log_psi))
    # print(timeit.timeit(partial(var_state.jac_log_psi, samples=basis), number=50))
    # print(timeit.timeit(partial(jvp, tangents=var_state.get_parameters()), number=50))
    # print(timeit.timeit(partial(vjp, cotangents=log_psi), number=50))

    # test_sampling(var_state)
    assert False

if __name__ == '__main__':
    test()



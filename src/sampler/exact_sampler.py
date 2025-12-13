from functools import partial
import pickle
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrnd

from src.sampler.abstract_sampler import AbstractSampler


class SimpleExactSampler(AbstractSampler):
    def __init__(self, nb_sites, *args, **kwargs):
        """
        Exactly enumerate all the basis and put weights to each basis according to |psi|^2
        args and kwargs are just place holders for easy debug (no need to delete old arguments)
        args and kwargs will not be used
        """
        super().__init__()
        self.is_exact_sampler = True
        self.nb_sites = nb_sites
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        self.var_state = None
        self.basis = jax.device_put_replicated(self.gen_basis(self.nb_sites), self.devices) # duplicate to all devices to be compatible with pmap

    def ints_to_binary(self, inds, nb_sites):
        """
        Convert a batch of integers to their binary representations.
        """
        # Create a matrix where each row is the powers of two for bitwise AND operation
        powers_of_two = jnp.array([1 << i for i in range(nb_sites-1, -1, -1)])

        # Compute the binary representation using bitwise AND
        binary_matrix = jnp.bitwise_and(inds[:, jnp.newaxis], powers_of_two) > 0
        return binary_matrix.astype(jnp.int32)

    def gen_basis(self, nb_sites):
        return self.ints_to_binary(jnp.arange(2**nb_sites), nb_sites)


    def set_var_state(self, var_state):
        """
        need to call this function before the sampler can be used
        """
        assert self.var_state is None, "var_state is already set, please create a new sampler"
        self.var_state = var_state

    def sample(self):
        log_psi = self.var_state.log_psi(self.basis)
        sqrt_weights = jnp.exp(log_psi.real - jsp.special.logsumexp(2 * log_psi.real) / 2) # same as psi/||psi||_2
        return self.basis, log_psi, sqrt_weights

    def thermalize(self, *args, **kwargs):
        """
        just to be compatible with code designed for mcmc sampler
        """
        return self.sample()

    def get_state(self):
        """
        no state to get
        """
        return {}

    def set_state(self, state):
        """
        no state to set
        """
        return

    def save_state(self, path):
        """
        no state to save
        """
        return

    def load_state(self, path):
        """
        no state to load
        """
        return


class QuditExactSampler(AbstractSampler):
    def __init__(self, nb_sites, local_dim, *args, **kwargs):
        """
        Exactly enumerate all the basis and put weights to each basis according to |psi|^2
        args and kwargs are just place holders for easy debug (no need to delete old arguments)
        args and kwargs will not be used
        """
        super().__init__()
        self.is_exact_sampler = True
        self.nb_sites = nb_sites
        self.local_dim = local_dim
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        self.var_state = None
        self.basis = jax.device_put_replicated(self.gen_basis(), self.devices) # duplicate to all devices to be compatible with pmap

    def gen_basis(self, inds=None):
        if inds is None:
            inds = jnp.arange(self.local_dim ** self.nb_sites)
        bases = []
        for i in range(self.nb_sites):
            inds, digits = jnp.divmod(inds, self.local_dim)
            bases.append(digits)
        bases = jnp.stack(bases[::-1], 1)
        return bases

    def gen_index(self, bases=None):
        if bases is None:
            return jnp.arange(self.local_dim ** self.nb_sites)
        digit_base = self.local_dim ** jnp.arange(self.nb_sites - 1, -1, -1)
        return (bases * digit_base).sum(-1)

    def set_var_state(self, var_state):
        """
        need to call this function before the sampler can be used
        """
        self.var_state = var_state

    def sample(self):
        log_psi = self.var_state.log_psi(self.basis)
        sqrt_weights = jnp.exp(log_psi.real - jsp.special.logsumexp(2 * log_psi.real) / 2) # same as psi/||psi||_2
        return self.basis, log_psi, sqrt_weights

    def thermalize(self, *args, **kwargs):
        """
        just to be compatible with code designed for mcmc sampler
        """
        return self.sample()

    def get_state(self):
        """
        no state to get
        """
        return {}

    def set_state(self, state):
        """
        no state to set
        """
        return

    def save_state(self, path):
        """
        no state to save
        """
        return

    def load_state(self, path):
        """
        no state to load
        """
        return

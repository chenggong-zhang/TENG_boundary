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
from dataclasses import dataclass
import pickle
from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jrnd

from src.sampler import AbstractSampler

NN = Any
Sampler = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PyTreeDef = Any

@dataclass(frozen=True)
class SimpleMCMCSamplerPure:
    var_state_pure: Any
    nb_sites: int

    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
    def move_func(self, sample, rand_key):
        """
        remember the following arguments are vmapped
        sample: a single sample of shape (nb_sites, )
        rand_key should be a single key
        """
        rand_site, = jrnd.randint(rand_key, (1,), 0, self.nb_sites)
        sample = sample.at[rand_site].set(1 - sample[rand_site])
        return sample

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def gen_accept_mask(self, new_log_psi, curr_log_psi, rand_key):
        prob_ratio = jnp.clip(jnp.exp((new_log_psi.real - curr_log_psi.real) * 2), a_min=None, a_max=1)
        accept_mask = jrnd.bernoulli(rand_key, prob_ratio)
        return accept_mask

    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0, 0))
    def split_keys(self, rand_key):
        rand_key1, rand_key2 = jrnd.split(rand_key)
        return rand_key1, rand_key2

    def update_sample(self, curr_samples, curr_log_psi, state, rand_keys):
        rand_keys1, rand_keys2 = self.split_keys(rand_keys)
        new_samples = self.move_func(curr_samples, rand_keys1)
        new_log_psi = self.var_state_pure.log_psi(state, new_samples)
        accept_mask = self.gen_accept_mask(new_log_psi, curr_log_psi, rand_keys2)
        repeated_mask = jnp.repeat(accept_mask[:, None], self.nb_sites, axis=1)
        new_samples = lax.select(repeated_mask, new_samples, curr_samples)
        new_log_psi = lax.select(accept_mask, new_log_psi, curr_log_psi)
        return new_samples, new_log_psi, accept_mask

    def sample_chain_body_func(self, i, val):
        curr_samples, curr_log_psi, accept_mask, state, rand_keys = val
        rand_keys, sub_rand_keys = self.split_keys(rand_keys)
        val = (*self.update_sample(curr_samples, curr_log_psi, state, sub_rand_keys), state, rand_keys)
        return val

    def sample_chain(self, chain_length, curr_samples, curr_log_psi, state, rand_keys):
        init_val = curr_samples, curr_log_psi, jnp.zeros_like(curr_log_psi,
                                                              dtype=bool), state, rand_keys  # zeros is a placeholder for accept mask
        return lax.fori_loop(0, chain_length, self.sample_chain_body_func, init_val)

    def sample_steps_body_func(self, carry, x):
        chain_length, curr_samples, curr_log_psi, accept_mask, state, rand_keys = carry
        val = self.sample_chain(chain_length, curr_samples, curr_log_psi, state, rand_keys)
        new_samples, new_log_psi, accept_mask, state, rand_keys = val
        new_carry = chain_length, new_samples, new_log_psi, accept_mask, state, rand_keys
        y = new_samples, new_log_psi
        return new_carry, y

    @partial(jax.pmap, in_axes=(None, None, None, 0, 0, None, 0), out_axes=(0, 0, 0, 0, 0, 0),
             static_broadcasted_argnums=(0, 1, 2))
    def sample_steps(self, nb_sample_steps, chain_length, curr_samples, curr_log_psi, state, rand_keys):
        init = chain_length, curr_samples, curr_log_psi, jnp.zeros_like(curr_log_psi,
                                                                        dtype=bool), state, rand_keys  # zeros is a placeholder for accept mask
        carry, ys = lax.scan(self.sample_steps_body_func, init, xs=None, length=nb_sample_steps)
        _, new_samples, new_log_psi, accept_mask, _, rand_keys = carry
        # new_samples is the samples from last mcmc step
        # all_samples is the samples collected from all mcmc steps
        all_samples, all_log_psi = ys
        all_samples = all_samples.reshape(-1, all_samples.shape[-1])
        all_log_psi = all_log_psi.ravel()
        return all_samples, all_log_psi, new_samples, new_log_psi, accept_mask, rand_keys

class SimpleMCMCSampler(AbstractSampler):
    def __init__(self, nb_sites, init_sample=None,
                 nb_chains_per_device=10, nb_samples=1000, chain_length=10,
                 rand_seed=1234):
        """
        nb_sites: number of qudits
        init_sample: starting samples, if not provide will default to all zeros
        nb_chains: number of chains to parallely sample per device
        nb_samples: number of samples in total (combined for all devices)
        chain_length: length of chain to get each sample
        rand_seed: starting random seed
        Note: to reproduce the same result, keeping nb_chains * nb_devices constant is sufficient
        """
        super().__init__()
        self.pure_funcs = None # needs to set using set_var_state before actually using this
        self.var_state = None
        self.nb_sites = nb_sites
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        if init_sample is not None:
            self.init_sample = init_sample
        else:
            self.init_sample = jnp.zeros(nb_sites, dtype=int)
        self.nb_chains_per_device = nb_chains_per_device
        self.nb_samples = nb_samples
        assert self.nb_samples % (self.nb_chains_per_device * self.nb_devices) == 0, f'{nb_samples=} must be divisible by {nb_chains_per_device=} * {jax.local_devices_count()=}'
        self.nb_sample_steps = nb_samples // (self.nb_chains_per_device * self.nb_devices)
        self.chain_length = chain_length
        # keep track of last sample of each chain
        self.curr_samples = jax.device_put_sharded([jnp.array([self.init_sample] * self.nb_chains_per_device)] * self.nb_devices, self.devices)
        self.curr_log_psi = None
        self.accept_mask = None # keep track of the last accept mask
        self.accept_rate = None # keep track of the last accept rate
        rand_key = jax.random.PRNGKey(rand_seed)
        self.rand_keys = jrnd.split(rand_key, num=self.nb_chains_per_device * self.nb_devices).reshape(self.nb_devices, self.nb_chains_per_device, 2) # each key is dim 2

    def set_var_state(self, var_state):
        """
        need to call this function before the sampler can be used
        """
        assert self.var_state is None, "var_state is already set, please create a new sampler"
        self.var_state = var_state
        # all the pure helper functions are collected in MCMCSamplerPure, which is immutable.
        self.pure_funcs = SimpleMCMCSamplerPure(var_state.pure_funcs, self.nb_sites)
        self.curr_log_psi = var_state.log_psi(self.curr_samples)

    def sample(self):
        """
        samples from the wave function given by self.var_state
        return: samples, log_psi corresponding to the samples and an array of ones as the weights of the samples
        """
        samples, log_psi, \
        curr_samples, curr_log_psi, \
        accept_mask, rand_keys = self.pure_funcs.sample_steps(self.nb_sample_steps, self.chain_length,
                                                              self.curr_samples, self.curr_log_psi,
                                                              self.var_state.get_state(), self.rand_keys)
        # samples have shape (nb_devices, nb_sample_steps, nb_chains_per_device, nb_sites)
        # log_psi have shape (
        self.curr_samples = curr_samples
        self.curr_log_psi = curr_log_psi
        self.rand_keys = rand_keys
        self.accept_mask = accept_mask
        self.accept_rate = accept_mask.mean()
        return samples, log_psi, jnp.ones_like(log_psi, dtype=float) / jnp.sqrt(log_psi.size) # sqrt_weights

    def thermalize(self, chain_length):
        _, _, \
        curr_samples, curr_log_psi, \
        accept_mask, rand_keys = self.pure_funcs.sample_steps(1, chain_length,
                                                              self.curr_samples, self.curr_log_psi,
                                                              self.var_state.get_state(), self.rand_keys)
        self.curr_samples = curr_samples
        self.curr_log_psi = curr_log_psi
        self.rand_keys = rand_keys
        self.accept_mask = accept_mask
        self.accept_rate = accept_mask.mean()
        return curr_samples, curr_log_psi


    def get_state(self):
        return {'curr_samples': self.curr_samples,
                'curr_log_psi': self.curr_log_psi,
                'accept_mask': self.accept_mask,
                'accept_rate':self.accept_rate,
                'rand_keys':self.rand_keys}

    def set_state(self, state):
        self.curr_samples = state['curr_samples']
        self.curr_log_psi = state['curr_log_psi']
        self.accept_mask = state['accept_mask']
        self.accept_rate = state['accept_rate']
        self.rand_keys = state['rand_keys']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))




### not tested yet but should work ###
@dataclass(frozen=True)
class QuditMCMCSamplerPure:
    var_state_pure: Any
    nb_sites: int
    local_dim: int

    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
    def move_func(self, sample, rand_key):
        """
        remember the following arguments are vmapped
        sample: a single sample of shape (nb_sites, )
        rand_key should be a single key
        """
        rand_site_key, rand_shift_key = jrnd.split(rand_key)
        rand_site, = jrnd.randint(rand_site_key, (1,), 0, self.nb_sites)
        rand_shift, = jrnd.randint(rand_shift_key, (1,), 1, self.local_dim)
        sample = sample.at[rand_site].set((sample[rand_site] + rand_shift) % self.local_dim)
        return sample

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def gen_accept_mask(self, new_log_psi, curr_log_psi, rand_key):
        prob_ratio = jnp.clip(jnp.exp((new_log_psi.real - curr_log_psi.real) * 2), a_min=None, a_max=1)
        accept_mask = jrnd.bernoulli(rand_key, prob_ratio)
        return accept_mask

    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0, 0))
    def split_keys(self, rand_key):
        rand_key1, rand_key2 = jrnd.split(rand_key)
        return rand_key1, rand_key2

    def update_sample(self, curr_samples, curr_log_psi, state, rand_keys):
        rand_keys1, rand_keys2 = self.split_keys(rand_keys)
        new_samples = self.move_func(curr_samples, rand_keys1)
        new_log_psi = self.var_state_pure.log_psi(state, new_samples)
        accept_mask = self.gen_accept_mask(new_log_psi, curr_log_psi, rand_keys2)
        repeated_mask = jnp.repeat(accept_mask[:, None], self.nb_sites, axis=1)
        new_samples = lax.select(repeated_mask, new_samples, curr_samples)
        new_log_psi = lax.select(accept_mask, new_log_psi, curr_log_psi)
        return new_samples, new_log_psi, accept_mask

    def sample_chain_body_func(self, i, val):
        curr_samples, curr_log_psi, accept_mask, state, rand_keys = val
        rand_keys, sub_rand_keys = self.split_keys(rand_keys)
        val = (*self.update_sample(curr_samples, curr_log_psi, state, sub_rand_keys), state, rand_keys)
        return val

    def sample_chain(self, chain_length, curr_samples, curr_log_psi, state, rand_keys):
        init_val = curr_samples, curr_log_psi, jnp.zeros_like(curr_log_psi,
                                                              dtype=bool), state, rand_keys  # zeros is a placeholder for accept mask
        return lax.fori_loop(0, chain_length, self.sample_chain_body_func, init_val)

    def sample_steps_body_func(self, carry, x):
        chain_length, curr_samples, curr_log_psi, accept_mask, state, rand_keys = carry
        val = self.sample_chain(chain_length, curr_samples, curr_log_psi, state, rand_keys)
        new_samples, new_log_psi, accept_mask, state, rand_keys = val
        new_carry = chain_length, new_samples, new_log_psi, accept_mask, state, rand_keys
        y = new_samples, new_log_psi
        return new_carry, y

    @partial(jax.pmap, in_axes=(None, None, None, 0, 0, None, 0), out_axes=(0, 0, 0, 0, 0, 0),
             static_broadcasted_argnums=(0, 1, 2))
    def sample_steps(self, nb_sample_steps, chain_length, curr_samples, curr_log_psi, state, rand_keys):
        init = chain_length, curr_samples, curr_log_psi, jnp.zeros_like(curr_log_psi,
                                                                        dtype=bool), state, rand_keys  # zeros is a placeholder for accept mask
        carry, ys = lax.scan(self.sample_steps_body_func, init, xs=None, length=nb_sample_steps)
        _, new_samples, new_log_psi, accept_mask, _, rand_keys = carry
        # new_samples is the samples from last mcmc step
        # all_samples is the samples collected from all mcmc steps
        all_samples, all_log_psi = ys
        all_samples = all_samples.reshape(-1, all_samples.shape[-1])
        all_log_psi = all_log_psi.ravel()
        return all_samples, all_log_psi, new_samples, new_log_psi, accept_mask, rand_keys



### not tested yet but should work ###
class QuditMCMCSampler(AbstractSampler):
    def __init__(self, nb_sites, local_dim, init_sample=None,
                 nb_chains_per_device=10, nb_samples=1000, chain_length=10,
                 rand_seed=1234):
        """
        nb_sites: number of qudits
        init_sample: starting samples, if not provide will default to all zeros
        nb_chains: number of chains to parallely sample per device
        nb_samples: number of samples in total (combined for all devices)
        chain_length: length of chain to get each sample
        rand_seed: starting random seed
        Note: to reproduce the same result, keeping nb_chains * nb_devices constant is sufficient
        """
        super().__init__()
        self.pure_funcs = None # needs to set using set_var_state before actually using this
        self.var_state = None
        self.nb_sites = nb_sites
        self.local_dim = local_dim
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        if init_sample is not None:
            self.init_sample = init_sample
        else:
            self.init_sample = jnp.zeros(nb_sites, dtype=int)
        self.nb_chains_per_device = nb_chains_per_device
        self.nb_samples = nb_samples
        assert self.nb_samples % (self.nb_chains_per_device * self.nb_devices) == 0, f'{nb_samples=} must be divisible by {nb_chains_per_device=} * {jax.local_devices_count()=}'
        self.nb_sample_steps = nb_samples // (self.nb_chains_per_device * self.nb_devices)
        self.chain_length = chain_length
        # keep track of last sample of each chain
        self.curr_samples = jax.device_put_sharded([jnp.array([self.init_sample] * self.nb_chains_per_device)] * self.nb_devices, self.devices)
        self.curr_log_psi = None
        self.accept_mask = None # keep track of the last accept mask
        self.accept_rate = None # keep track of the last accept rate
        rand_key = jax.random.PRNGKey(rand_seed)
        self.rand_keys = jrnd.split(rand_key, num=self.nb_chains_per_device * self.nb_devices).reshape(self.nb_devices, self.nb_chains_per_device, 2) # each key is dim 2

    def set_var_state(self, var_state):
        """
        need to call this function before the sampler can be used
        """
        self.var_state = var_state
        # all the pure helper functions are collected in MCMCSamplerPure, which is immutable.
        self.pure_funcs = QuditMCMCSamplerPure(var_state.pure_funcs, self.nb_sites, self.local_dim)
        self.curr_log_psi = var_state.log_psi(self.curr_samples)

    def sample(self):
        """
        samples from the wave function given by self.var_state
        return: samples, log_psi corresponding to the samples and an array of ones as the weights of the samples
        """
        samples, log_psi, \
        curr_samples, curr_log_psi, \
        accept_mask, rand_keys = self.pure_funcs.sample_steps(self.nb_sample_steps, self.chain_length,
                                                              self.curr_samples, self.curr_log_psi,
                                                              self.var_state.get_state(), self.rand_keys)
        # samples have shape (nb_devices, nb_sample_steps, nb_chains_per_device, nb_sites)
        # log_psi have shape (
        self.curr_samples = curr_samples
        self.curr_log_psi = curr_log_psi
        self.rand_keys = rand_keys
        self.accept_mask = accept_mask
        self.accept_rate = accept_mask.mean()
        return samples, log_psi, jnp.ones_like(log_psi, dtype=float) / jnp.sqrt(log_psi.size) # sqrt_weights

    def thermalize(self, chain_length):
        _, _, \
        curr_samples, curr_log_psi, \
        accept_mask, rand_keys = self.pure_funcs.sample_steps(1, chain_length,
                                                              self.curr_samples, self.curr_log_psi,
                                                              self.var_state.get_state(), self.rand_keys)
        self.curr_samples = curr_samples
        self.curr_log_psi = curr_log_psi
        self.rand_keys = rand_keys
        self.accept_mask = accept_mask
        self.accept_rate = accept_mask.mean()
        return curr_samples, curr_log_psi


    def get_state(self):
        return {'curr_samples': self.curr_samples,
                'curr_log_psi': self.curr_log_psi,
                'accept_mask': self.accept_mask,
                'accept_rate':self.accept_rate,
                'rand_keys':self.rand_keys}

    def set_state(self, state):
        self.curr_samples = state['curr_samples']
        self.curr_log_psi = state['curr_log_psi']
        self.accept_mask = state['accept_mask']
        self.accept_rate = state['accept_rate']
        self.rand_keys = state['rand_keys']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))
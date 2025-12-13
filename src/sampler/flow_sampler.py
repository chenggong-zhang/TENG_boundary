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

@dataclass(frozen=True) # to make it immutable and hashable
class SimpleFlowSamplerPure:
    var_state_pure: Any
    system_shape: Shape

    @partial(jax.pmap, in_axes=(None, None, 0), out_axes=(0, 0, 0), static_broadcasted_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, None, 0), out_axes=(0, 0, 0))
    def sample(self, state, rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        sample, log_psi = self.var_state_pure.net.apply(state, (1, *self.system_shape), sub_rand_key, method=self.var_state_pure.net.__class__.sample)
        return sample.squeeze(0), log_psi.squeeze(0), rand_key

class SimpleFlowSampler(AbstractSampler):
    def __init__(self, system_shape, nb_samples, rand_seed=1234):
        super().__init__()
        self.pure_funcs = None
        self.var_state = None
        self.system_shape = system_shape
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        self.nb_samples = nb_samples
        self.nb_samples_per_device = nb_samples // self.nb_devices
        assert self.nb_samples % self.nb_devices == 0, f"{self.nb_samples=} must be a multiple of {self.nb_devices=}"
        rand_key = jax.random.PRNGKey(rand_seed)
        self.rand_keys = jrnd.split(rand_key, num=nb_samples).reshape(self.nb_devices, self.nb_samples_per_device, 2) # each key is dim 2

    def set_var_state(self, var_state):
        assert self.var_state is None, "var_state is already set, please create a new sampler"
        self.var_state = var_state
        self.pure_funcs = SimpleFlowSamplerPure(var_state.pure_funcs, self.system_shape)

    def sample(self):
        samples, log_psi, rand_keys = self.pure_funcs.sample(self.var_state.get_state(), self.rand_keys)
        self.rand_keys = rand_keys
        return samples, log_psi, jnp.ones_like(log_psi, dtype=float) / jnp.sqrt(log_psi.size)

    def get_state(self):
        return {'rand_keys': self.rand_keys}

    def set_state(self, state):
        self.rand_keys = state['rand_keys']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))
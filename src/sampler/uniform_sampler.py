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
class ContinuousUniformSamplerPure:
    nb_sites: int

    @partial(jax.pmap, in_axes=(None, None, None, 0), out_axes=(0, 0), static_broadcasted_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=(0, 0))
    def sample(self, minvals, maxvals, rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        sample = jrnd.uniform(sub_rand_key, (self.nb_sites,), minval=minvals, maxval=maxvals)
        return sample, rand_key

class ContinuousUniformSampler(AbstractSampler):
    def __init__(self, nb_sites, nb_samples, minvals, maxvals, rand_seed=1234):
        super().__init__()
        self.nb_sites = nb_sites
        if isinstance(minvals, (int, float)):
            minvals = [minvals] * nb_sites
        else:
            assert len(minvals) == nb_sites, f"{len(minvals)=} must be equal to {nb_sites=} or a single value"
        if isinstance(maxvals, (int, float)):
            maxvals = [maxvals] * nb_sites
        else:
            assert len(maxvals) == nb_sites, f"{len(maxvals)=} must be equal to {nb_sites=} or a single value"
        self.minvals = jnp.array(minvals)
        self.maxvals = jnp.array(maxvals)
        self.area = jnp.prod(self.maxvals - self.minvals)
        self.pure_funcs = ContinuousUniformSamplerPure(self.nb_sites)
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        self.nb_samples = nb_samples
        self.nb_samples_per_device = nb_samples // self.nb_devices
        assert self.nb_samples % self.nb_devices == 0, f"{self.nb_samples=} must be a multiple of {self.nb_devices=}"
        rand_key = jax.random.PRNGKey(rand_seed)
        self.rand_keys = jrnd.split(rand_key, num=nb_samples).reshape(self.nb_devices, self.nb_samples_per_device, 2) # each key is dim 2

    def set_var_state(self, var_state):
        # we don't need this function in this case
        pass

    def sample(self):
        samples, rand_keys = self.pure_funcs.sample(self.minvals, self.maxvals, self.rand_keys)
        self.rand_keys = rand_keys
        return samples, None, jnp.ones_like(rand_keys[:, :, 0], dtype=float) * jnp.sqrt(self.area / rand_keys[:, :, 0].size)

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
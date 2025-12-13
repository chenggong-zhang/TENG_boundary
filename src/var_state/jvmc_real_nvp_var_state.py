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
from abc import abstractmethod
from dataclasses import dataclass
import pickle
from functools import partial
from itertools import accumulate
from math import prod
import jax
import jax.numpy as jnp
from jax import tree_util
import flax
from flax.core import frozen_dict
from src.var_state import AbstractVarState, AbstractVarStatePure
from src.var_state.simple_var_state_real import SimpleVarStateRealPure, SimpleVarStateReal
from src.sampler import AbstractSampler

NN = Any
Sampler = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PyTreeDef = Any

assert False, "This file is not ready yet"
class JVMCRealNVPVarState(AbstractVarState):
    """
    real valued simple var_state
    parameters must also be real valued
    """

    def __init__(self, nb_dims, init_seed=1234, init_sample=None):
        """
        net: an neural network for the quantum state. should support input of shape (..., nb_sites) and output a single (complex) number
        batch_size: the batch_size used for sampling
        sampler: a sampler that samples from (normalized) |psi|^2
        init_seed: seed to initialize the neural network
        init_sample: samples used to initialize the network
        """

        init_seed, new_seed = jax.random.split(jax.random.PRNGKey(init_seed))


        super().__init__()

        # everything here except self.batch_size, self.sampler and self.state should remain the same after the instance is created, otherwise it will lead to a bug
        # consider creating a new instance to change anything

        self.sampler = sampler
        assert isinstance(self.sampler, AbstractSampler)

        # initialize the network
        if init_sample is None:
            init_sample = jnp.empty((1, *system_shape))
        self.state = net.init(jax.random.PRNGKey(init_seed), init_sample) # neural network state
        flattened_params, param_treedef = tree_util.tree_flatten(self.state['params'])
        param_shapes = tuple(param.shape for param in flattened_params)
        param_lens = tuple(prod(param_shape) for param_shape in param_shapes)
        param_cuts = tuple(accumulate(param_lens)) # the cuts to separate the parameters, cumulative sum of self.param_lens
        self.param_is_complex = jnp.iscomplexobj(flattened_params[0]) # this may be improved to keep track of each parameter. Will keep as is for now
        assert not self.param_is_complex, "parameters must be real valued"

        # use the network and all the frozen state to create pure helper functions with var_statePure (immutable)
        self.pure_funcs = SimpleVarStateRealPure(
            net=net, system_shape=system_shape,
            param_treedef=param_treedef, param_shapes=param_shapes, param_lens=param_lens, param_cuts=param_cuts)

        self.sampler.set_var_state(self)  # pass self to sampler so sampler knows how to sample

    # we can access them, but we cannot set them
    @property
    def system_shape(self):
        return self.pure_funcs.system_shape

    @property
    def nb_sites(self):
        return prod(self.system_shape)

    @property
    def net(self):
        return self.pure_funcs.net

    def update_parameters(self, d_params):
        """
        update the trainable parameters
        """
        if isinstance(d_params, jnp.ndarray):
            # flattend
            d_params = self.pure_funcs.unflatten_parameters(d_params)
        new_params = tree_util.tree_map(jnp.add, self.state['params'], d_params)
        self.state = flax.core.copy(self.state, add_or_replace={'params': new_params})

    # already implemented in abstract var_state, copied here for clarity
    # def log_psi(self, samples):
    #     """
    #     evaluates the log_psi at the provided samples. Can be unnormalized
    #     """
    #     return self.pure_funcs.log_psi_pmapped(self.state, samples)
    #
    # def jac_log_psi(self, samples):
    #     """
    #     returns the jacobian matrix regarding the current samples
    #     return a complex valued matrix or array,
    #     with real/imag part denoting the jacobian of real/imag part of log psi to flattened params
    #     """
    #
    #     return self.pure_funcs.jac_log_psi_pmapped(self.state, samples)
    #
    #
    # def jvp_vjp_log_psi_func(self, samples, return_value=True, jit=True):
    #     jvp, vjp, value = self.pure_funcs.jvp_vjp_log_psi_func(self.state, samples)
    #     if jit:
    #         jvp = jax.jit(jvp)
    #         vjp = jax.jit(vjp)
    #     if return_value:
    #         return jvp, vjp, value
    #     return jvp, vjp

    def sample(self):
        """
        sample from normalized |psi|^2
        """
        return self.sampler() # same as self.sampler.sample()

    def get_parameters(self, flatten=False):
        """
        get the trainable parameters
        Note we are keeping complex parameters as is
        """
        params = self.state['params']
        if not flatten:
            return params
        else:
            return self.pure_funcs.flatten_parameters(params)

    def set_parameters(self, new_params):
        """
        set the trainable parameters
        Note we are keeping complex parameters as is
        """
        if isinstance(new_params, jnp.ndarray):
            # flattend
            new_params = self.pure_funcs.unflatten_parameters(new_params)
        self.state = flax.core.copy(self.state, add_or_replace={'params': new_params})

    def count_parameters(self):
        """
        count trainable parameters (each complex parameter counted as two parameters)
        """
        return len(self.get_parameters(flatten=True))

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = frozen_dict.freeze(state)

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(frozen_dict.unfreeze(self.get_state()), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))




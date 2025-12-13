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
from src.sampler import AbstractSampler

NN = Any
Sampler = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PyTreeDef = Any

@dataclass(frozen=True) # to make it immutable and hashable
class SimpleVarStatePureBase(AbstractVarStatePure):
    net: NN
    nb_sites: int
    param_treedef: PyTreeDef
    param_shapes: Tuple[Shape, ...]
    param_lens: Tuple[int, ...]
    param_cuts: Tuple[int, ...]

    def flatten_parameters_naive(self, params):
        """
        naive way of flatten parameters without distinguishing real and complex values
        very likely you want to use flatten_parameters instead of this method
        use with caution
        """
        params = tree_util.tree_flatten(params)[0]
        params = [param.ravel() for param in params]
        params = jnp.concatenate(params)
        return params

    @abstractmethod
    def flatten_parameters(self, params):
        """
        should be subclassed to distinguish real and complex parameters
        """
        pass

    @partial(jax.jit, static_argnums=0)
    def flatten_parameters_jitted(self, params):
        return self.flatten_parameters(params)

    def unflatten_parameters_naive(self, params):
        """
        naive way of unflatten parameters without distinguishing real and complex values
        very likely you want to use unflatten_parameters instead of this method
        use with caution
        """
        params = jnp.split(params, self.param_cuts)
        params = [param.reshape(param_shape) for param, param_shape in zip(params, self.param_shapes)]
        params = tree_util.tree_unflatten(self.param_treedef, params)
        return params

    @abstractmethod
    def unflatten_parameters(self, params):
        """
        should be subclassed to distinguish real and complex parameters
        """
        pass

    @partial(jax.jit, static_argnums=0)
    def unflatten_parameters_jitted(self, params):
        return self.unflatten_parameters(params)

    def evaluate(self, state, samples):
        return self.net.apply(state, samples)

    # # already implemented in abstract var_state. reimplemented here for clarity
    # @partial(jax.jit, static_argnums=0)
    # def log_psi_jitted(self, state, samples):
    #     return self.net.apply(state, samples)
    #
    # # already implemented in abstract var_state. reimplemented here for clarity
    # @partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=0)
    # def log_psi_pmapped(self, state, samples):
    #     return self.net.apply(state, samples)

    def jvp_vjp_func(self, state, samples):
        """
        samples will be applied in parallel to multiple devices
        we will not jit inside this function as it can be jitted later one when needed
        """

        curr_params = self.flatten_parameters(state['params'])

        # alternatively don't use the pmapped apply here, but manually put the data to different devices and copy state to different devices
        def net_apply_func(params):
            params = self.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = self.evaluate_pmapped(new_state, samples)
            return value.real, value.imag

        def jvp_func(tangents):
            pushforwards = jax.jvp(net_apply_func, (curr_params,), (tangents,))[1]
            return pushforwards[0] + 1j * pushforwards[1]

        value, vjp_func_raw = jax.vjp(net_apply_func, curr_params)

        def vjp_func(cotangents):
            cotangents = (cotangents.real, cotangents.imag)
            return vjp_func_raw(cotangents)[0]

        return jvp_func, vjp_func, value

    def jacobian(self, state, samples):
        """
        returns the jacobian matrix regarding the current samples
        return a complex valued matrix or array,
        with real/imag part denoting the jacobian of real/imag part of log psi to flattened params
        """
        # commented out jit because very likely this function is used only once per compile
        # @jax.jit
        def net_apply_func(params):
            params = self.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = self.evaluate(new_state, samples)
            return (value.real, value.imag)
        jac = jax.jacrev(net_apply_func)(self.flatten_parameters(state['params']))
        return jac[0] + 1j * jac[1]

    # already implemented in abstract var_state. copied here for clarity
    # @partial(jax.jit, static_argnums=0)
    # def jac_log_psi_jitted(self, state, samples):
    #     return self.jac_log_psi(state, samples)

    # already implemented in abstract var_state. copied here for clarity
    # @partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=0)
    # def jac_log_psi_pmapped(self, state, samples):
    #     return self.jac_log_psi(state, samples)


@dataclass(frozen=True) # to make it immutable and hashable
class SimpleVarStatePureRealParam(SimpleVarStatePureBase):

    def flatten_parameters(self, params):
        """
        convert a param in pytree format into a single vector
        assuming parameters are real
        """
        return self.flatten_parameters_naive(params)

    def unflatten_parameters(self, params):
        return self.unflatten_parameters_naive(params)


@dataclass(frozen=True) # to make it immutable and hashable
class SimpleVarStatePureComplexParam(SimpleVarStatePureBase):

    def flatten_parameters(self, params):
        params = self.flatten_parameters_naive(params)
        # separate real and imaginary part
        return jnp.concatenate([params.real, params.imag])

    def unflatten_parameters(self, params):
        # combine real and imaginary part first
        params = params.reshape(2, -1)
        params = params[0] + params[1] * 1j
        return self.unflatten_parameters_naive(params)


class SimpleVarState(AbstractVarState):

    def __init__(self, net, nb_sites, sampler, init_seed=1234, init_sample=None):
        """
        net: an neural network for the quantum state. should support input of shape (..., nb_sites) and output a single (complex) number
        batch_size: the batch_size used for sampling
        sampler: a sampler that samples from (normalized) |psi|^2
        init_seed: seed to initialize the neural network
        init_sample: samples used to initialize the network
        """
        super().__init__()

        # everything here except self.batch_size, self.sampler and self.state should remain the same after the instance is created, otherwise it will lead to a bug
        # consider creating a new instance to change anything

        self.sampler = sampler
        assert isinstance(self.sampler, AbstractSampler)

        # initialize the network
        if init_sample is None:
            init_sample = jnp.empty((1, nb_sites))
        self.state = net.init(jax.random.PRNGKey(init_seed), init_sample) # neural network state
        flattened_params, param_treedef = tree_util.tree_flatten(self.state['params'])
        param_shapes = tuple(param.shape for param in flattened_params)
        param_lens = tuple(prod(param_shape) for param_shape in param_shapes)
        param_cuts = tuple(accumulate(param_lens)) # the cuts to separate the parameters, cumulative sum of self.param_lens
        self.param_is_complex = jnp.iscomplexobj(flattened_params[0]) # this may be improved to keep track of each parameter. Will keep as is for now

        # use the network and all the frozen state to create pure helper functions with var_statePure (immutable)
        if self.param_is_complex:
            self.pure_funcs = SimpleVarStatePureComplexParam(
                net=net, nb_sites=nb_sites,
                param_treedef=param_treedef, param_shapes=param_shapes, param_lens=param_lens, param_cuts=param_cuts)
        else:
            self.pure_funcs = SimpleVarStatePureRealParam(
                net=net, nb_sites=nb_sites,
                param_treedef=param_treedef, param_shapes=param_shapes, param_lens=param_lens, param_cuts=param_cuts)

        self.sampler.set_var_state(self)  # pass self to sampler so sampler knows how to sample

    # we can access them, but we cannot set them
    @property
    def nb_sites(self):
        return self.pure_funcs.nb_sites

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
        self.state = state

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(frozen_dict.unfreeze(self.get_state()), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))




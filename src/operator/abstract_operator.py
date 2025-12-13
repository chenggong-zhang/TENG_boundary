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
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import numpy as np
import scipy as sp
from scipy import sparse
import jax
from jax import lax
import jax.numpy as jnp
# from jax.experimental import sparse as jexp_sparse

Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Union[jnp.ndarray, np.ndarray]
PyTreeDef = Any

class AbstractOperator(ABC):

    frozen = False

    def __init__(self):
        """
        init_function freezes the class, so when called using super from subclass, it should be called last.
        """
        self.frozen = True

    def __setattr__(self, name, value):
        """
        try to make this class frozon
        """
        if self.frozen and hasattr(self, name):
            raise AttributeError(f"Cannot modify attribute {name} of a frozen instance")
        super().__setattr__(name, value)

    def __call__(self, var_state: Any, samples: Array, log_psi: Array=None, compile: bool=True) -> Array:
        """
        convinient wrapper of local_energy
        compile: whether to use the compiled version or not
        """
        return self.local_energy(var_state, samples, log_psi, compile)


    @abstractmethod
    def reverse_flip_func(self, samples: Any) -> Tuple[Array, Tuple[int, Callable[[int], Tuple[Any, Array]]]]:
        """
        reverse_flip function for operator
        given |x>, computes all |y> and <x|H|y> such that <x|H|y> is non zero
        returns:(diagonal_weights, (nb_flips, flip_func(i)->(flipped_samples_all[i], flipped_weights_all[i])))
                equivalently(<x|H|x>, (nb_flips, flip_func(i)->(|y>, <x|H|y>)))
        Note: by providing diagonal weights, the subsequent computation can be faster (no need to evaluate <x|psi>),
              but this is not requires, in which case, diagonal weights can be returned as a zero array of complex dtype
        Note: this function should be implemented either as a pure function, or the state it depends on cannot be modified
              otherwise the compiled version may not work as expected.
        """
        pass

    def reverse_flip_generator(self, samples: Any) -> Tuple[Array, Iterable[Tuple[Any, Array]]]:
        """
        a wrapper of reverse_flip_func where the second out put becomes an iterable of flipped_samples and flipped_weights
        returns:(diagonal_weights, zip(flipped_samples_all, flipped weights_all))
                equivalently(<x|H|x>, zip(|y>, <x|H|y>))
        Note: it does not necessarity us the zip function. just to denote what the result looks like
        """
        diag_weights, (nb_flips, flip_func) = self.reverse_flip_func(samples)
        return diag_weights, (flip_func(i) for i in range(nb_flips))

    def local_energy(self, var_state: Any, samples: Array, log_psi: Array=None, compile: bool=True) -> Array:
        if compile:
            # assert False, "don't use the compiled version. it has a bug now, probably due to the use of for loop instead lax scan"
            return self.local_energy_compiled(var_state, samples, log_psi)
        else:
            return self.local_energy_noncompiled(var_state, samples, log_psi)

    # def local_energy_noncompiled(self, var_state: Any, samples: Any, log_psi: Array=None) -> Array:
    #     """
    #     simple implementation to compute the local energy with for loop.
    #     if self.reverse_flip outputs a generator, this method can be very memory efficient
    #     however, it may not be the most time efficient
    #     Note: we may name this function as local_observable, but local observable has distinct means, so it can be confusing
    #     Note: although this function is not explicitly pmapped, it should still execute in parallel
    #     """
    #     if log_psi is None:
    #         log_psi = var_state.log_psi(samples)
    #     diag_weights, flipped = self.reverse_flip(samples)
    #     E_loc = diag_weights
    #     log_psi_ratios = []
    #     for flipped_samples, flipped_weights in flipped:
    #         log_psi_flipped = var_state.log_psi(flipped_samples)
    #         log_psi_ratio = log_psi_flipped - log_psi
    #         log_psi_ratios.append(log_psi_ratio)
    #         E_loc += flipped_weights * jnp.exp(log_psi_ratio)
    #     return E_loc

    # @partial(jax.pmap, in_axes=(None, None, 0, 0, None), static_broadcasted_argnums=(0, 1))
    # def local_energy_compiled_pure(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
    #     """
    #     a pure function version of computing the local energy.
    #     will be pmapped and compiled.
    #     however, compilation unrolls the for loop. unclear how much gain using this function
    #     """
    #     diag_weights, flipped = self.reverse_flip(samples)
    #     E_loc = diag_weights
    #     for flipped_samples, flipped_weights in flipped:
    #         log_psi_flipped = var_state_pure.log_psi(state, flipped_samples)
    #         log_psi_ratio = log_psi_flipped - log_psi
    #         E_loc += flipped_weights * jnp.exp(log_psi_ratio)
    #     return E_loc

    # @partial(jax.jit, static_argnums=(0, 1))
    # def local_energy_compiled_pure(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
    #     """
    #     a pure function version of computing the local energy.
    #     will be pmapped and compiled.
    #     however, compilation unrolls the for loop. unclear how much gain using this function
    #     """
    #     diag_weights, flipped = self.reverse_flip(samples)
    #     E_loc = diag_weights
    #     log_psi_ratios = []
    #     for flipped_samples, flipped_weights in zip(*flipped):
    #         log_psi_flipped = var_state_pure.log_psi_pmapped(state, flipped_samples)
    #         log_psi_ratio = log_psi_flipped - log_psi
    #         log_psi_ratios.append(log_psi_ratio)
    #         E_loc = E_loc + flipped_weights * jnp.exp(log_psi_ratio)
    #     return E_loc# , diag_weights, tuple(flipped), tuple(log_psi_ratios)

    # @partial(jax.jit, static_argnums=(0, 1))
    # def local_energy_compiled_pure(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
    #     """
    #     a pure function version of computing the local energy.
    #     will be pmapped and compiled.
    #     however, compilation unrolls the for loop. unclear how much gain using this function
    #     """
    #     diag_weights, flipped = self.reverse_flip_func(samples)
    #     E_loc = diag_weights
    #     def scan_body_func(carry, x):
    #         e_loc = carry
    #         flipped_samples, flipped_weights = x
    #         log_psi_flipped = var_state_pure.log_psi_pmapped(state, flipped_samples)
    #         log_psi_ratio = log_psi_flipped - log_psi
    #         return e_loc + flipped_weights * jnp.exp(log_psi_ratio), None
    #     E_loc, _ = lax.scan(scan_body_func, init=E_loc, xs=flipped)
    #     return E_loc  # , diag_weights, tuple(flipped), tuple(log_psi_ratios)

    @partial(jax.pmap, in_axes=(None, None, 0, 0, None), static_broadcasted_argnums=(0, 1))
    def local_energy_pure_pmapped(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
        """
        a pure function version of computing the local energy.
        will be pmapped and compiled.
        however, compilation unrolls the for loop. unclear how much gain using this function
        """
        diag_weights, (nb_flips, flip_func) = self.reverse_flip_func(samples)
        E_loc = diag_weights
        def fori_body_func(i, val):
            e_loc = val
            flipped_samples, flipped_weights = flip_func(i)
            log_psi_flipped = var_state_pure.log_psi(state, flipped_samples)
            log_psi_ratio = log_psi_flipped - log_psi
            return e_loc + flipped_weights * jnp.exp(log_psi_ratio)
        E_loc = lax.fori_loop(0, nb_flips, fori_body_func, init_val=E_loc)
        return jnp.clip(E_loc, a_min=-1e20, a_max=1e20)  # , diag_weights, tuple(flipped), tuple(log_psi_ratios)

    # @partial(jax.jit, static_argnums=(0, 1))
    # def local_energy_pure_jitted(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
    #     return self.local_energy_pure_unjitted(var_state_pure, samples, log_psi, state)

    def local_energy_compiled(self, var_state: Any, samples: Any, log_psi: Array=None) -> Array:
        """
        wrapper of self.local_energy_pure
        """
        if log_psi is None:
            log_psi = var_state.log_psi(samples)
        return self.local_energy_pure_pmapped(var_state.pure_funcs, samples, log_psi, var_state.get_state())

    def local_energy_noncompiled(self, var_state: Any, samples: Any, log_psi: Array=None) -> Array:
        """
        wrapper of self.local_energy_pure
        """
        if log_psi is None:
            log_psi = var_state.log_psi(samples)
        diag_weights, flipped = self.reverse_flip_generator(samples)
        E_loc = diag_weights

        # def fori_body_func(i, val):
        #     e_loc = val
        #     flipped_samples, flipped_weights = flip_func(i)
        #     log_psi_flipped = var_state.log_psi(flipped_samples)
        #     log_psi_ratio = log_psi_flipped - log_psi
        #     return e_loc + flipped_weights * jnp.exp(log_psi_ratio)
        #
        # E_loc = lax.fori_loop(0, nb_flips, fori_body_func, init_val=E_loc)

        for flipped_samples, flipped_weights in flipped:
            log_psi_flipped = var_state.log_psi(flipped_samples)
            log_psi_ratio = log_psi_flipped - log_psi
            E_loc = E_loc + flipped_weights * jnp.exp(log_psi_ratio)

        return jnp.clip(E_loc, a_max=1e20)  # , diag_weights, tuple(flipped), tuple(log_psi_ratios)
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
from .abstract_p_operator import AbstractPOperator

Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Union[jnp.ndarray, np.ndarray]
PyTreeDef = Any


class FokkerPlanckOperator(AbstractPOperator):

    def __init__(self, nb_dims: int, drift_coefs: Array, diffusion_coefs: Array, check_validity=True):
        """
        nb_dims: number of dimensions
        drift_coefs (mu_i): array of shape (nb_dims)
        diffusion_coefs (D_ij): array of shape (nb_dims, nb_dims)
        check_validity: whether to check the validity of the diffusion matrix
        computes the local operator O p (x) / p (x) with O = -\sum mu_i \partial_i + \sum D_ij \partial_i \partial_j
        """
        super().__init__()
        self.nb_dims = nb_dims
        self.drift_coefs = drift_coefs
        self.diffusion_coefs = diffusion_coefs
        if check_validity:
            assert np.allclose(self.diffusion_coefs, self.diffusion_coefs.T), "diffusion matrix must be symmetric"
            assert np.all(np.linalg.eigvals(self.diffusion_coefs) > -1e-7), "diffusion matrix must be positive definite"
            assert self.drift_coefs.shape == (self.nb_dims,), "drift coefs must be of shape (nb_dims,)"
            assert self.diffusion_coefs.shape == (self.nb_dims, self.nb_dims), "diffusion coefs must be of shape (nb_dims, nb_dims)"
        # not used for now
        ((self.nonzero_drift_dims, self.nonzero_drift_coefs),
         (self.nonzero_diffusion_dims, self.nonzero_diffusion_coefs)) = self.get_nonzero_coefs(self.drift_coefs,
                                                                                               self.diffusion_coefs)


    def get_nonzero_coefs(self, drift_coefs, diffusion_coefs):
        """
        returns the nonzero coefficients and dims of the drift vector and diffusion matrix
        """
        nonzero_drift_dims = np.where(drift_coefs != 0)[0]
        nonzero_diffusion_dims = np.where(diffusion_coefs != 0)
        nonzero_drift_coefs = drift_coefs[nonzero_drift_dims]
        nonzero_diffusion_coefs = diffusion_coefs[nonzero_diffusion_dims[0], nonzero_diffusion_dims[1]]
        return (nonzero_drift_dims, nonzero_drift_coefs), (nonzero_diffusion_dims, nonzero_diffusion_coefs) # MAYBE CONSIDER TAKING ADVANTAGE OF SYMMETRIC MATRIX

    def local_operator(self, var_state: Any, samples: Array, log_psi: Array=None, compile: bool=True) -> Array:
        """
        computes the local operator O p (x) / p (x) with O = -\sum mu_i \partial_i + \sum D_ij \partial_i \partial_j
        """
        if compile:
            return self.local_operator_compiled(var_state, samples, log_psi)
        else:
            assert False, "not implemented"


    @partial(jax.pmap, in_axes=(None, None, 0, 0, None), static_broadcasted_argnums=(0, 1))
    def local_operator_pure_pmapped(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
        """
        a pure function version of computing the local energy.
        will be pmapped and compiled.
        however, compilation unrolls the for loop. unclear how much gain using this function
        we are writing the (possibly) less efficient version for now
        Not that we are computing O p(x) / p(x)
        Here, grad p(x) / p(x) = grad log p(x) and H p(x) / p(x) = H log p(x) + grad log p(x) (grad log p(x))^T
        """
        log_prob_func = lambda state, samples: var_state_pure.log_psi(state, samples[None, ...]).squeeze(0).real * 2 # mimic batch dimension due to vmap and explicit taking real part here
        jac_func_log_prob = jax.jacrev(log_prob_func, argnums=1)
        hes_func_log_prob = jax.jacfwd(jac_func_log_prob, argnums=1)
        jac_func_log_prob = jax.vmap(jac_func_log_prob, in_axes=(None, 0), out_axes=0)
        hes_func_log_prob = jax.vmap(hes_func_log_prob, in_axes=(None, 0), out_axes=0)
        jac_log_prob = jac_func_log_prob(state, samples).reshape(samples.shape[0], samples.shape[-1]) # combine system dimensions
        hes_log_prob = hes_func_log_prob(state, samples).reshape(samples.shape[0], samples.shape[-1], samples.shape[-1])  # combine system dimensions
        jac_prob_over_prob = jac_log_prob
        hes_prob_over_prob = hes_log_prob + jac_prob_over_prob[..., None] * jac_prob_over_prob[..., None, :]
        drift_ratio = (jac_prob_over_prob * self.drift_coefs).sum(axis=-1)
        diffusion_ratio = (hes_prob_over_prob * self.diffusion_coefs).sum(axis=(-1, -2))
        return jnp.clip(-drift_ratio + diffusion_ratio, a_min=-1e20, a_max=1e20)  # follow the convention of sign

    # @partial(jax.jit, static_argnums=(0, 1))
    # def local_energy_pure_jitted(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
    #     return self.local_energy_pure_unjitted(var_state_pure, samples, log_psi, state)

    def local_operator_compiled(self, var_state: Any, samples: Any, log_psi: Array = None) -> Array:
        """
        wrapper of self.local_energy_pure
        """
        # if log_psi is None:
        #     log_psi = var_state.log_psi(samples) # we don't actually need this for now
        return self.local_operator_pure_pmapped(var_state.pure_funcs, samples, log_psi, var_state.get_state())
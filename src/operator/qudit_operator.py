from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union, Generator,
)
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import numpy as np
import scipy as sp
from jax import Array
from scipy import sparse
import jax
from jax import lax
import jax.numpy as jnp
# from jax.experimental import sparse as jexp_sparse
from src.operator import AbstractOperator

Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Union[jnp.ndarray, np.ndarray]
PyTreeDef = Any


class QuditOperator(AbstractOperator):

    def __init__(self, nb_sites, local_dim):
        self.nb_sites = nb_sites
        self.local_dim = local_dim
        # call super last because it can freeze the class
        super().__init__()

    # copied over for clarity, this should be defined in subclass
    @abstractmethod
    def reverse_flip_func(self, samples: Array) -> Tuple[Array, Iterable[Tuple[Array, Array]]]:
        """
        reverse_flip function for operator
        given |x>, computes all |y> and <x|H|y> such that <x|H|y> is non zero
        returns:(diagonal weights, zip(flipped samples, flipped weights))
                equivalently(<x|H|x>, zip(|y>, <x|H|y>))
        Note: by providing diagonal weights, the subsequent computation can be faster (no need to evaluate <x|psi>),
              but this is not requires, in which case, diagonal weights can be returned as 0
        Note: this function should be implemented either as a pure function, or the state it depends on cannot be modified
              otherwise the compiled version may not work as expected.
        """
        pass

    # following scipy convention, no underscore
    def tosparse(self):
        """
        returns a scipy sparse matrix for the operator
        """
        inds = self.gen_index()
        bases = self.gen_basis()
        diag_weights, flipped = self.reverse_flip_generator(bases)
        if isinstance(diag_weights, (int, float, complex)):
            diag_weights = jnp.ones_like(inds, dtype=type(diag_weights)) * diag_weights
        ham = sparse.coo_matrix((diag_weights, (inds, inds)))
        for flipped_bases, flipped_weights in flipped:
            flipped_inds = self.gen_index(flipped_bases)
            if isinstance(flipped_weights, (int, float, complex)):
                flipped_weights = jnp.ones_like(inds, dtype=type(flipped_weights)) * flipped_weights
            ham += sparse.coo_matrix((flipped_weights, (inds, flipped_inds)))
        return ham

    # following scipy convention, no underscore
    def todense(self, jarr=False):
        """
        jarr: whether to use jax array.
        returns a dense array for the operator
        """
        ham = self.tosparse().todense()
        if jarr:
            ham = jax.array(ham)
        return ham

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


class OneBodyOperator:

    def __init__(self, dense: np.ndarray, name=None):
        self.name = name
        self.dense = dense.astype(complex)
        # self.is_diag = (np.diag(np.diag(dense)) == dense).all()
        self.values, self.shifts = self.vs_repr()
        self.is_diag = (self.shifts == 0).all()

    def __repr__(self):
        return f'{super().__repr__()} with name={self.name}, value={self.dense}, and is_diag={self.is_diag}'

    def __str__(self):
        return f'{super().__str__()} with name={self.name}, value={self.dense}, and is_diag={self.is_diag}'

    def __neg__(self):
        return OneBodyOperator(-self.dense)

    def __add__(self, other):
        if isinstance(other, (int, float, complex, np.ndarray)):
            return OneBodyOperator(self.dense + other)
        elif isinstance(other, OneBodyOperator):
            return OneBodyOperator(self.dense + other.dense)
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'coo_matrix' and '{type(other)}'")

    def __sub__(self, other):
        if isinstance(other, (int, float, complex, np.ndarray)):
            return OneBodyOperator(self.dense - other)
        elif isinstance(other, OneBodyOperator):
            return OneBodyOperator(self.dense - other.dense)
        else:
            raise TypeError(f"unsupported operand type(s) for -: 'coo_matrix' and '{type(other)}'")

    def __mul__(self, other):
        if isinstance(other, (int, float, complex, np.ndarray)):
            return OneBodyOperator(self.dense * other)
        elif isinstance(other, OneBodyOperator):
            return OneBodyOperator(self.dense * other.dense)
        else:
            raise TypeError(f"unsupported operand type(s) for *: 'coo_matrix' and '{type(other)}'")

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex, np.ndarray)):
            return OneBodyOperator(self.dense / other)
        elif isinstance(other, OneBodyOperator):
            return OneBodyOperator(self.dense / other.dense)
        else:
            raise TypeError(f"unsupported operand type(s) for /: 'coo_matrix' and '{type(other)}'")

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            return OneBodyOperator(self.dense @ other)
        elif isinstance(other, OneBodyOperator):
            return OneBodyOperator(self.dense @ other.dense)
        else:
            raise TypeError(f"unsupported operand type(s) for @: 'coo_matrix' and '{type(other)}'")

    def tosparse(self):
        return sparse.coo_matrix(self.dense)

    def todense(self):
        return self.dense

    def vs_repr(self):
        """
        return the value-shift representation useful for calculating reverse_flip
        returns: value: the values of each shift
        shifts: the union of shifts (ind1 - ind0) for each row
        will be zero padded for rows with fewer shifts
        """
        non_zero_elements = np.argwhere(self.dense != 0)
        non_zero_values = self.dense[self.dense != 0]


        shifts = (non_zero_elements[:, 1] - non_zero_elements[:, 0]) % self.dense.shape[1]
        unique_shifts, indices = np.unique(shifts, return_inverse=True)
        shifts = unique_shifts
        values = jnp.zeros((unique_shifts.shape[0], self.dense.shape[1]), dtype=complex)
        values = values.at[indices, non_zero_elements[:, 0]].set(non_zero_values)
        return values, shifts




### This class may require further test ###
### only tested to work with TFIM for now ###
class SpinHalfOperator(QuditOperator):
    class KnownOperators:
        sigma_x = OneBodyOperator(np.array([[0, 1],
                                            [1, 0]]), name='sigma_x')
        sigma_y = OneBodyOperator(np.array([[0, -1j],
                                            [1j, 0]]), name='sigma_y')
        sigma_z = OneBodyOperator(np.array([[1, 0],
                                            [0, -1]]), name='sigma_z')
        sigma_eye = OneBodyOperator(np.array([[1, 0],
                                              [0, 1]]), name='sigma_eye')

        X = sigx = sig_x = sigmax = sigma_x
        Y = sigy = sig_y = sigmay = sigma_y
        Z = sigz = sig_z = sigmaz = sigma_z
        I = id = identity = sigI = sig_I = sigmaI = sigma_I = sigeye = sig_eye = sigmaeye = sigma_eye
        Sx = S_x = sigma_x / 2
        Sy = S_y = sigma_x / 2
        Sz = S_z = sigma_x / 2

    def __init__(self, nb_sites, op_list, op_coef_list, op_sites_list):
        """
        generator an operator for spin 1/2 system
        op_list: a list of tuples of oprators by names or custom OneBodyOperator, eg: [("sigma_x", "sigma_z"), ("sigma_y", "sigma_z")]
        op_coef_list: a list of the coefficients for each tuple of opreators, eg: [2, 1]
        op_sites_list: a list of tuples containing the sites to apply the corresponding operators, eg: [(0, 1), (1, 2)]
        """
        # assert False, 'this does not work now'
        assert len(op_list) == len(op_coef_list) == len(op_sites_list)
        self.nb_sites = nb_sites # although will be defined in super(), we need it right now
        self.op_list = op_list
        self.op_coef_list = op_coef_list
        self.op_sites_list = op_sites_list
        # get operators from strings and put the in the form of list of tuples even if the tuple has only one item
        # in addition get if all the operators in the tuple is diagonal
        op_list, op_is_diag_list = self.get_op_list_and_op_is_diag_list(op_list)
        # get the operator sites in the form of list of tuples even if the tuple has only one item
        op_sites_list = self.get_op_sites_list(op_sites_list)
        diags, offdiags, diag_shapes, offdiag_shapes = self.sort_ops(op_list, op_is_diag_list, op_coef_list, op_sites_list)
        # try to make the immutable but note that the internal of OneBodyOperator is not immutable
        self.diags = tuple(diags)
        self.offdiags = tuple(offdiags)
        self.diag_shapes = tuple(diag_shapes)
        self.offdiag_shapes = tuple(offdiag_shapes)
        self.diag_length = len(self.diags)
        self.offdiag_length = len(self.offdiags)

        # # try to make the immutable but note that the internal of OneBodyOperator is not immutable
        # self.op_list = tuple(self.op_list)
        # self.op_is_diag_list = tuple(self.op_is_diag_list)
        # self.op_coef_list = tuple(self.op_coef_list)
        # self.op_sites_list = tuple(self.op_sites_list)

        super().__init__(nb_sites, local_dim=2)

    def get_op_list_and_op_is_diag_list(self, op_list):
        new_op_list = []
        new_op_is_diag_list = []  # whether all the sub operators are diagonal
        for op in op_list:
            is_diag = True
            new_op = []
            # each operator can contain multiple operators on multiple sites
            if isinstance(op, (tuple, list)):
                for each_op in op:
                    new_each_op = self.get_operator(each_op)
                    new_op.append(new_each_op)
                    # new_op.append((new_each_op.values, new_each_op.shifts)) # convert to tuple so jax knows how to handle it
                    is_diag = is_diag and new_each_op.is_diag
            else:
                new_each_op = self.get_operator(op)
                new_op.append(new_each_op)
                # new_op.append((new_each_op.values, new_each_op.shifts))
                is_diag = is_diag and new_each_op.is_diag
            new_op_list.append(tuple(new_op))
            new_op_is_diag_list.append(is_diag)
        return new_op_list, new_op_is_diag_list

    def get_operator(self, op):
        if isinstance(op, OneBodyOperator):
            return op
        elif isinstance(op, str):
            new_op = getattr(self.KnownOperators, op, None)
            if new_op is None:
                raise ValueError(f'unknown operator {op}')
            else:
                return new_op
        else:
            raise TypeError(f'unknown operator {op}')

    def get_op_sites_list(self, op_sites_list):
        new_op_sites_list = []
        for op_sites in op_sites_list:
            new_op_sites = []
            if isinstance(op_sites, (tuple, list)):
                for op_site in op_sites:
                    assert 0 <= op_site < self.nb_sites
                    new_op_sites.append(op_site)
            else:
                assert 0 <= op_sites < self.nb_sites
                new_op_sites.append(op_sites)
            new_op_sites_list.append(tuple(new_op_sites))
        return new_op_sites_list

    def sort_ops(self, op_list, op_is_diag_list, op_coef_list, op_sites_list):
        assert len(op_list) == len(op_is_diag_list) == len(op_coef_list) == len(op_sites_list)
        diag_list = []
        diag_shapes = []
        offdiag_list = []
        offdiag_shapes = []
        for op, op_is_diag, op_coef, op_sites in zip(op_list, op_is_diag_list, op_coef_list, op_sites_list):
            assert len(op) == len(op_sites)
            if op_is_diag:
                diag_list.append((op, op_coef, op_sites))
                diag_shapes.append(len(op_sites))
            else:
                offdiag_list.append((op, op_coef, op_sites))
                offdiag_shapes.append(len(op_sites))
        return diag_list, offdiag_list, diag_shapes, offdiag_shapes

    def __call__(self, var_state: Any, samples: Array, log_psi: Array=None, compile: bool=False) -> Array:
        """
        convinient wrapper of local_energy
        compile: whether to use the compiled version or not
        """
        return self.local_energy(var_state, samples, log_psi, compile)

    def local_energy(self, var_state: Any, samples: Array, log_psi: Array=None, compile: bool=False) -> Array:
        if compile:
            assert False, "not implemented, do not use the compiled version"
            # assert False, "don't use the compiled version. it has a bug now, probably due to the use of for loop instead lax scan"
            # return self.local_energy_compiled(var_state, samples, log_psi)
        else:
            return self.local_energy_noncompiled(var_state, samples, log_psi)

    def reverse_flip_generator(self, samples):
        diag_weights = 0.
        for ops, op_coef, op_sites in self.diags:
            temp_weights = op_coef
            for op, op_site in zip(ops, op_sites):
                temp_weights = temp_weights * op.values[0, samples[..., op_site]]  # 0 means shift = 0
            diag_weights = diag_weights + temp_weights

        def gen_flipped():
            # loop through all off diagonal operators
            for ops, op_coef, op_sites in self.offdiags:
                # for each operator, there are many sub operators, that generates exponentially many samples and weights
                # initialize samples to the input samples and weights to op_coef
                flipped_samples_list = [samples]
                flipped_weights_list = [op_coef]
                # for each sub operator, modify the current flipped samples and weights by considering the current sub operator
                for op, op_site in zip(ops, op_sites):
                    new_flipped_samples_list = []
                    new_flipped_weights_list = []
                    for flipped_samples, flipped_weights in zip(flipped_samples_list, flipped_weights_list):
                        for shift, value in zip(op.shifts, op.values):
                            new_flipped_samples_list.append(
                                flipped_samples.at[..., op_site].set(
                                    (flipped_samples[..., op_site] + shift) % self.local_dim
                                )
                            )
                            new_flipped_weights_list.append(
                                flipped_weights * op.values[shift, flipped_samples[..., op_site]]
                            )
                    flipped_samples_list = new_flipped_samples_list
                    flipped_weights_list = new_flipped_weights_list
                # now the flipped samples and flipped weights are properly constructed for this operator, yield them
                for flipped_samples, flipped_weights in zip(flipped_samples_list, flipped_weights_list):
                    yield flipped_samples, flipped_weights

        return diag_weights, gen_flipped()


    def reverse_flip_func(self, samples):
        assert False, "not implemented, do not use the compiled version"
    #     # assert False, 'still need to work on this part'
    #     # diag_weights = jnp.zeros(samples.shape[:-1], dtype=complex)
    #     #
    #     # def fori_inner_loop(j, val):
    #     #     i, temp_weights = val
    #     #     ops, _, op_sites = self.diags[i]
    #     #     op = ops[j]
    #     #     op_site = op_sites[j]
    #     #     # samples_at_site = samples[..., op_site]
    #     #     # dyn_indices = jnp.stack([jnp.zeros_like(samples_at_site), samples_at_site])
    #     #     return i, temp_weights * op.values[0, samples[..., op_site]]
    #     #
    #     # def fori_outer_loop(i, val):
    #     #     diag_weights = val
    #     #     ops, op_coef, op_sites = self.diags[i]
    #     #     _, temp_weights = lax.fori_loop(0, self.diag_shapes[i], fori_inner_loop, init_val=(i, op_coef))
    #     #     return diag_weights + temp_weights
    #     #
    #     # diag_weights = lax.fori_loop(0, self.diag_length, fori_outer_loop, init_val=diag_weights)
    #
    #     diag_weights = 0.
    #     for ops, op_coef, op_sites in self.diags:
    #         temp_weights = op_coef
    #         for op, op_site in zip(ops, op_sites):
    #             temp_weights = temp_weights * op.values[0, samples[..., op_site]]  # 0 means shift = 0
    #         diag_weights = diag_weights + temp_weights
    #
    #     def gen_flipped():
    #         # loop through all off diagonal operators
    #         for ops, op_coef, op_sites in self.offdiags:
    #             # for each operator, there are many sub operators, that generates exponentially many samples and weights
    #             # initialize samples to the input samples and weights to op_coef
    #             flipped_samples_list = [samples]
    #             flipped_weights_list = [op_coef]
    #             # for each sub operator, modify the current flipped samples and weights by considering the current sub operator
    #             for op, op_site in zip(ops, op_sites):
    #                 new_flipped_samples_list = []
    #                 new_flipped_weights_list = []
    #                 for flipped_samples, flipped_weights in zip(flipped_samples_list, flipped_weights_list):
    #                     for shift, value in zip(op.shifts, op.values):
    #                         new_flipped_samples_list.append(
    #                             flipped_samples.at[..., op_site].set(
    #                                 (flipped_samples[..., op_site] + shift) % self.local_dim
    #                             )
    #                         )
    #                         new_flipped_weights_list.append(
    #                             flipped_weights * op.values[shift, flipped_samples[..., op_site]]
    #                         )
    #                 flipped_samples_list = new_flipped_samples_list
    #                 flipped_weights_list = new_flipped_weights_list
    #             # now the flipped samples and flipped weights are properly constructed for this operator, yield them
    #             for flipped_samples, flipped_weights in zip(flipped_samples_list, flipped_weights_list):
    #                 yield flipped_samples, flipped_weights
    #
    #     flipped_all = tuple(gen_flipped())
    #
    #     nb_flips = len(flipped_all)
    #
    #     def flip_func(i):
    #         # return lax.gather(flipped_all, jnp.array([i]))[0]
    #         return flipped_all[i]
    #
    #     return diag_weights, (nb_flips, flip_func)



# class SpinHalfSumOfOneNTwoBodyOperator(QuditOperator):
#     class KnownOperators:
#         sigma_x = OneBodyOperator(np.array([[0, 1],
#                                             [1, 0]]), name='sigma_x')
#         sigma_y = OneBodyOperator(np.array([[0, -1j],
#                                             [1j, 0]]), name='sigma_y')
#         sigma_z = OneBodyOperator(np.array([[1, 0],
#                                             [0, -1]]), name='sigma_z')
#         sigma_eye = OneBodyOperator(np.array([[1, 0],
#                                               [0, 1]]), name='sigma_eye')
#
#         X = sigx = sig_x = sigmax = sigma_x
#         Y = sigy = sig_y = sigmay = sigma_y
#         Z = sigz = sig_z = sigmaz = sigma_z
#         I = id = identity = sigI = sig_I = sigmaI = sigma_I = sigeye = sig_eye = sigmaeye = sigma_eye
#         Sx = S_x = sigma_x / 2
#         Sy = S_y = sigma_x / 2
#         Sz = S_z = sigma_x / 2
#
#     def __init__(self, nb_sites, op_list, op_coef_list, op_sites_list):
#         assert len(op_list) == len(op_coef_list) == len(op_sites_list)
#         self.nb_sites = nb_sites  # although will be defined in super(), we need it right now
#         self.op_list = op_list
#         self.op_coef_list = op_coef_list
#         self.op_sites_list = op_sites_list
#
#         self.one_body_diags, self.one_body_offdiags, self.two_body_diags, self.two_body_off_diags = self.get_op_list(op_list)
#
#         super().__init__(nb_sites, local_dim=2)
#
#     def get_op_list(self, op_list):
#         new_one_body_diag_op_list = []
#         new_one_body_offdiag_op_list = []
#         new_two_body_diag_op_list = []
#         new_two_body_offdiag_op_list = []
#         for op in op_list:
#             is_diag = True
#             new_op = []
#             # each operator can contain multiple operators on multiple sites
#             if isinstance(op, (tuple, list)):
#                 for each_op in op:
#                     new_each_op = self.get_operator(each_op)
#                     new_op.append(new_each_op)
#                     # new_op.append((new_each_op.values, new_each_op.shifts)) # convert to tuple so jax knows how to handle it
#                     is_diag = is_diag and new_each_op.is_diag
#             else:
#                 new_each_op = self.get_operator(op)
#                 new_op.append(new_each_op)
#                 # new_op.append((new_each_op.values, new_each_op.shifts))
#                 is_diag = is_diag and new_each_op.is_diag
#             if len(new_op) == 1:
#                 if is_diag:
#                     new_one_body_diag_op_list.append(new_op[0])
#                 else:
#                     new_one_body_offdiag_op_list.append(new_op[0])
#             elif len(new_op) == 2:
#                 if is_diag:
#                     new_two_body_diag_op_list.append(tuple(new_op))
#                 else:
#                     new_two_body_offdiag_op_list.append(tuple(new_op))
#             else:
#                 assert False, f'only support one and two body operators, but get {len(new_op)=}'
#         return (tuple(new_one_body_diag_op_list),
#                 tuple(new_one_body_offdiag_op_list),
#                 tuple(new_two_body_diag_op_list),
#                 tuple(new_two_body_offdiag_op_list))
#
#     def get_operator(self, op):
#         if isinstance(op, OneBodyOperator):
#             return op
#         elif isinstance(op, str):
#             new_op = getattr(self.KnownOperators, op, None)
#             if new_op is None:
#                 raise ValueError(f'unknown operator {op}')
#             else:
#                 return new_op
#         else:
#             raise TypeError(f'unknown operator {op}')
#
#     def reverse_flip_func(self, samples: Array) -> Tuple[Array, Iterable[Tuple[Array, Array]]]:




class TFIMHamiltonian1D(QuditOperator):

    def __init__(self, nb_sites, J=1., h=1., periodic=True):
        """
        Hamiltonian of the form -J ZZ - h X
        J: ZZ couping term
        h: X coupling term
        periodic: whether to use the periodic boundary condition
        """
        self.J = J
        self.h = h
        self.periodic = periodic
        super().__init__(nb_sites, local_dim=2)

    # def reverse_flip(self, samples):
    #     # ZZ term compute the parity of two nearby qubits
    #     ZZ = (samples[..., 1:] == samples[..., :-1]) * 2 - 1
    #     diag_weights = - self.J * ZZ.sum(-1) # negative sign due to convention
    #     if self.periodic: # this should work even if compiled
    #         diag_weights += -self.J * ((samples[..., -1] == samples[..., 0]) * 2 - 1)# negative sign due to convention
    #     def gen_flipped():
    #         for i in range(self.nb_sites):
    #             flipped_samples = samples.at[..., i].set(1 - samples[..., i]) # checked! works with jit! need to check if this will replace the samples if jit compiled
    #             flipped_weights = - self.h # negative sign due to convention
    #             yield flipped_samples, flipped_weights
    #     return diag_weights, tuple(gen_flipped())

    # def reverse_flip(self, samples):
    #     # ZZ term compute the parity of two nearby qubits
    #     ZZ = (samples[..., 1:] == samples[..., :-1]) * 2 - 1
    #     diag_weights = (- self.J * ZZ.sum(-1)).astype(complex) # negative sign due to convention
    #     def cond_periodic():
    #         return diag_weights + -self.J * ((samples[..., -1] == samples[..., 0]) * 2 - 1)
    #     def cond_not_periodic():
    #         return diag_weights
    #     diag_weights = lax.cond(self.periodic, cond_periodic, cond_not_periodic)
    #     def scan_body_func(carry, x):
    #         i = carry
    #         flipped_samples = samples.at[..., i].set(1 - samples[..., i])
    #         flipped_weights = - self.h
    #         return i + 1, (flipped_samples, flipped_weights)
    #     _, flipped = lax.scan(scan_body_func, init=0, xs=None, length=self.nb_sites)
    #     return diag_weights, flipped

    def reverse_flip_func(self, samples):
        # ZZ term compute the parity of two nearby qubits
        ZZ = (samples[..., 1:] == samples[..., :-1]) * 2 - 1
        diag_weights = (- self.J * ZZ.sum(-1)).astype(complex) # negative sign due to convention
        def cond_periodic():
            return diag_weights + -self.J * ((samples[..., -1] == samples[..., 0]) * 2 - 1)
        def cond_not_periodic():
            return diag_weights
        diag_weights = lax.cond(self.periodic, cond_periodic, cond_not_periodic)
        nb_flips = self.nb_sites
        def flip_func(i):
            flipped_samples = samples.at[..., i].set(1 - samples[..., i])
            flipped_weights = - self.h
            return flipped_samples, flipped_weights
        return diag_weights, (nb_flips, flip_func)



class TFIMHamiltonian2D(QuditOperator):

    def __init__(self, nb_rows, nb_cols, J=1., h=1., periodic=True):
        """
        Hamiltonian of the form -J ZZ - h X
        by convention, the samples are treated to have the shape either (..., nb_sites) or (..., nb_rows, nb_cols)
        J: ZZ couping term
        h: X coupling term
        periodic: whether to use the periodic boundary condition
        """
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.J = J
        self.h = h
        self.periodic = periodic
        super().__init__(nb_rows * nb_cols, local_dim=2)

    def reverse_flip_func(self, samples):
        # return lax.cond(samples.shape[-1] == self.nb_sites, self.reverse_flip_1d, self.reverse_flip_2d, samples)
        # the if statements here should work because they only depend on shapes, not values
        # if it is the 1d representation
        if samples.shape[-1] == self.nb_sites:
            return self.reverse_flip_1d(samples)
        elif samples.shape[-2] == (self.nb_rows, self.nb_cols):
            return self.reverse_flip_2d(samples)
        else:
            raise ValueError(f'expect samples to have either shape (..., {self.nb_sites}) or (..., {self.nb_rows}, {self.nb_cols}), but get {samples.shape}')

    # def reverse_flip_1d(self, samples: Array):
    #     """
    #     reverse flip assuming the samples is one dimensional
    #     """
    #     diag_weights, flipped = self.reverse_flip_2d(samples.reshape(samples.shape[:-1] + (self.nb_rows, self.nb_cols)))
    #     # a wrapper to change the samples back to 1d representation
    #     def gen_flipped():
    #         for flipped_samples, flipped_weights in flipped:
    #             yield flipped_samples.reshape(samples.shape), flipped_weights
    #     return diag_weights, gen_flipped()

    # def reverse_flip_2d(self, samples: Array):
    #     # ZZ along the rows
    #     ZZ_r = (samples[..., 1:, :] == samples[..., :-1, :]) * 2 - 1
    #     # ZZ along columns
    #     ZZ_c = (samples[..., :, 1:] == samples[..., :, :-1]) * 2 - 1
    #     diag_weights = -self.J * (ZZ_r.sum((-2, -1)) + ZZ_c.sum((-2, -1)))
    #     if self.periodic:
    #         # row boundary
    #         ZZ_rb = (samples[..., -1, :] == samples[..., 0, :]) * 2 - 1
    #         # column boundary
    #         ZZ_cb = (samples[..., :, -1] == samples[..., :, 0]) * 2 - 1
    #         diag_weights += -self.J * (ZZ_rb.sum(-1) + ZZ_cb.sum(-1))
    #     def gen_flipped():
    #         for r in range(self.nb_rows):
    #             for c in range(self.nb_cols):
    #                 flipped_samples = samples.at[..., r, c].set(1 - samples[..., r, c]) # checked! works with jit! need to check if this will replace the samples if jit compiled
    #                 flipped_weights = - self.h # negative sign due to convention
    #                 yield flipped_samples, flipped_weights
    #     return diag_weights, gen_flipped()

    def reverse_flip_1d(self, samples: Array):
        """
        reverse flip assuming the samples is one dimensional
        """
        diag_weights, (nb_flips, flip_func_2d) = self.reverse_flip_2d(samples.reshape(samples.shape[:-1] + (self.nb_rows, self.nb_cols)))
        # a wrapper to change the samples back to 1d representation
        def flip_func(i):
            flipped_samples, flipped_weights = flip_func_2d(i)
            return flipped_samples.reshape(samples.shape), flipped_weights
        return diag_weights, (nb_flips, flip_func)

    def reverse_flip_2d(self, samples: Array):
        # ZZ along the rows
        ZZ_r = (samples[..., 1:, :] == samples[..., :-1, :]) * 2 - 1
        # ZZ along columns
        ZZ_c = (samples[..., :, 1:] == samples[..., :, :-1]) * 2 - 1
        diag_weights = -self.J * (ZZ_r.sum((-2, -1)) + ZZ_c.sum((-2, -1))).astype(complex)
        def cond_periodic():
            # row boundary
            ZZ_rb = (samples[..., -1, :] == samples[..., 0, :]) * 2 - 1
            # column boundary
            ZZ_cb = (samples[..., :, -1] == samples[..., :, 0]) * 2 - 1
            return diag_weights + -self.J * (ZZ_rb.sum(-1) + ZZ_cb.sum(-1))
        def cond_not_periodic():
            return diag_weights
        diag_weights = lax.cond(self.periodic, cond_periodic, cond_not_periodic)
        nb_flips = self.nb_sites
        def flip_func(i):
            r, c = divmod(i, self.nb_cols)
            flipped_samples = samples.at[..., r, c].set(1 - samples[..., r, c])
            flipped_weights = - self.h
            return flipped_samples, flipped_weights
        return diag_weights, (nb_flips, flip_func)


class TFIMHamiltonian(QuditOperator):
    def __init__(self, nb_site, zz_sites_list, J=1., h=1.):
        """
        Hamiltonian of the form -J ZZ - h X
        this operator works for any geometry provided the zz_sites are provided
        zz_sites: a list of tuples of the sites of the two z operators
        J: ZZ couping term
        h: X coupling term
        periodic: whether to use the periodic boundary condition
        """
        self.zz_sites_list = jnp.array(tuple(zz_sites_list)) # make it immutable
        self.J = J
        self.h = h
        super().__init__(nb_site, local_dim=2)

    # def reverse_flip(self, samples):
    #     # ZZ term compute the parity of two sites
    #     ZZ = (samples[..., self.zz_sites_list[:, 0]] == samples[..., self.zz_sites_list[:, 1]]) * 2 - 1
    #     diag_weights = - self.J * ZZ.sum(-1)  # negative sign due to convention
    #
    #     def gen_flipped():
    #         for i in range(self.nb_sites):
    #             flipped_samples = samples.at[..., i].set(1 - samples[..., i])  # checked! works with jit! need to check if this will replace the samples if jit compiled
    #             flipped_weights = - self.h  # negative sign due to convention
    #             yield flipped_samples, flipped_weights
    #
    #     return diag_weights, gen_flipped()

    def reverse_flip_func(self, samples):
        # ZZ term compute the parity of two sites
        ZZ = (samples[..., self.zz_sites_list[:, 0]] == samples[..., self.zz_sites_list[:, 1]]) * 2 - 1
        diag_weights = - self.J * ZZ.sum(-1).astype(complex)  # negative sign due to convention
        nb_flips = self.nb_sites
        def flip_func(i):
            flipped_samples = samples.at[..., i].set(1 - samples[..., i])
            flipped_weights = - self.h
            return flipped_samples, flipped_weights
        return diag_weights, (nb_flips, flip_func)



# test code
if __name__ == '__main__':
    # tfim1d1 = TFIMHamiltonian1D(4, J=1., h=0.5, periodic=True)
    # # tfim1d2 = TFIMHamiltonian(4, [(0, 1), (1, 2), (2, 3), (3, 0)], J=1., h=0.5)
    # tfim1d3 = SpinHalfOperator(
    #     nb_sites=4,
    #     op_list=[
    #         ('Z', 'Z'),
    #          ('Z', 'Z'),
    #          ('Z', 'Z'),
    #          ('Z', 'Z'),
    #          'X',
    #          'X',
    #          'X',
    #          'X'
    #     ],
    #     op_coef_list=[
    #         -1,
    #         -1,
    #         -1,
    #         -1,
    #         -0.5,
    #         -0.5,
    #         -0.5,
    #         -0.5
    #     ],
    #     op_sites_list=[
    #         (0, 1),
    #         (1, 2),
    #         (2, 3),
    #         (3, 0),
    #         0,
    #         1,
    #         2,
    #         3
    #     ]
    # )
    # tfim1d4 = SpinHalfOperator(
    #     nb_sites=4,
    #     op_list=[
    #         ('Z', 'Z'),
    #         ('Z', 'Z'),
    #         ('Z', 'Z'),
    #         ('Z', 'Z'),
    #         SpinHalfOperator.KnownOperators.X + SpinHalfOperator.KnownOperators.Y + SpinHalfOperator.KnownOperators.Z,
    #         SpinHalfOperator.KnownOperators.X + SpinHalfOperator.KnownOperators.Y + SpinHalfOperator.KnownOperators.Z,
    #         SpinHalfOperator.KnownOperators.X + SpinHalfOperator.KnownOperators.Y + SpinHalfOperator.KnownOperators.Z,
    #         SpinHalfOperator.KnownOperators.X + SpinHalfOperator.KnownOperators.Y + SpinHalfOperator.KnownOperators.Z,
    #         - SpinHalfOperator.KnownOperators.Y - SpinHalfOperator.KnownOperators.Z,
    #         - SpinHalfOperator.KnownOperators.Y - SpinHalfOperator.KnownOperators.Z,
    #         - SpinHalfOperator.KnownOperators.Y - SpinHalfOperator.KnownOperators.Z,
    #         - SpinHalfOperator.KnownOperators.Y - SpinHalfOperator.KnownOperators.Z
    #     ],
    #     op_coef_list=[
    #         -1,
    #         -1,
    #         -1,
    #         -1,
    #         -0.5,
    #         -0.5,
    #         -0.5,
    #         -0.5,
    #         -0.5,
    #         -0.5,
    #         -0.5,
    #         -0.5
    #     ],
    #     op_sites_list=[
    #         (0, 1),
    #         (1, 2),
    #         (2, 3),
    #         (3, 0),
    #         0,
    #         1,
    #         2,
    #         3,
    #         0,
    #         1,
    #         2,
    #         3,
    #     ]
    # )
    # print(tfim1d1.todense())
    # # print(tfim1d2.todense())
    # print(tfim1d3.todense())
    # # print(jnp.allclose(tfim1d1.todense(), tfim1d2.todense()))
    # print(jnp.allclose(tfim1d1.todense(), tfim1d3.todense()))
    tfim2d1 = TFIMHamiltonian2D(nb_rows=3, nb_cols=3, J=1, h=0.5, periodic=True)
    tfim2d2 = TFIMHamiltonian(nb_site=9, zz_sites_list=[(0, 1), (1, 2), (2, 0),
                                                        (3, 4), (4, 5), (5, 3),
                                                        (6, 7), (7, 8), (8, 6),
                                                        (0, 3), (3, 6), (6, 0),
                                                        (1, 4), (4, 7), (7, 1),
                                                        (2, 5), (5, 8), (8, 2),
                                                        ],
                              J=1, h=0.5)
    print(jnp.allclose(tfim2d1.todense(), tfim2d2.todense()))
    print()



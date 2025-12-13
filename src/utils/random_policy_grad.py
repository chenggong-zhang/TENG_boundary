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
import flax

NN = Any
Sampler = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PyTreeDef = Any

# @dataclass(frozen=True) # to make it immutable and hashable
# class RandomNaturalPolicyGradLSPure:
#     nb_params_total: int
#
#     @partial(jax.jit, static_argnums=(0, 2, 6))
#     def __call__(self, state, var_state_pure, samples, sqrt_weights, rewards, ls_solver, params_to_take):
#         """
#         natural policy gradient with a subset of parameters
#         """
#         jvp_raw, vjp_raw, value = var_state_pure.jvp_vjp_func(state, samples)
#
#         def jvp(tangents):
#             tangents = jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, tangents, inplace=False) # inplace=False is required in jax
#             pushforward = jvp_raw(tangents)
#             pushforward = pushforward * sqrt_weights
#             return pushforward
#
#         def vjp(cotangents):
#             cotangents = cotangents * sqrt_weights
#             pullback = vjp_raw(cotangents)
#             pullback = jnp.take(pullback, params_to_take)
#             return pullback
#
#         rewards = rewards * sqrt_weights
#
#         update, info = ls_solver(jvp, vjp, rewards)
#
#         return jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, update, inplace=False), info
#
#     @partial(jax.jit, static_argnums=(0, 1))
#     def sample_params(self, nb_params_to_take, rand_key):
#         rand_key, sub_rand_key = jrnd.split(rand_key)
#         return jrnd.choice(sub_rand_key, self.nb_params_total, shape=(nb_params_to_take,), replace=False), rand_key


@dataclass(frozen=True) # to make it immutable and hashable
class RandomNaturalPolicyGradLSPure:
    nb_params_total: int

    @partial(jax.jit, static_argnums=(0, 2, 6))
    def __call__(self, state, var_state_pure, samples, sqrt_weights, rewards, ls_solver, params_to_take):
        """
        natural policy gradient with a subset of parameters
        """

        curr_params = var_state_pure.flatten_parameters(state['params'])
        sampled_params = jnp.take(curr_params, params_to_take)

        # alternatively don't use the pmapped apply here, but manually put the data to different devices and copy state to different devices
        def net_apply_func(params):
            params = jnp.put(curr_params, params_to_take, params, inplace=False) # inplace=False is required in jax
            params = var_state_pure.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = var_state_pure.evaluate_pmapped(new_state, samples)
            return value

        def jvp_func(tangents):
            pushforwards = jax.jvp(net_apply_func, (sampled_params,), (tangents,))[1] * sqrt_weights
            return pushforwards

        value, vjp_func_raw = jax.vjp(net_apply_func, sampled_params)

        def vjp_func(cotangents):
            return vjp_func_raw(cotangents * sqrt_weights)[0]

        rewards = rewards * sqrt_weights

        update, info = ls_solver(jvp_func, vjp_func, rewards)

        return jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, update, inplace=False), info

    @partial(jax.jit, static_argnums=(0, 1))
    def sample_params(self, nb_params_to_take, rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        return jrnd.choice(sub_rand_key, self.nb_params_total, shape=(nb_params_to_take,), replace=False), rand_key


class RandomNaturalPolicyGradLS:

    def __init__(self, var_state, ls_solver, nb_params_to_take=None, rand_seed=8848):
        self.var_state = var_state
        self.ls_solver = ls_solver
        self.rand_key = jax.random.PRNGKey(rand_seed)
        self.nb_params_total = var_state.count_parameters()
        if nb_params_to_take is None:
            self.nb_params_to_take = self.nb_params_total
        else:
            self.nb_params_to_take = min(nb_params_to_take, self.nb_params_total)
        self.pure_funcs = RandomNaturalPolicyGradLSPure(self.nb_params_total)
        self.params_to_take = None

    def sample_params(self):
        self.params_to_take, self.rand_key = self.pure_funcs.sample_params(self.nb_params_to_take, self.rand_key)
        # self.params_to_take = jnp.sort(self.params_to_take)
        return self.params_to_take

    def __call__(self, samples, sqrt_weights, rewards, *, var_state=None, resample_params=False):
        if resample_params:
            self.sample_params()
        if var_state is None:
            var_state = self.var_state.state
        return self.pure_funcs(var_state.state, var_state.pure_funcs, samples, sqrt_weights, rewards, self.ls_solver, self.params_to_take)


# @dataclass(frozen=True)
# class RandomNaturalPolicyGradTDVPPure:
#     """
#     This is actually the mclanchlan's principle (or real dirac's principle)
#     """
#     nb_params_total: int
#
#     @partial(jax.jit, static_argnums=(0, 2, 6))
#     def __call__(self, state, var_state_pure, samples, sqrt_weights, rewards, ls_solver, params_to_take):
#         """
#         natural policy gradient with a subset of parameters
#         """
#
#         curr_params = var_state_pure.flatten_parameters(state['params'])
#         sampled_params = jnp.take(curr_params, params_to_take)
#
#         # alternatively don't use the pmapped apply here, but manually put the data to different devices and copy state to different devices
#         @jax.jit
#         def net_apply_func(params):
#             params = jnp.put(curr_params, params_to_take, params, inplace=False)  # inplace=False is required in jax
#             params = var_state_pure.unflatten_parameters(params)
#             new_state = flax.core.copy(state, add_or_replace={'params': params})
#             value = var_state_pure.evaluate(new_state, samples.squeeze(0))[None, ...]
#             return value
#
#         jac = jax.jacfwd(net_apply_func)(sampled_params) * sqrt_weights[..., None]
#
#         rewards = rewards * sqrt_weights
#
#         update, res = jnp.linalg.lstsq(jac.squeeze(0), rewards.squeeze(0), rcond=1e-14)[:2]
#
#         return jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, update, inplace=False), (res,)
#
#     @partial(jax.jit, static_argnums=(0, 1))
#     def sample_params(self, nb_params_to_take, rand_key):
#         rand_key, sub_rand_key = jrnd.split(rand_key)
#         return jrnd.choice(sub_rand_key, self.nb_params_total, shape=(nb_params_to_take,), replace=False), rand_key


@dataclass(frozen=True)
class RandomNaturalPolicyGradTDVPPure:
    """
    This is actually the mclanchlan's principle (or real dirac's principle)
    """
    nb_params_total: int

    @partial(jax.jit, static_argnums=(0, 2, 6))
    def __call__(self, state, var_state_pure, samples, sqrt_weights, rewards, ls_solver, params_to_take):
        """
        natural policy gradient with a subset of parameters
        """

        curr_params = var_state_pure.flatten_parameters(state['params'])
        # sampled_params = jnp.take(curr_params, params_to_take)

        # alternatively don't use the pmapped apply here, but manually put the data to different devices and copy state to different devices
        def net_apply_func(params, sample):
            params = var_state_pure.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = var_state_pure.evaluate(new_state, sample[None, ...]).squeeze(0)
            return value

        jac = jax.vmap(jax.grad(net_apply_func), (None, 0))(curr_params, samples.squeeze(0))[..., params_to_take]  * sqrt_weights[0, ..., None]

        rewards = rewards  * sqrt_weights

        update, res = jnp.linalg.lstsq(jac, rewards.squeeze(0))[:2]

        return jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, update, inplace=False), (res,)

    @partial(jax.jit, static_argnums=(0, 1))
    def sample_params(self, nb_params_to_take, rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        return jrnd.choice(sub_rand_key, self.nb_params_total, shape=(nb_params_to_take,), replace=False), rand_key


class RandomNaturalPolicyGradTDVP:
    """
    This is actually the mclanchlan's principle (or real dirac's principle)
    """

    def __init__(self, var_state, ls_solver, nb_params_to_take=None, rand_seed=8848):
        self.var_state = var_state
        self.ls_solver = ls_solver
        self.rand_key = jax.random.PRNGKey(rand_seed)
        self.nb_params_total = var_state.count_parameters()
        if nb_params_to_take is None:
            self.nb_params_to_take = self.nb_params_total
        else:
            self.nb_params_to_take = min(nb_params_to_take, self.nb_params_total)
        self.pure_funcs = RandomNaturalPolicyGradTDVPPure(self.nb_params_total)
        self.params_to_take = None

    def sample_params(self):
        self.params_to_take, self.rand_key = self.pure_funcs.sample_params(self.nb_params_to_take, self.rand_key)
        # self.params_to_take = jnp.sort(self.params_to_take)
        return self.params_to_take

    def __call__(self, samples, sqrt_weights, rewards, *, var_state=None, resample_params=False):
        if resample_params:
            self.sample_params()
        if var_state is None:
            var_state = self.var_state
        return self.pure_funcs(var_state.state, var_state.pure_funcs, samples, sqrt_weights, rewards, self.ls_solver, self.params_to_take)


@dataclass(frozen=True)
class RandomNaturalPolicyGradTDVPPure2:
    """
    This is actually the mclanchlan's principle (or real dirac's principle)
    """
    nb_params_total: int
    rcond: float = 1e-4

    @partial(jax.jit, static_argnums=(0, 2, 6))
    def __call__(self, state, var_state_pure, samples, sqrt_weights, rewards, ls_solver, params_to_take):
        """
        natural policy gradient with a subset of parameters
        """

        curr_params = var_state_pure.flatten_parameters(state['params'])
        # sampled_params = jnp.take(curr_params, params_to_take)

        # alternatively don't use the pmapped apply here, but manually put the data to different devices and copy state to different devices
        def net_apply_func(params, sample):
            params = var_state_pure.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = var_state_pure.evaluate(new_state, sample[None, ...]).squeeze(0)
            return value

        jac = jax.vmap(jax.grad(net_apply_func), (None, 0))(curr_params, samples.squeeze(0))[..., params_to_take]  * sqrt_weights[0, ..., None]

        rewards = rewards  * sqrt_weights

        update, res = jnp.linalg.lstsq(jac, rewards.squeeze(0), rcond=self.rcond)[:2]

        return jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, update, inplace=False), (res,)

    @partial(jax.jit, static_argnums=(0, 1))
    def sample_params(self, nb_params_to_take, rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        return jrnd.choice(sub_rand_key, self.nb_params_total, shape=(nb_params_to_take,), replace=False), rand_key


class RandomNaturalPolicyGradTDVP2:
    """
    This is actually the mclanchlan's principle (or real dirac's principle)
    """

    def __init__(self, var_state, ls_solver, nb_params_to_take=None, rand_seed=8848, rcond=1e-4):
        self.var_state = var_state
        self.ls_solver = ls_solver
        self.rand_key = jax.random.PRNGKey(rand_seed)
        self.nb_params_total = var_state.count_parameters()
        if nb_params_to_take is None:
            self.nb_params_to_take = self.nb_params_total
        else:
            self.nb_params_to_take = min(nb_params_to_take, self.nb_params_total)
        self.pure_funcs = RandomNaturalPolicyGradTDVPPure2(self.nb_params_total, rcond)
        self.params_to_take = None

    def sample_params(self):
        self.params_to_take, self.rand_key = self.pure_funcs.sample_params(self.nb_params_to_take, self.rand_key)
        # self.params_to_take = jnp.sort(self.params_to_take)
        return self.params_to_take

    def __call__(self, samples, sqrt_weights, rewards, *, var_state=None, resample_params=False):
        if resample_params:
            self.sample_params()
        if var_state is None:
            var_state = self.var_state
        return self.pure_funcs(var_state.state, var_state.pure_funcs, samples, sqrt_weights, rewards, self.ls_solver, self.params_to_take)


@dataclass(frozen=True) # to make it immutable and hashable
class StackedRandomNaturalPolicyGradLSPure:
    nb_params_total: int

    @partial(jax.jit, static_argnums=(0, 2, 6))
    def __call__(self, state, var_state_pure, samples, sqrt_weights, rewards, ls_solver, params_to_take):
        """
        natural policy gradient with a subset of parameters
        """
        jvp_raw, vjp_raw, value = var_state_pure.jvp_vjp_func(state, samples)
        length_ratio = rewards.size // value.size

        def jvp(tangents):
            tangents = jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, tangents, inplace=False) # inplace=False is required in jax
            pushforward = jvp_raw(tangents)
            pushforward = pushforward * sqrt_weights
            return jnp.repeat(pushforward, length_ratio, axis=-1)

        def vjp(cotangents):
            cotangentss = jnp.split(cotangents.reshape(1, -1, length_ratio), length_ratio, axis=-1)
            res = 0.
            for cotangents in cotangentss:
                cotangents = cotangents.squeeze(-1)
                cotangents = cotangents * sqrt_weights
                pullback = vjp_raw(cotangents)
                pullback = jnp.take(pullback, params_to_take)
                res += pullback
            return res

        rewards = rewards * jnp.repeat(sqrt_weights, length_ratio, axis=-1)

        update, info = ls_solver(jvp, vjp, rewards)

        return jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, update, inplace=False), info

    @partial(jax.jit, static_argnums=(0, 1))
    def sample_params(self, nb_params_to_take, rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        return jrnd.choice(sub_rand_key, self.nb_params_total, shape=(nb_params_to_take,), replace=False), rand_key


class StackedRandomNaturalPolicyGradLS:

    def __init__(self, var_state, ls_solver, nb_params_to_take=None, rand_seed=8848):
        self.var_state = var_state
        self.ls_solver = ls_solver
        self.rand_key = jax.random.PRNGKey(rand_seed)
        self.nb_params_total = var_state.count_parameters()
        if nb_params_to_take is None:
            self.nb_params_to_take = self.nb_params_total
        else:
            self.nb_params_to_take = min(nb_params_to_take, self.nb_params_total)
        self.pure_funcs = StackedRandomNaturalPolicyGradLSPure(self.nb_params_total)
        self.params_to_take = None

    def sample_params(self):
        self.params_to_take, self.rand_key = self.pure_funcs.sample_params(self.nb_params_to_take, self.rand_key)
        # self.params_to_take = jnp.sort(self.params_to_take)
        return self.params_to_take

    def __call__(self, samples, sqrt_weights, rewards, *, var_state=None, resample_params=False):
        if resample_params:
            self.sample_params()
        if var_state is None:
            var_state = self.var_state
        return self.pure_funcs(var_state.state, var_state.pure_funcs, samples, sqrt_weights, rewards, self.ls_solver, self.params_to_take)



@dataclass(frozen=True)
class StackedRandomNaturalPolicyGradTDVPPure:
    """
    This is actually the mclanchlan's principle (or real dirac's principle)
    """
    nb_params_total: int

    @partial(jax.jit, static_argnums=(0, 2, 6))
    def __call__(self, state, var_state_pure, samples, sqrt_weights, rewards, ls_solver, params_to_take):
        """
        natural policy gradient with a subset of parameters
        """

        curr_params = var_state_pure.flatten_parameters(state['params'])
        # sampled_params = jnp.take(curr_params, params_to_take)

        # alternatively don't use the pmapped apply here, but manually put the data to different devices and copy state to different devices
        def net_apply_func(params, sample):
            params = var_state_pure.unflatten_parameters(params)
            new_state = flax.core.copy(state, add_or_replace={'params': params})
            value = var_state_pure.evaluate(new_state, sample[None, ...]).squeeze(0)
            return value

        jac = jax.vmap(jax.grad(net_apply_func), (None, 0))(curr_params, samples.squeeze(0))[..., params_to_take]#  * sqrt_weights[0, ..., None]

        rewards = rewards#  * sqrt_weights

        update, res = jnp.linalg.lstsq(jnp.repeat(jac, 3, axis=0), rewards.squeeze(0))[:2]

        return jnp.put(jnp.zeros((self.nb_params_total,)), params_to_take, update, inplace=False), (res,)

    @partial(jax.jit, static_argnums=(0, 1))
    def sample_params(self, nb_params_to_take, rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        return jrnd.choice(sub_rand_key, self.nb_params_total, shape=(nb_params_to_take,), replace=False), rand_key


class StackedRandomNaturalPolicyGradTDVP:
    """
    This is actually the mclanchlan's principle (or real dirac's principle)
    """

    def __init__(self, var_state, ls_solver, nb_params_to_take=None, rand_seed=8848):
        self.var_state = var_state
        self.ls_solver = ls_solver
        self.rand_key = jax.random.PRNGKey(rand_seed)
        self.nb_params_total = var_state.count_parameters()
        if nb_params_to_take is None:
            self.nb_params_to_take = self.nb_params_total
        else:
            self.nb_params_to_take = min(nb_params_to_take, self.nb_params_total)
        self.pure_funcs = StackedRandomNaturalPolicyGradTDVPPure(self.nb_params_total)
        self.params_to_take = None

    def sample_params(self):
        self.params_to_take, self.rand_key = self.pure_funcs.sample_params(self.nb_params_to_take, self.rand_key)
        # self.params_to_take = jnp.sort(self.params_to_take)
        return self.params_to_take

    def __call__(self, samples, sqrt_weights, rewards, *, var_state=None, resample_params=False):
        if resample_params:
            self.sample_params()
        if var_state is None:
            var_state = self.var_state
        return self.pure_funcs(var_state.state, var_state.pure_funcs, samples, sqrt_weights, rewards, self.ls_solver, self.params_to_take)
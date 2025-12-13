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

from math import sqrt
import jax
import jax.numpy as jnp

def compute_observables_and_fidelities(var_state, *obs, psi2_ratio_funcs=None, nb_batches=10, sampler=None):
    """
    we may need to change this for autoregressive sampling in the future,
    but it might be a good idea to still write a sampler for autoregressive models as well
    just to separate the sampling process out of the var_state
    var_state: var_state
    *obs: any number of observables that are subclasses of AbstractOperator
    psi2_ratio_funcs: a list of tuple [(callable, bool)...], where each callable should return phi(x)/psi(x) where psi(x) is the var_state and the bool indicates that whether we can assume both var_state and phi are normalized
    """
    if sampler is None:
        sampler = var_state.sampler

    nb_obs = len(obs)
    # nb_psi2_ratio_funcs = len(psi2_ratio_funcs)

    if psi2_ratio_funcs is not None:
        all_obs = obs + tuple(psi2_ratio_func for (psi2_ratio_func, is_normalized) in psi2_ratio_funcs)
    else:
        all_obs = obs

    if sampler.is_exact_sampler:
        all_obs_means, all_obs_errs, all_obs_vars = compute_observables_exact_sampler(var_state, *all_obs, sampler=sampler)
    else:
        all_obs_means, all_obs_errs, all_obs_vars = compute_observables_mc_sampler(var_state, *all_obs, nb_batches=nb_batches, sampler=sampler)

    obs_means = all_obs_means[:nb_obs]
    obs_errs = all_obs_errs[:nb_obs]
    obs_vars = all_obs_vars[:nb_obs]

    vdot_means = all_obs_means[nb_obs:]
    vdot_errs = all_obs_errs[nb_obs:]
    vdot_vars = all_obs_vars[nb_obs:]

    if psi2_ratio_funcs is not None:
        fids = [abs(vdot_mean)**2 if is_normalized else abs(vdot_mean)**2 / (vdot_var + abs(vdot_mean)**2) for (vdot_mean, vdot_var, (psi2_ratio_func, is_normalized)) in zip(vdot_means, vdot_vars, psi2_ratio_funcs)]
        fids_err = [2 * abs(vdot_mean) * vdot_err if is_normalized else 2 * abs(vdot_mean) * vdot_err / (vdot_var + abs(vdot_mean)**2) for (vdot_mean, vdot_err, vdot_var, (psi2_ratio_func, is_normalized)) in zip(vdot_means, vdot_errs, vdot_vars, psi2_ratio_funcs)]
    else:
        fids = []
        fids_err = []
    return (obs_means, obs_errs, obs_vars), (fids, fids_err)


def compute_observables(var_state, *obs, nb_batches=10, sampler=None):
    """
    we may need to change this for autoregressive sampling in the future,
    but it might be a good idea to still write a sampler for autoregressive models as well
    just to separate the sampling process out of the var_state
    var_state: var_state
    *obs: any number of observables that are subclasses of AbstractOperator
    """
    if sampler is None:
        sampler = var_state.sampler

    if sampler.is_exact_sampler:
        return compute_observables_exact_sampler(var_state, *obs, sampler=sampler)
    else:
        return compute_observables_mc_sampler(var_state, *obs, nb_batches=nb_batches, sampler=sampler)


def compute_observables_exact_sampler(var_state, *obs, sampler=None):
    assert sampler is not None and sampler.is_exact_sampler
    obs_means = [0.j] * len(obs)
    obs_errs = [0.] * len(obs)
    obs_vars = [0.] * len(obs)
    samples, log_psi, sqrt_weights = sampler.sample()
    weights = jnp.square(sqrt_weights)
    for i in range(len(obs)):
        local_obs = obs[i](var_state, samples, log_psi)
        obs_means[i] = obs_means[i] + (local_obs * weights).sum().item()
        obs_vars[i] = (jnp.square(jnp.abs(local_obs - obs_means[i])) * weights).sum().item()

    return obs_means, obs_errs, obs_vars


def compute_observables_mc_sampler(var_state, *obs, nb_batches=10, sampler=None):
    """
    assume samples are drawn from |psi|^2
    """
    assert sampler is not None and not sampler.is_exact_sampler
    obs_means = [0.j] * len(obs)
    obs2_means = [0.] * len(obs)
    obs_shifts = [0.j] * len(obs)
    # obs_scales = [0.] * len(obs)
    nb_samples_total = nb_batches * sampler.nb_samples

    # sample first batch
    samples, log_psi, sqrt_weights = sampler.sample()
    for i in range(len(obs)):
        local_obs = obs[i](var_state, samples, log_psi)
        obs_mean = local_obs.mean().item()
        local_obs = local_obs - obs_mean
        # obs_scale = sqrt(jnp.square(jnp.abs(local_obs)).mean().item())
        obs2_means[i] = obs2_means[i] + jnp.square(jnp.abs(local_obs)).mean().item()
        obs_shifts[i] = obs_shifts[i] + obs_mean
        # obs_scales[i] = obs_scales[i] + obs_scale

    for _ in range(nb_batches - 1):
        samples, log_psi, sqrt_weights = sampler.sample()
        for i in range(len(obs)):
            local_obs = obs[i](var_state, samples, log_psi)
            local_obs = (local_obs - obs_shifts[i])
            obs_means[i] = obs_means[i] + local_obs.mean().item()
            obs2_means[i] = obs2_means[i] + jnp.square(jnp.abs(local_obs)).mean().item()

    obs_means = [obs_mean / nb_batches for obs_mean in obs_means]
    obs2_means = [obs2_mean / nb_batches for obs2_mean in obs2_means]
    obs_vars = [obs2_mean - abs(obs_mean)**2 for obs_mean, obs2_mean in zip(obs_means, obs2_means)]

    obs_means = [obs_mean + obs_shift for obs_mean, obs_shift in zip(obs_means, obs_shifts)]
    # obs_means = [obs_mean * obs_scale + obs_shift for obs_mean, obs_scale, obs_shift in zip(obs_means, obs_scales, obs_shifts)]
    # obs_vars = [obs_var * obs_scale**2 for obs_var, obs_scale in zip(obs_vars, obs_scales)]
    obs_errs = [sqrt(obs_var / nb_samples_total) for obs_var in obs_vars]

    return obs_means, obs_errs, obs_vars



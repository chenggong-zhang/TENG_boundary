from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax import tree_util

### please follow the convention that everything after rewards are keyword only arguments

### simple quantum policy gradient ###
def simple_q_policy_grad(var_state, samples, sqrt_weights, rewards, *, joint_compile=True):
    """
    rewards: does not need to subtract mean. will be handled inside
    """
    if joint_compile:
        return simple_q_policy_grad_joint_jit(var_state.get_state(), var_state.pure_funcs, samples, sqrt_weights, rewards)
    else:
        return simple_q_policy_grad_no_jit(var_state, samples, sqrt_weights, rewards)

@partial(jax.jit, static_argnums=1)
def simple_q_policy_grad_joint_jit(state, var_state_pure, samples, sqrt_weights, rewards):
    jvp_raw, vjp_raw, log_psi = var_state_pure.jvp_vjp_log_psi_func(state, samples)
    weights = jnp.square(sqrt_weights)
    rewards = rewards - (weights * rewards).sum()
    return vjp_raw(rewards * weights), () # place holder for info, to be compatible with ls solver

def simple_q_policy_grad_no_jit(var_state, samples, sqrt_weights, rewards):
    jvp_raw, vjp_raw, log_psi = var_state.jvp_vjp_log_psi_func(samples, jit=False)
    weights = jnp.square(sqrt_weights)
    rewards = rewards - (weights * rewards).sum()
    return vjp_raw(rewards * weights), () # place holder for info, to be compatible with ls solver


### natural quantum policy gradient with least square algorithm ###
def natural_q_policy_grad_ls(var_state, samples, sqrt_weights, rewards, *, ls_solver=None, joint_compile=True, real_output=False):
    """
    rewards: does not need to subtract mean. will be handled inside
    ls_solver: should be a callable of the signiture ls_solver(A: callable, A^T: callable, b: Array) -> (x, info_tuple)
               if it requires extra kwargs, must be handled beforehand with partial(ls_solver, **kwargs)
    """
    if joint_compile:
        if real_output:
            return natural_q_policy_grad_ls_real_joint_jit(var_state.get_state(), var_state.pure_funcs, samples, sqrt_weights, rewards, ls_solver)
        else:
            return natural_q_policy_grad_ls_joint_jit(var_state.get_state(), var_state.pure_funcs, samples, sqrt_weights, rewards, ls_solver)
    else:
        if real_output:
            return natural_q_policy_grad_ls_real_separate_jit(var_state, samples, sqrt_weights, rewards, ls_solver)
        else:
            return natural_q_policy_grad_ls_separate_jit(var_state, samples, sqrt_weights, rewards, ls_solver)

@partial(jax.jit, static_argnums=(1, 5))
def natural_q_policy_grad_ls_joint_jit(state, var_state_pure, samples, sqrt_weights, rewards, ls_solver):
    jvp_raw, vjp_raw, log_psi = var_state_pure.jvp_vjp_log_psi_func(state, samples)
    weights = jnp.square(sqrt_weights)
    vjp_raw_weights_real = vjp_raw(weights + 0j) # store this to use in vjp to compute the mean
    vjp_raw_weights_imag = vjp_raw(weights * 1j) # store this to use in vjp to compute the mean

    def jvp(tangents):
        pushforwards = jvp_raw(tangents)
        pushforwards = pushforwards - (weights * pushforwards).sum()
        pushforwards = pushforwards * sqrt_weights
        # commented out because complex numbers work as well and is equivalent to minsr or McLachlan's principle
        # # separate real and imaginary part to be solved as a real-valued lstsq problem, equivalent to McLachlan's principle
        # pushforwards = jnp.concatenate([pushforwards.real, pushforwards.imag], 1) # concatenate along the batch dimension (1) (0 is device dimension)
        return pushforwards

    def vjp(cotangents):
        # commented out because complex numbers work as well and is equivalent to minsr or McLachlan's principle
        # # combine real and imaginary part of cotangents first
        # cotangents = cotangents.reshape(cotangents.shape[0], 2, -1) # split the batch dimension
        # cotangents = cotangents[:, 0, :] + cotangents[:, 1, :] * 1j
        cotangents = cotangents * sqrt_weights
        pullbacks = vjp_raw(cotangents)
        cotangents_sum = cotangents.sum()
        pullbacks_mean = cotangents_sum.real * vjp_raw_weights_real + cotangents_sum.imag * vjp_raw_weights_imag # vjp_raw(weights) * cotangents.sum()
        return pullbacks - pullbacks_mean

    rewards = rewards - (weights * rewards).sum()
    rewards = rewards * sqrt_weights
    # rewards = jnp.concatenate([rewards.real, rewards.imag], 1)

    return ls_solver(jvp, vjp, rewards)

@partial(jax.jit, static_argnums=(1, 5))
def natural_q_policy_grad_ls_real_joint_jit(state, var_state_pure, samples, sqrt_weights, rewards, ls_solver):
    jvp_raw, vjp_raw, log_psi = var_state_pure.jvp_vjp_log_psi_func(state, samples)
    weights = jnp.square(sqrt_weights)
    vjp_raw_weights = vjp_raw(weights) # store this to use in vjp to compute the mean

    def jvp(tangents):
        pushforwards = jvp_raw(tangents)
        pushforwards = pushforwards - (weights * pushforwards).sum()
        pushforwards = pushforwards * sqrt_weights

        return pushforwards

    def vjp(cotangents):
        cotangents = cotangents * sqrt_weights
        pullbacks = vjp_raw(cotangents)
        cotangents_sum = cotangents.sum()
        pullbacks_mean = cotangents_sum * vjp_raw_weights # + cotangents_sum.imag * vjp_raw_weights_imag # vjp_raw(weights) * cotangents.sum()
        return pullbacks - pullbacks_mean

    rewards = rewards - (weights * rewards).sum()
    rewards = rewards * sqrt_weights

    return ls_solver(jvp, vjp, rewards)

def natural_q_policy_grad_ls_separate_jit(var_state, samples, sqrt_weights, rewards, ls_solver):
    jvp_raw, vjp_raw, log_psi = var_state.jvp_vjp_log_psi_func(samples, jit=False)
    weights = jnp.square(sqrt_weights)
    vjp_raw_weights_real = vjp_raw(weights + 0j)
    vjp_raw_weights_imag = vjp_raw(weights * 1j)

    @jax.jit
    def jvp(tangents):
        pushforwards = jvp_raw(tangents)
        pushforwards = pushforwards - (weights * pushforwards).sum()
        pushforwards = pushforwards * sqrt_weights
        # commented out because complex numbers work as well and is equivalent to minsr or McLachlan's principle
        # # separate real and imaginary part to be solved as a real-valued lstsq problem, equivalent to McLachlan's principle
        # pushforwards = jnp.concatenate([pushforwards.real, pushforwards.imag], 1)  # concatenate along the batch dimension (1) (0 is device dimension)
        return pushforwards

    @jax.jit
    def vjp(cotangents):
        # commented out because complex numbers work as well and is equivalent to minsr or McLachlan's principle
        # combine real and imaginary part of cotangents first
        # cotangents = cotangents.reshape(cotangents.shape[0], 2, -1)  # split the batch dimension
        # cotangents = cotangents[:, 0, :] + cotangents[:, 1, :] * 1j
        cotangents = cotangents * sqrt_weights
        pullbacks = vjp_raw(cotangents)
        cotangents_sum = cotangents.sum()
        pullbacks_mean = cotangents_sum.real * vjp_raw_weights_real + cotangents_sum.imag * vjp_raw_weights_imag  # vjp_raw(weights) * cotangents.sum()
        return pullbacks - pullbacks_mean

    rewards = rewards - (weights * rewards).sum()
    rewards = rewards * sqrt_weights

    return ls_solver(jvp, vjp, rewards)


def natural_q_policy_grad_ls_real_separate_jit(var_state, samples, sqrt_weights, rewards, ls_solver):
    jvp_raw, vjp_raw, log_psi = var_state.jvp_vjp_log_psi_func(samples, jit=False)
    weights = jnp.square(sqrt_weights)
    vjp_raw_weights = vjp_raw(weights)

    @jax.jit
    def jvp(tangents):
        pushforwards = jvp_raw(tangents)
        pushforwards = pushforwards - (weights * pushforwards).sum()
        pushforwards = pushforwards * sqrt_weights
        return pushforwards

    @jax.jit
    def vjp(cotangents):
        cotangents = cotangents * sqrt_weights
        pullbacks = vjp_raw(cotangents)
        cotangents_sum = cotangents.sum()
        pullbacks_mean = cotangents_sum * vjp_raw_weights # vjp_raw(weights) * cotangents.sum()
        return pullbacks - pullbacks_mean

    rewards = rewards - (weights * rewards).sum()
    rewards = rewards * sqrt_weights

    return ls_solver(jvp, vjp, rewards)


### natural quantum policy gradient with minsr algorithm ###
def natural_q_policy_grad_minsr(var_state, samples, sqrt_weights, rewards, *, rcond=1e-14, joint_compile=True, real_output=False):
    """
    rewards: does not need to subtract mean. will be handled inside
    """
    if joint_compile:
        return natural_q_policy_grad_minsr_joint_jit(var_state.get_state(), var_state.pure_funcs, samples, sqrt_weights, rewards, rcond=rcond, real_output=real_output)
    else:
        return natural_q_policy_grad_minsr_no_jit(var_state, samples, sqrt_weights, rewards, rcond=rcond, real_output=real_output)

@partial(jax.jit, static_argnums=(1, 6))
def natural_q_policy_grad_minsr_joint_jit(state, var_state_pure, samples, sqrt_weights, rewards, rcond=1e-14, real_output=False):
    weights = jnp.square(sqrt_weights)
    jacobian = var_state_pure.jac_log_psi_pmapped(state, samples)
    jacobian = jacobian - (jacobian * weights[:, :, None]).sum((0, 1))
    jacobian = jacobian * sqrt_weights[:, :, None]
    jacobian = jacobian.reshape(-1, jacobian.shape[-1])
    if not real_output:
        jacobian = jnp.concatenate([jacobian.real, jacobian.imag], 0)  # view as real to solve lstsq problem, equivalent to McLachlan's principle
    ntk = jacobian @ jacobian.T
    ntk_inv = jnp.linalg.pinv(ntk, rcond=rcond, hermitian=True)
    rewards = rewards - (rewards * weights).sum((0, 1))
    rewards = rewards * sqrt_weights
    rewards = rewards.ravel()
    if not real_output:
        rewards = jnp.concatenate([rewards.real, rewards.imag], 0)  # same with rewards
    return jacobian.T @ (ntk_inv @ rewards), ()  # place holder for info, to be compatible with ls solver

def natural_q_policy_grad_minsr_no_jit(var_state, samples, sqrt_weights, rewards, rcond=1e-14, real_output=False):
    weights = jnp.square(sqrt_weights)
    jacobian = var_state.jac_log_psi(samples)
    jacobian = jacobian - (jacobian * weights[:, :, None]).sum((0, 1))
    jacobian = jacobian * sqrt_weights[:, :, None]
    jacobian = jacobian.reshape(-1, jacobian.shape[-1])
    if not real_output:
        jacobian = jnp.concatenate([jacobian.real, jacobian.imag], 0) # view as real to solve lstsq problem, equivalent to McLachlan's principle
    ntk = jacobian @ jacobian.T
    ntk_inv = jnp.linalg.pinv(ntk, rcond=rcond, hermitian=True)
    rewards = rewards - (rewards * weights).sum((0, 1))
    rewards = rewards * sqrt_weights
    rewards = rewards.ravel()
    if not real_output:
        rewards = jnp.concatenate([rewards.real, rewards.imag], 0) # same with rewards
    return jacobian.T @ (ntk_inv @ rewards), () # place holder for info, to be compatible with ls solver



### natural quantum policy gradient with minsr algorithm ###
def natural_q_policy_grad_tdvp(var_state, samples, sqrt_weights, rewards, *, rcond=1e-14, version='dirac', joint_compile=True):
    """
    rewards: does not need to subtract mean. will be handled inside
    """
    assert version in ['mclanchlan', 'tdvp'], f'{version} not supported, current code only support mclanchlan and tdvp (even dirac does not work))'# because we always map parameter to real
    if joint_compile:
        return natural_q_policy_grad_tdvp_joint_jit(var_state.get_state(), var_state.pure_funcs, samples, sqrt_weights, rewards, rcond=rcond, version=version)
    else:
        return natural_q_policy_grad_tdvp_no_jit(var_state, samples, sqrt_weights, rewards, rcond=rcond, version=version)

@partial(jax.jit, static_argnums=(1, 6))
def natural_q_policy_grad_tdvp_joint_jit(state, var_state_pure, samples, sqrt_weights, rewards, rcond=1e-14, version='dirac'):
    """
    rewards: does not need to subtract mean. will be handled inside
    """
    weights = jnp.square(sqrt_weights)
    jacobian = var_state_pure.jac_log_psi_pmapped(state, samples)
    jacobian = jacobian - (jacobian * weights[:, :, None]).sum((0, 1))
    jacobian = jacobian * sqrt_weights[:, :, None]
    jacobian = jacobian.reshape(-1, jacobian.shape[-1])
    metric = jacobian.conj().T @ jacobian
    rewards = rewards - (rewards * weights).sum((0, 1))
    rewards = rewards * sqrt_weights
    rewards = rewards.ravel()
    force = jacobian.conj().T @ rewards
    if version == 'mclanchlan':
        metric = metric.real
        force = force.real  # usually we should take imaginary but since this is reward, which is already -i H dt for real time evolution, taking real part is correct
    elif version == 'tdvp':
        metric = metric.imag
        force = - force.imag  # usually we should take real but since this is reward, which is already -i H dt for real time evolution, taking - imaginary part is correct
    metric_inv = jnp.linalg.pinv(metric, rcond=rcond, hermitian=True)
    return metric_inv @ force, ()  # place holder for info, to be compatible with ls solver

def natural_q_policy_grad_tdvp_no_jit(var_state, samples, sqrt_weights, rewards, rcond=1e-14, version='dirac'):
    weights = jnp.square(sqrt_weights)
    jacobian = var_state.jac_log_psi(samples)
    jacobian = jacobian - (jacobian * weights[:, :, None]).sum((0, 1))
    jacobian = jacobian * sqrt_weights[:, :, None]
    jacobian = jacobian.reshape(-1, jacobian.shape[-1])
    metric = jacobian.conj().T @ jacobian
    rewards = rewards - (rewards * weights).sum((0, 1))
    rewards = rewards * sqrt_weights
    rewards = rewards.ravel()
    force = jacobian.conj().T @ rewards
    if version == 'mclanchlan':
        metric = metric.real
        force = force.real # usually we should take imaginary but since this is reward, which is already -i H dt for real time evolution, taking real part is correct
    elif version == 'tdvp':
        metric = metric.imag
        force = - force.imag # usually we should take real but since this is reward, which is already -i H dt for real time evolution, taking - imaginary part is correct
    metric_inv = jnp.linalg.pinv(metric, rcond=rcond, hermitian=True)
    return metric_inv @ force, () # place holder for info, to be compatible with ls solver







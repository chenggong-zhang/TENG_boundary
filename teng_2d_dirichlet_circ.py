import argparse
import copy
from argparse import ArgumentParser
import time
from datetime import datetime
import os
import shutil
import logging
import random
import json
import pickle
from functools import partial
from decimal import Decimal

import jax
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)
# config.update("jax_debug_nans", True)

import jax.random as jrnd
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn

import optax

import numpy as np
import scipy as sp

from src.model import PDENet
from src.sampler import CircularQuadratureSampler2
from src.var_state import SimpleVarStateReal
from src.operator import HeatOperatorNoLog, AlanCahnOperator, BurgersOperator
from src.utils import RandomNaturalPolicyGradTDVP2, boolargparse

now = datetime.now()


def get_config():
    parser = ArgumentParser()

    ### general configs ###
    parser.add_argument("--nb_dims", type=int, default=2)
    parser.add_argument("--mu", type=float, nargs='+', default=[0])
    parser.add_argument("--D", type=float, nargs='+', default=[0.1])
    parser.add_argument("--equation", type=str, default='heat',
                        choices=['heat'])
    parser.add_argument("--nb_steps", type=int, default=4000)
    parser.add_argument("--nb_iters_per_step", type=int, default=5)
    parser.add_argument("--dt", type=str, default='0.002')
    parser.add_argument("--integrator", type=str, default='heun',
                        choices=['euler', 'heun', 'rk4'])
    parser.add_argument("--save_dir", type=str, nargs='?', default=None)  # can be emtpy
    # do this by setting enviroment variable from outside
    parser.add_argument("--load_config_from", type=str, nargs='?', default=None,
                        help='if specified, will load the config from the provided json file and overwrite current config, save_dir and load_config_from will not be overwritten')  # can be empty

    ### model configs ###
    parser.add_argument("--model", type=str, default='PDENet',
                        choices=['PDENet'])
    parser.add_argument("--load_model_state_from", type=str, nargs='?',
                        default='results/PDyNG/test_03-27-2024-12-06-37_1024_7_40_no_resnet_tanh_large_lr2_dirichlet_tdvp_circ_init_learn_boundary/model_state.pickle')
    parser.add_argument("--model_seed", type=int, default=1234)

    ### sampler configs ###
    parser.add_argument("--load_sampler_state_from", type=str, nargs='?', default=None)
    parser.add_argument("--nb_samples", type=int, default=262144,
                        help='number of samples')
    parser.add_argument("--sampler_seed", type=int, default=4321)
    parser.add_argument("--random_sample", type=boolargparse, default=False)

    ### policy grad configs ###
    parser.add_argument("--policy_grad_nb_params", type=int, default=768)
    parser.add_argument("--policy_grad_seed", type=int, default=8844)
    parser.add_argument("--policy_grad2_nb_params", type=int, default=512)
    parser.add_argument("--policy_grad2_seed", type=int, default=8848)

    args = parser.parse_args()
    if args.save_dir is None or args.save_dir.lower() == 'none':
        args.save_dir = f'./results/heat_2d/test_{now.strftime("%m-%d-%Y-%H-%M-%S")}_jac_rk4/'
    else:
        args.save_dir = "./results/" + args.save_dir

    os.makedirs(args.save_dir, exist_ok=True)
    backup_dir = os.path.join(args.save_dir, 'code_backup')
    os.makedirs(backup_dir, exist_ok=True)
    if args.load_config_from is not None and args.save_dir.lower() != 'none':
        with open(args.save_dir + '/config_overwritten.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        with open(args.load_config_from, 'r') as f:
            new_arg_dict = json.load(f)
        new_args = argparse.Namespace(**new_arg_dict)
        new_args.save_dir = args.save_dir
        new_args.load_config_from = args.load_config_from
        args = new_args
    with open(args.save_dir + '/config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for filename in os.listdir('./'):
        if '.sh' in filename or \
                '.swb' in filename or \
                '.py' in filename:
            if filename == '.pylint.d':
                continue
            if '.swp' in filename:
                continue
            if '__pycache__' in filename:
                continue
            shutil.copy(filename, backup_dir)
        shutil.copytree('./src', os.path.join(backup_dir, 'src'), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*__pycache__*'))

    return args


def write_to_file(file, *items, flush=False):
    if type(items[0]) == list or type(items[0]) == tuple:
        items = items[0]
    for item in items:
        file.write('%s ' % item)
    file.write('\n')
    if flush:
        file.flush()


def square_loss_func(u, v):
    reward = -(u - v)
    loss = reward ** 2 / 2
    return reward, loss


@partial(jax.pmap, in_axes=(None, None, 0, 0, 0), static_broadcasted_argnums=0)
def loss_func_pure(var_state_pure, state, samples, sqrt_weights, u_target):
    u = var_state_pure.evaluate(state, samples)
    reward, losses = square_loss_func(u, u_target)
    loss = (losses * sqrt_weights ** 2).sum()
    return reward, loss


def loss_func(var_state, samples, sqrt_weights, u_target):
    return loss_func_pure(var_state.pure_funcs, var_state.get_state(), samples, sqrt_weights, u_target)


# class CompareWithExact:
#     def __init__(self, points_per_dim=512, config=None):
#         self.points_per_dim = points_per_dim
#         grid = jnp.linspace(0, 2 * jnp.pi, points_per_dim, endpoint=False)
#         grid2d = jnp.stack(jnp.meshgrid(grid, grid, indexing='ij'), axis=-1).reshape(1, -1, 2)
#         self.xs = grid2d
#         if config.equation == 'heat':
#             self.exact_solution_dir = 'heat_equation_2d_spectral_fourier'
#         elif config.equation == 'alan_cahn':
#             self.exact_solution_dir = 'alan_cahn_equation_2d_spectral_fourier_new_case2'
#         elif config.equation == 'burgers':
#             self.exact_solution_dir = 'burgers_equation_2d_spectral_fourier'
#         else:
#             raise NotImplementedError
#
#     def __call__(self, var_state, T: Decimal):
#         exact_u_hat = np.load(os.path.join(self.exact_solution_dir, f'T_{T.normalize()}.npy'))
#         exact_u = self.ifft(exact_u_hat, max_N=self.points_per_dim).ravel()
#         var_state_u = var_state.evaluate(self.xs).squeeze(0)
#         abs_err = jnp.linalg.norm(exact_u - var_state_u)
#         rel_err = abs_err / jnp.linalg.norm(exact_u)
#         return abs_err.item() / self.points_per_dim * (
#                     2 * jnp.pi) ** 2, rel_err.item()  # points_per_dim is the same as sqrt(N)
#
#     def ifft(self, x_hat, max_N):
#         """Compute the inverse fourier transform of the given fourier coefficients"""
#         x_hat = jnp.fft.ifftshift(x_hat)
#         if max_N is not None:
#             max_k = x_hat.shape[0] // 2
#             new_x_hat = jnp.zeros((max_N * 2 - 2, max_N * 2 - 2), dtype=jnp.complex128)
#             new_x_hat = new_x_hat.at[:max_k + 1, :max_k + 1].set(x_hat[:max_k + 1, :max_k + 1])
#             new_x_hat = new_x_hat.at[:max_k + 1, -max_k:].set(x_hat[:max_k + 1, -max_k:])
#             new_x_hat = new_x_hat.at[-max_k:, :max_k + 1].set(x_hat[-max_k:, :max_k + 1])
#             new_x_hat = new_x_hat.at[-max_k:, -max_k:].set(x_hat[-max_k:, -max_k:])
#             x_hat = new_x_hat
#         x = jnp.fft.ifft2(x_hat, norm='forward')
#         x = x[:max_N, :max_N].real
#         return x


class CompareWithExact:
    def __init__(self, points_per_dim=513, config=None):
        self.points_per_dim = points_per_dim
        grid = jnp.linspace(-1, 1, points_per_dim, endpoint=True)
        grid_2d = jnp.stack(jnp.meshgrid(grid, grid, indexing='ij'), axis=-1).reshape(1, -1, 2)
        self.xy = grid_2d
        # only need the ones within the unit circle
        self.xy = self.xy[jnp.linalg.norm(self.xy, axis=-1) <= 1].reshape(1, -1, 2)

        self.bessel_zeros_list = [sp.special.jn_zeros(i, 5) for i in range(5)]

        self.rs = np.linspace(0, 20, 1000000)
        self.jns = [sp.special.jn(n, self.rs) for n in range(5)]

        self.D = config.D[0]

        # jax_jns = [[jnp.interp(rs, sp.special.jn(0, rs))]]

    def bessel_zeros(self, m, n):
        return self.bessel_zeros_list[m][n - 1]

    def diskharmonic(self, r, theta, m, n, T = 0.):
        # return sp.special.jn(m, bessel_zeros[m][n-1] * r) * np.cos(m * theta)
        return jnp.interp(self.bessel_zeros(m, n) * r, self.rs, self.jns[m]) * jnp.cos(m * theta) * jnp.exp(-T * self.D * self.bessel_zeros(m, n)**2)
        # return sp.special.jn(m, bessel_zeros[m][n-1] * r) * jnp.cos(m * theta)

    def solution(self, T: Decimal):
        T = float(T)
        xx = self.xy[..., 0]
        yy = self.xy[..., 1]
        r = jnp.sqrt(xx ** 2 + yy ** 2)
        theta = jnp.arctan2(yy, xx)
        return (self.diskharmonic(r, theta, 0, 1, T) - \
                self.diskharmonic(r, theta, 0, 2, T) / 4 + \
                self.diskharmonic(r, theta, 0, 3, T) / 16 - \
                self.diskharmonic(r, theta, 0, 4, T) / 64 + \
                self.diskharmonic(r, theta, 1, 1, T) - \
                self.diskharmonic(r, theta, 1, 2, T) / 2 + \
                self.diskharmonic(r, theta, 1, 3, T) / 4 - \
                self.diskharmonic(r, theta, 1, 4, T) / 8 + \
                self.diskharmonic(r, theta, 2, 1, T) + \
                self.diskharmonic(r, theta, 3, 1, T) + \
                self.diskharmonic(r, theta, 4, 1, T)) / 4

    def __call__(self, var_state, T: Decimal):
        exact_u = self.solution(T)
        var_state_u = var_state.evaluate(self.xy).squeeze(0)
        abs_err = jnp.linalg.norm(exact_u - var_state_u)
        rel_err = abs_err / jnp.linalg.norm(exact_u)
        return abs_err.item() / self.points_per_dim * (
                    2 * jnp.pi) ** 2, rel_err.item()


def euler_step(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps, pde_operator, optimizer,
               policy_grad, policy_grad2):
    var_state_old.set_state(var_state_new.get_state())
    samples, (boundaries, sqrt_weights_b), sqrt_weights = var_state_old.sampler.sample(start=0)

    samples = jnp.concatenate([samples, boundaries], axis=1)
    sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)

    final_losses = []

    # first train var_state_temp0
    stage = 0
    var_state_new.set_state(var_state_old.get_state())  # var_state_old is a better initial guess
    u_old = var_state_old.evaluate(samples)
    u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
    u_target = u_old + u_old_dot * float(dt)

    u_target = u_target.at[:, -boundaries.shape[1]:].set(0)

    for iter in range(config.nb_iters_per_step + 2):
        reward, loss = loss_func(var_state_new, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward,
                                                                    var_state=var_state_new, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_new.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_new, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    return final_losses


def heun_step(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps, pde_operator, optimizer,
              policy_grad, policy_grad2):
    var_state_temp0 = var_state_temps[0]
    var_state_old.set_state(var_state_new.get_state())
    samples, (boundaries, sqrt_weights_b), sqrt_weights = var_state_old.sampler.sample(start=0)

    samples = jnp.concatenate([samples, boundaries], axis=1)
    sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)

    final_losses = []

    # first train var_state_temp0
    stage = 0
    var_state_temp0.set_state(var_state_old.get_state())  # var_state_old is a better initial guess
    u_old = var_state_old.evaluate(samples)
    u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
    u_target = u_old + u_old_dot * float(dt)

    u_target = u_target.at[:, -boundaries.shape[1]:].set(0)

    # sqrt_weights = sqrt_weights.at[:, -boundaries.shape[1]:].set(jnp.sqrt(2 * jnp.pi / boundaries.shape[1] / 10))
    # u_target_b = jnp.zeros_like(sqrt_weights_b)
    # u_target = jnp.concatenate([u_target, u_target_b], axis=1)
    # samples = jnp.concatenate([samples, boundaries], axis=1)
    # sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)

    for iter in range(config.nb_iters_per_step + 2):
        reward, loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward,
                                                                    var_state=var_state_temp0, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp0.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_new
    # samples, (boundaries, sqrt_weights_b), sqrt_weights = var_state_old.sampler.sample(start=0)

    stage = 1
    var_state_new.set_state(var_state_temp0.get_state())  # var_state_temp0 is a better initial guess
    u_temp0 = var_state_temp0.evaluate(samples)
    u_temp0_dot = pde_operator(var_state_temp0, samples, u_temp0, compile=True)
    u_target = u_old + (u_old_dot + u_temp0_dot) * float(dt / 2)

    u_target = u_target.at[:, -boundaries.shape[1]:].set(0)
    # u_target = u_target.at[:, -boundaries.shape[1]:].set(0)
    # u_target_b = jnp.zeros_like(sqrt_weights_b)
    # u_target = jnp.concatenate([u_target, u_target_b], axis=1)
    # samples = jnp.concatenate([samples, boundaries], axis=1)
    # sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)
    for iter in range(config.nb_iters_per_step):
        reward, loss = loss_func(var_state_new, samples, sqrt_weights, u_target)
        update, info = policy_grad2(samples=samples, sqrt_weights=sqrt_weights,
                                    rewards=reward,
                                    var_state=var_state_new, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_new.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_new, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    return final_losses


def heun_step_rand(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps, pde_operator, optimizer,
              policy_grad, policy_grad2):
    assert False, 'not implemented'
    var_state_temp0 = var_state_temps[0]
    var_state_old.set_state(var_state_new.get_state())
    # samples, _, sqrt_weights = var_state_old.sampler.sample(start=None)
    final_losses = []

    # first train var_state_temp0
    stage = 0
    var_state_temp0.set_state(var_state_old.get_state())  # var_state_old is a better initial guess
    # u_old = var_state_old.evaluate(samples)
    # u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
    # u_target = u_old + u_old_dot * float(dt)
    for iter in range(config.nb_iters_per_step + 2):
        samples, _, sqrt_weights = var_state_old.sampler.sample(start=None)
        u_old = var_state_old.evaluate(samples)
        u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
        u_target = u_old + u_old_dot * float(dt)
        reward, loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward,
                                                                    var_state=var_state_temp0, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp0.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_new
    stage = 1
    var_state_new.set_state(var_state_temp0.get_state())  # var_state_temp0 is a better initial guess
    # u_temp0 = var_state_temp0.evaluate(samples)
    # u_temp0_dot = pde_operator(var_state_temp0, samples, u_temp0, compile=True)
    # u_target = u_old + (u_old_dot + u_temp0_dot) * float(dt / 2)
    for iter in range(config.nb_iters_per_step):
        samples, _, sqrt_weights = var_state_old.sampler.sample(start=None)
        u_old = var_state_old.evaluate(samples)
        u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
        u_temp0 = var_state_temp0.evaluate(samples)
        u_temp0_dot = pde_operator(var_state_temp0, samples, u_temp0, compile=True)
        u_target = u_old + (u_old_dot + u_temp0_dot) * float(dt / 2)
        reward, loss = loss_func(var_state_new, samples, sqrt_weights, u_target)
        update, info = policy_grad2(samples=samples, sqrt_weights=sqrt_weights,
                                    rewards=reward,
                                    var_state=var_state_new, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_new.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_new, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    return final_losses


def rk4_step(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps, pde_operator, optimizer,
             policy_grad, policy_grad2):
    var_state_temp0 = var_state_temps[0]
    var_state_temp1 = var_state_temps[1]
    var_state_temp2 = var_state_temps[2]
    var_state_old.set_state(var_state_new.get_state())
    samples, (boundaries, sqrt_weights_b), sqrt_weights = var_state_old.sampler.sample(start=0)
    
    samples = jnp.concatenate([samples, boundaries], axis=1)
    sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)
    final_losses = []

    # first train var_state_temp0
    stage = 0
    var_state_temp0.set_state(var_state_old.get_state())  # var_state_old is a better initial guess
    u_old = var_state_old.evaluate(samples)
    u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
    u_target = u_old + u_old_dot * float(dt / 2)
    u_target = u_target.at[:, -boundaries.shape[1]:].set(0)
    # u_target_b = jnp.zeros_like(sqrt_weights_b)
    # u_target = jnp.concatenate([u_target, u_target_b], axis=1)
    # samples = jnp.concatenate([samples, boundaries], axis=1)
    # sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)
    for iter in range(config.nb_iters_per_step + 2):
        reward, loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward,
                                                                    var_state=var_state_temp0, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp0.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_temp1
    # samples, (boundaries, sqrt_weights_b), sqrt_weights = var_state_old.sampler.sample(angles=0)
    stage = 1
    var_state_temp1.set_state(var_state_temp0.get_state())  # var_state_temp0 is a better initial guess
    u_temp0 = var_state_temp0.evaluate(samples)
    u_temp0_dot = pde_operator(var_state_temp0, samples, u_temp0, compile=True)
    u_target = u_old + u_temp0_dot * float(dt / 2)
    u_target = u_target.at[:, -boundaries.shape[1]:].set(0)
    # u_target_b = jnp.zeros_like(sqrt_weights_b)
    # u_target = jnp.concatenate([u_target, u_target_b], axis=1)
    # samples = jnp.concatenate([samples, boundaries], axis=1)
    # sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)
    for iter in range(config.nb_iters_per_step):
        reward, loss = loss_func(var_state_temp1, samples, sqrt_weights, u_target)
        update, info = policy_grad2(samples=samples, sqrt_weights=sqrt_weights,
                                    rewards=reward,
                                    var_state=var_state_temp1, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp1.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp1, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_temp2
    # samples, (boundaries, sqrt_weights_b), sqrt_weights = var_state_old.sampler.sample(angles=0)
    stage = 2
    var_state_temp2.set_state(var_state_temp1.get_state())  # var_state_temp1 is a better initial guess
    u_temp1 = var_state_temp1.evaluate(samples)
    u_temp1_dot = pde_operator(var_state_temp1, samples, u_temp1, compile=True)
    u_target = u_old + u_temp1_dot * float(dt)
    u_target = u_target.at[:, -boundaries.shape[1]:].set(0)
    # u_target_b = jnp.zeros_like(sqrt_weights_b)
    # u_target = jnp.concatenate([u_target, u_target_b], axis=1)
    # samples = jnp.concatenate([samples, boundaries], axis=1)
    # sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)
    for iter in range(config.nb_iters_per_step + 2):
        reward, loss = loss_func(var_state_temp2, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward,
                                                                    var_state=var_state_temp2, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp2.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp2, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_new
    # samples, (boundaries, sqrt_weights_b), sqrt_weights = var_state_old.sampler.sample(angles=0)
    stage = 3
    var_state_new.set_state(var_state_temp2.get_state())  # var_state_temp2 is a better initial guess
    u_temp2 = var_state_temp2.evaluate(samples)
    u_temp2_dot = pde_operator(var_state_temp2, samples, u_temp2, compile=True)
    u_target = u_old + (u_old_dot + 2 * u_temp0_dot + 2 * u_temp1_dot + u_temp2_dot) * float(dt / 6)
    u_target = u_target.at[:, -boundaries.shape[1]:].set(0)
    # u_target_b = jnp.zeros_like(sqrt_weights_b)
    # u_target = jnp.concatenate([u_target, u_target_b], axis=1)
    # samples = jnp.concatenate([samples, boundaries], axis=1)
    # sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], axis=1)
    for iter in range(config.nb_iters_per_step):
        reward, loss = loss_func(var_state_new, samples, sqrt_weights, u_target)
        update, info = policy_grad2(samples=samples, sqrt_weights=sqrt_weights,
                                    rewards=reward,
                                    var_state=var_state_new, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_new.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_new, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    return final_losses


def rk4_step_rand(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps, pde_operator, optimizer,
             policy_grad, policy_grad2):
    assert False, 'not implemented'
    var_state_temp0 = var_state_temps[0]
    var_state_temp1 = var_state_temps[1]
    var_state_temp2 = var_state_temps[2]
    var_state_old.set_state(var_state_new.get_state())
    # samples, _, sqrt_weights = var_state_old.sampler.sample(start=0)
    final_losses = []

    # first train var_state_temp0
    stage = 0
    var_state_temp0.set_state(var_state_old.get_state())  # var_state_old is a better initial guess
    # u_old = var_state_old.evaluate(samples)
    # u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
    # u_target = u_old + u_old_dot * float(dt / 2)
    for iter in range(config.nb_iters_per_step + 2):
        samples, _, sqrt_weights = var_state_old.sampler.sample(start=None)
        u_old = var_state_old.evaluate(samples)
        u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
        u_target = u_old + u_old_dot * float(dt / 2)
        reward, loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward,
                                                                    var_state=var_state_temp0, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp0.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp0, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_temp1
    stage = 1
    var_state_temp1.set_state(var_state_temp0.get_state())  # var_state_temp0 is a better initial guess
    # u_temp0 = var_state_temp0.evaluate(samples)
    # u_temp0_dot = pde_operator(var_state_temp0, samples, u_temp0, compile=True)
    # u_target = u_old + u_temp0_dot * float(dt / 2)
    for iter in range(config.nb_iters_per_step):
        samples, _, sqrt_weights = var_state_old.sampler.sample(start=None)
        u_old = var_state_old.evaluate(samples)
        u_temp0 = var_state_temp0.evaluate(samples)
        u_temp0_dot = pde_operator(var_state_temp0, samples, u_temp0, compile=True)
        u_target = u_old + u_temp0_dot * float(dt / 2)
        reward, loss = loss_func(var_state_temp1, samples, sqrt_weights, u_target)
        update, info = policy_grad2(samples=samples, sqrt_weights=sqrt_weights,
                                    rewards=reward,
                                    var_state=var_state_temp1, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp1.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp1, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_temp2
    stage = 2
    var_state_temp2.set_state(var_state_temp1.get_state())  # var_state_temp1 is a better initial guess
    # u_temp1 = var_state_temp1.evaluate(samples)
    # u_temp1_dot = pde_operator(var_state_temp1, samples, u_temp1, compile=True)
    # u_target = u_old + u_temp1_dot * float(dt)
    for iter in range(config.nb_iters_per_step + 2):
        samples, _, sqrt_weights = var_state_old.sampler.sample(start=None)
        u_old = var_state_old.evaluate(samples)
        u_temp1 = var_state_temp1.evaluate(samples)
        u_temp1_dot = pde_operator(var_state_temp1, samples, u_temp1, compile=True)
        u_target = u_old + u_temp1_dot * float(dt)
        reward, loss = loss_func(var_state_temp2, samples, sqrt_weights, u_target)
        update, info = (policy_grad if iter == 0 else policy_grad2)(samples=samples, sqrt_weights=sqrt_weights,
                                                                    rewards=reward,
                                                                    var_state=var_state_temp2, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_temp2.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_temp2, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    # then train var_state_new
    stage = 3
    var_state_new.set_state(var_state_temp2.get_state())  # var_state_temp2 is a better initial guess
    # u_temp2 = var_state_temp2.evaluate(samples)
    # u_temp2_dot = pde_operator(var_state_temp2, samples, u_temp2, compile=True)
    # u_target = u_old + (u_old_dot + 2 * u_temp0_dot + 2 * u_temp1_dot + u_temp2_dot) * float(dt / 6)
    for iter in range(config.nb_iters_per_step):
        samples, _, sqrt_weights = var_state_old.sampler.sample(start=None)
        u_old = var_state_old.evaluate(samples)
        u_old_dot = pde_operator(var_state_old, samples, u_old, compile=True)
        u_temp0 = var_state_temp0.evaluate(samples)
        u_temp0_dot = pde_operator(var_state_temp0, samples, u_temp0, compile=True)
        u_temp1 = var_state_temp1.evaluate(samples)
        u_temp1_dot = pde_operator(var_state_temp1, samples, u_temp1, compile=True)
        u_temp2 = var_state_temp2.evaluate(samples)
        u_temp2_dot = pde_operator(var_state_temp2, samples, u_temp2, compile=True)
        u_target = u_old + (u_old_dot + 2 * u_temp0_dot + 2 * u_temp1_dot + u_temp2_dot) * float(dt / 6)
        reward, loss = loss_func(var_state_new, samples, sqrt_weights, u_target)
        update, info = policy_grad2(samples=samples, sqrt_weights=sqrt_weights,
                                    rewards=reward,
                                    var_state=var_state_new, resample_params=True)
        info = tuple(
            each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
        var_state_new.update_parameters(update)
        loss = loss.squeeze().item()
        logging.info(f'{step=}, {T=}, {stage=}, {iter=}, {loss=}, {info=}')
        write_to_file(fiters, T, step, stage, iter, loss, *info, flush=False)
    loss = loss_func(var_state_new, samples, sqrt_weights, u_target)[1].squeeze().item()
    logging.info(f'{step=}, {T=}, {stage=}, {loss=}')
    final_losses.append(loss)

    return final_losses


def save_states(config, var_state_new, var_state_old, var_state_temps, step):
    var_state_new.save_state(os.path.join(config.save_dir, f'var_state_new_{step}.pickle'))
    var_state_old.save_state(os.path.join(config.save_dir, f'var_state_old_{step}.pickle'))
    for i, var_state_temp in enumerate(var_state_temps):
        var_state_temp.save_state(os.path.join(config.save_dir, f'var_state_temp_{i}_{step}.pickle'))


def train(config, var_state_new, var_state_old, var_state_temps, pde_operator, optimizer, policy_grad, policy_grad2):
    error_from_exact = CompareWithExact(config=config)
    if config.integrator == 'euler':
        stepper = euler_step
    elif config.integrator == 'heun':
        if config.random_sample:
            stepper = heun_step_rand
        else:
            stepper = heun_step
    elif config.integrator == 'rk4':
        if config.random_sample:
            stepper = rk4_step_rand
        else:
            stepper = rk4_step
    else:
        raise ValueError(f'Unknown integrator {config.integrator}')
    training_time = 0
    with open(os.path.join(config.save_dir, f'iters.txt'), 'w') as fiters, \
            open(os.path.join(config.save_dir, f'steps.txt'), 'w') as fsteps:
        T = Decimal('0')
        dt = Decimal(config.dt)
        err = error_from_exact(var_state_new, T)
        # err = (np.nan,)
        logging.info(f'step={-1}, {T=}, {err=}, loss={(0, 0)}, {training_time=}')
        write_to_file(fsteps, -1, T, *err, 0, 0, flush=True)
        for step in range(config.nb_steps):
            T += dt
            start_time = time.perf_counter()
            loss = stepper(config, fiters, T, step, dt, var_state_new, var_state_old, var_state_temps,
                           pde_operator, optimizer, policy_grad, policy_grad2)
            training_time += time.perf_counter() - start_time
            fiters.flush()
            err = error_from_exact(var_state_new, T)
            logging.info(f'{step=}, {T=}, {err=}, {loss=}, {training_time=}')
            write_to_file(fsteps, step, T, *err, *loss, flush=True)
            save_states(config, var_state_new, var_state_old, var_state_temps, step)
    return config, var_state_new, var_state_old, var_state_temps, pde_operator, optimizer, policy_grad, policy_grad2


def main():
    # get config
    config = get_config()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    file_handler = logging.FileHandler(os.path.join(config.save_dir, "log.txt"), mode="w")
    logger.addHandler(file_handler)
    start_time = time.perf_counter()

    # net = DirichletPDENet2(width=40, depth=7, minvals=(0., 0.), maxvals=(jnp.pi, jnp.pi))
    # net = CircularDirichletPDENet2(width=40, depth=7, radius=1.)
    net = PDENet(width=40, depth=7)

    # the var_states can share the same sampler in this case
    # sampler = PeriodicQuadratureSampler(nb_sites=config.nb_dims,
    #                                     nb_samples=config.nb_samples,
    #                                     minvals=0., maxvals=jnp.pi * 2, quad_rule='trapezoid',
    #                                     rand_seed=config.sampler_seed)
    # sampler = OpenQuadratureSampler(nb_sites=config.nb_dims,
    #                                 nb_samples=config.nb_samples,
    #                                 minvals=0., maxvals=jnp.pi,
    #                                 rand_seed=config.sampler_seed)
    sampler = CircularQuadratureSampler2(nb_sites=config.nb_dims,
                                         nb_samples=config.nb_samples,
                                         radius=1.,
                                         rand_seed=config.sampler_seed)

    # load sampler state is needed
    if config.load_sampler_state_from is not None and config.load_sampler_state_from.lower() != 'none':
        sampler.load_state(config.load_sampler_state_from)

    # define the var_state
    # we need to define multiple copies of the var_state for the intermediate results of heun's method or rk4 method
    # the net can be shared because it is just a pure function, which will not cause any issue

    var_state_new = SimpleVarStateReal(net=net, system_shape=(config.nb_dims,), sampler=sampler,
                                       init_seed=config.model_seed)
    var_state_old = SimpleVarStateReal(net=net, system_shape=(config.nb_dims,), sampler=sampler,
                                       init_seed=config.model_seed)
    # temporary var_states for storing the intermediate results of heun's method or rk4 method
    var_state_temps = []
    if config.integrator == 'heun':
        for _ in range(1):
            var_state_temps.append(SimpleVarStateReal(net=net, system_shape=(config.nb_dims,), sampler=sampler,
                                                      init_seed=config.model_seed))
    elif config.integrator == 'rk4':
        for _ in range(3):
            var_state_temps.append(SimpleVarStateReal(net=net, system_shape=(config.nb_dims,), sampler=sampler,
                                                      init_seed=config.model_seed))

    # load model state if needed
    if config.load_model_state_from is not None and config.load_model_state_from.lower() != 'none':
        # we will only load the state to the new var_state, and the old var_state will be updated by the new var_state
        var_state_new.load_state(config.load_model_state_from)

    # define the operator of the pde
    # first parse the input
    if len(config.mu) == 1:
        drift_coefs = jnp.ones(config.nb_dims) * config.mu[0]
    elif len(config.mu) == config.nb_dims:
        drift_coefs = jnp.array(config.mu)
    else:
        assert False, f'mu can take either 1 argument or {config.nb_dims=} arguments, but got {len(config.mu)=} arguments'
    if len(config.D) == 1:
        diffusion_coefs = jnp.diag(jnp.ones(config.nb_dims) * config.D[0])
    elif len(config.D) == config.nb_dims:
        diffusion_coefs = jnp.diag(jnp.array(config.D))
    elif len(config.D) == config.nb_dims ** 2:
        diffusion_coefs = jnp.array(config.D).reshape(config.nb_dims, config.nb_dims)
    else:
        assert False, f'D can take either 1 argument or {config.nb_dims=} arguments or {config.nb_dims**2=} arguments, but got {len(config.mu)=} arguments'

    # then define the operator
    if config.equation == 'heat':
        drift_coefs = jnp.zeros(config.nb_dims)
        pde_operator = HeatOperatorNoLog(config.nb_dims, drift_coefs, diffusion_coefs, check_validity=True)
    elif config.equation == 'alan_cahn':
        pde_operator = AlanCahnOperator(config.nb_dims, diffusion_coefs, check_validity=True)
    elif config.equation == 'burgers':
        pde_operator = BurgersOperator(config.nb_dims, diffusion_coefs, check_validity=True)
    else:
        assert False, f'Unknown equation: {config.equation}'

    # define policy grad function
    policy_grad = RandomNaturalPolicyGradTDVP2(var_state=var_state_new, ls_solver=None,
                                              nb_params_to_take=config.policy_grad_nb_params,
                                              rand_seed=config.policy_grad_seed, rcond=1e-7)
    policy_grad2 = RandomNaturalPolicyGradTDVP2(var_state=var_state_new, ls_solver=None,
                                               nb_params_to_take=config.policy_grad2_nb_params,
                                               rand_seed=config.policy_grad2_seed, rcond=1e-7)

    config, var_state_new, var_state_old, var_state_temps, pde_operator, optimizer, policy_grad, policy_grad2 = \
        train(config, var_state_new, var_state_old, var_state_temps, pde_operator, None, policy_grad, policy_grad2)

    end_time = time.perf_counter()
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")

    return


if __name__ == '__main__':
    main()

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
from math import prod
from collections import namedtuple

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

# there may be an object oriented way to do this, but we will use a functional approach for now
def gen_periodic_quadrature_sample_funcs(nb_sites, nb_samples, minvals, maxvals):
    nb_points_each_site = (round(nb_samples ** (1 / nb_sites)), ) * nb_sites
    nb_points_total = prod(nb_points_each_site)
    assert nb_points_total == nb_samples, "nb_samples must be a perfect power of nb_sites"
    ranges = maxvals - minvals
    def gen_grid():
        grid_each_site = [jnp.linspace(minvals[i], maxvals[i], nb_points_each_site[i], endpoint=False) for i in range(nb_sites)]
        grid_points = jnp.stack(jnp.meshgrid(*grid_each_site, indexing='ij'), axis=-1)
        return grid_points.reshape((1, nb_points_total, nb_sites))
    grid_points = gen_grid()

    @jax.jit
    def sample(rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        starts = jrnd.uniform(sub_rand_key, (nb_sites,), minval=0, maxval=ranges)
        samples = ((starts + grid_points) % ranges + minvals)
        return samples, rand_key

    @jax.jit
    def sample_deterministic(starts):
        samples = ((starts + grid_points) % ranges + minvals)
        return samples

    @jax.jit
    def sample_grid():
        return grid_points

    return sample, sample_deterministic, sample_grid

# @dataclass(frozen=True) # to make it immutable and hashable
# class PeriodicQuadratureSamplerPure:
#     nb_sites: int
#     nb_points_each_site: Tuple
#     nb_points_total: int
#     minvals: Tuple
#     maxvals: Tuple
#     ranges: Tuple
#     grid_points: Tuple
#     sqrt_weights: float
#
#     def __init__(self, nb_sites, nb_points_each_site, minvals, maxvals):
#         object.__setattr__(self, "nb_sites", nb_sites)
#         object.__setattr__(self, "nb_points_each_site", nb_points_each_site)
#         object.__setattr__(self, "nb_points_total", prod(nb_points_each_site))
#         object.__setattr__(self, "minvals", minvals)
#         object.__setattr__(self, "maxvals", maxvals)
#         object.__setattr__(self, "ranges", maxvals - minvals)
#         grid_points = []
#         sqrt_weights = []
#         for i in range(nb_sites):
#             # generate nd grid points
#
#
#     @partial(jax.jit, static_argnums=(0, 1))
#     def sample(self, nb_points_per_site, minvals, maxvals, rand_key):
#         ranges = maxvals - minvals
#         rand_key, sub_rand_key = jrnd.split(rand_key)
#         starts = jrnd.uniform(sub_rand_key, (self.nb_sites,), minval=0, maxval=ranges)
#         # sample grid points
#         samples = (jnp.linspace(starts, starts + ranges, nb_points_per_site, endpoint=False) % ranges + minvals).reshape(1, nb_points_per_site, self.nb_sites)
#         return samples, rand_key
#
#     @partial(jax.jit, static_argnums=(0, 1))
#     def sample_deterministic(self, nb_points_per_site, minvals, maxvals, starts):
#         ranges = maxvals - minvals
#         samples = (jnp.linspace(starts, starts + ranges, nb_points_per_site, endpoint=False) % ranges + minvals).reshape(1, nb_points_per_site, self.nb_sites)
#         return samples


class PeriodicQuadratureSampler(AbstractSampler):
    """
    samples at fixed intervals with appropriate quadrature weights for periodic boundary conditions
    the starting point is uniformly sampled
    does not support multiple devices now
    """
    def __init__(self, nb_sites, nb_samples, minvals, maxvals, quad_rule=None, rand_seed=1234):
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
        # self.pure_funcs = PeriodicQuadratureSamplerPure(self.nb_sites)
        self.pure_funcs = namedtuple('PeriodicQuadratureSamplerPure', ['sample', 'sample_deterministic', 'sample_grid'])\
            (*gen_periodic_quadrature_sample_funcs(self.nb_sites, nb_samples, self.minvals, self.maxvals))
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        assert self.nb_devices == 1, f"{self.nb_devices=} must be 1"
        self.nb_samples = nb_samples
        if quad_rule is None:
            quad_rule = 'midpoint'
        quad_rule = quad_rule.lower()
        if quad_rule == 'midpoint' or quad_rule == 'trapezoidal' or quad_rule == 'mid' or quad_rule == 'trap' or quad_rule == 'trapezoid' or quad_rule == 'trapozoidal' or quad_rule == 'trapozoid':
            sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.area / nb_samples)
        elif quad_rule == 'simpson' or quad_rule == 'simpsons':
            assert nb_samples % 2 == 0, f"{nb_samples=} must be even for (periodic) simpson's rule"
            sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.area / (1.5 * nb_samples))
            sqrt_weights = sqrt_weights.at[::2].multiply(jnp.sqrt(2))
        elif type(quad_rule) == str:
            raise NotImplementedError(f"{quad_rule=} is not implemented")
        elif type(quad_rule) == jnp.ndarray:
            sqrt_weights = quad_rule
        else:
            raise ValueError(f"{quad_rule=} is not a valid value, must be a string or a jnp.ndarray, or None")
        self.sqrt_weights = sqrt_weights.reshape(1, nb_samples)

        self.rand_key = jax.random.PRNGKey(rand_seed)


    def set_var_state(self, var_state):
        # we don't need this function in this case
        pass

    # def sample(self, start=None):
    #     if start is None:
    #         samples, rand_key = self.pure_funcs.sample(self.nb_samples, self.minvals, self.maxvals, self.rand_key)
    #     else:
    #         samples = self.pure_funcs.sample_deterministic(self.nb_samples, self.minvals, self.maxvals, start)
    #         rand_key = self.rand_key
    #     self.rand_key = rand_key
    #     return samples, None, self.sqrt_weights

    def sample(self, start=None):
        if start is None:
            samples, rand_key = self.pure_funcs.sample(self.rand_key)
            self.rand_key = rand_key
        elif start == 0:
            samples = self.pure_funcs.sample_grid()
        else:
            samples = self.pure_funcs.sample_deterministic(start)
        return samples, None, self.sqrt_weights

    def get_state(self):
        return {'rand_key': self.rand_key}

    def set_state(self, state):
        self.rand_key = state['rand_key']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))


def gen_open_quadrature_sample_funcs(nb_sites, nb_samples, minvals, maxvals):
    nb_points_each_site = (round(nb_samples ** (1 / nb_sites)), ) * nb_sites
    nb_points_total = prod(nb_points_each_site)
    assert nb_points_total == nb_samples, "nb_samples must be a perfect power of nb_sites"
    ranges = maxvals - minvals
    def gen_grid():
        grid_each_site = [jnp.linspace(minvals[i], maxvals[i], nb_points_each_site[i], endpoint=True) for i in range(nb_sites)]
        grid_points = jnp.stack(jnp.meshgrid(*grid_each_site, indexing='ij'), axis=-1)
        grid_each_site_no_end = [jnp.linspace(minvals[i], maxvals[i], nb_points_each_site[i], endpoint=False) for i in range(nb_sites)]
        grid_points_no_end = jnp.stack(jnp.meshgrid(*grid_each_site_no_end, indexing='ij'), axis=-1)
        return grid_points.reshape((1, nb_points_total, nb_sites)), grid_points_no_end.reshape((1, nb_points_total, nb_sites))
    grid_points, grid_points_no_end = gen_grid()

    @jax.jit
    def sample(rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        starts = jrnd.uniform(sub_rand_key, (nb_sites,), minval=0, maxval=ranges)
        samples = ((starts + grid_points_no_end) % ranges + minvals) # ignore the last point
        return samples, rand_key
        # return grid_points

    @jax.jit
    def sample_deterministic(starts):
        samples = ((starts + grid_points_no_end) % ranges + minvals)
        return samples
        # return grid_points

    @jax.jit
    def sample_grid():
        return grid_points

    return sample, sample_deterministic, sample_grid
class OpenQuadratureSampler(AbstractSampler):
    """
    samples at fixed intervals with appropriate quadrature weights for periodic boundary conditions
    the starting point is uniformly sampled
    does not support multiple devices now
    """
    def __init__(self, nb_sites, nb_samples, minvals, maxvals, quad_rule=None, rand_seed=1234):
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
        # self.pure_funcs = PeriodicQuadratureSamplerPure(self.nb_sites)
        self.pure_funcs = namedtuple('OpenQuadratureSamplerPure', ['sample', 'sample_deterministic', 'sample_grid'])\
            (*gen_open_quadrature_sample_funcs(self.nb_sites, nb_samples, self.minvals, self.maxvals))
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        assert self.nb_devices == 1, f"{self.nb_devices=} must be 1"
        self.nb_samples = nb_samples
        assert quad_rule is None, f"{quad_rule=} must be None for open quadrature sampler, will automatically pick the simplist ones"
        # if quad_rule is None:
        #     quad_rule = 'trapezoidal'
        # quad_rule = quad_rule.lower()
        # if quad_rule == 'trapezoidal' or quad_rule == 'trap' or quad_rule == 'trapezoid' or quad_rule == 'trapozoidal' or quad_rule == 'trapozoid':
        sqrt_weights_trap = jnp.ones(nb_samples) * jnp.sqrt(self.area / (nb_samples - 1))
        sqrt_weights_trap = sqrt_weights_trap.at[0].divide(jnp.sqrt(2))
        sqrt_weights_trap = sqrt_weights_trap.at[-1].divide(jnp.sqrt(2))
        sqrt_weights_mid = jnp.ones(nb_samples) * jnp.sqrt(self.area / nb_samples)
        # elif quad_rule == 'simpson' or quad_rule == 'simpsons':
        #     assert nb_samples % 2 == 0, f"{nb_samples=} must be even for (periodic) simpson's rule"
        #     sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.area / (1.5 * nb_samples))
        #     sqrt_weights.at[::2].multiply(jnp.sqrt(2))
        # elif type(quad_rule) == str:
        #     raise NotImplementedError(f"{quad_rule=} is not implemented")
        # elif type(quad_rule) == jnp.ndarray:
        #     sqrt_weights = quad_rule
        # else:
        #     raise ValueError(f"{quad_rule=} is not a valid value, must be a string or a jnp.ndarray, or None")
        self.sqrt_weights_trap = sqrt_weights_trap.reshape(1, nb_samples)
        self.sqrt_weights_mid = sqrt_weights_mid.reshape(1, nb_samples)

        self.rand_key = jax.random.PRNGKey(rand_seed)


    def set_var_state(self, var_state):
        # we don't need this function in this case
        pass

    # def sample(self, start=None):
    #     if start is None:
    #         samples, rand_key = self.pure_funcs.sample(self.nb_samples, self.minvals, self.maxvals, self.rand_key)
    #     else:
    #         samples = self.pure_funcs.sample_deterministic(self.nb_samples, self.minvals, self.maxvals, start)
    #         rand_key = self.rand_key
    #     self.rand_key = rand_key
    #     return samples, None, self.sqrt_weights

    def sample(self, start=None):
        if start is None:
            samples, rand_key = self.pure_funcs.sample(self.rand_key)
            self.rand_key = rand_key
            return samples, None, self.sqrt_weights_mid
        elif start == 0:
            samples = self.pure_funcs.sample_grid()
            return samples, None, self.sqrt_weights_trap
        else:
            samples = self.pure_funcs.sample_deterministic(start)
            return samples, None, self.sqrt_weights_mid

    def get_state(self):
        return {'rand_key': self.rand_key}

    def set_state(self, state):
        self.rand_key = state['rand_key']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))


def gen_circular_quadrature_sample_funcs(nb_sites, nb_samples, radius):
    assert nb_sites == 2, f"{nb_sites=} must be 2 for circular quadrature sampler"
    nb_points_each_site = (round(nb_samples ** (1 / nb_sites)), ) * nb_sites
    nb_points_total = prod(nb_points_each_site)
    assert nb_points_total == nb_samples, "nb_samples must be a perfect power of nb_sites"
    def gen_grid():
        interval = radius / (nb_points_each_site[0])
        grid_point_r = jnp.linspace(interval/2, radius-interval/2, nb_points_each_site[0], endpoint=True)
        grid_point_theta = jnp.linspace(0, 2 * jnp.pi, nb_points_each_site[1], endpoint=False)
        grid_points = jnp.stack(jnp.meshgrid(grid_point_r, grid_point_theta, indexing='ij'), axis=-1)
        boundary_points = jnp.stack(jnp.meshgrid(jnp.ones(1) * radius, grid_point_theta, indexing='ij'), axis=-1)
        return grid_points.reshape((1, nb_points_total, nb_sites)), boundary_points.reshape((1, nb_points_each_site[1], nb_sites))
    grid_points, boundary_points = gen_grid()

    @jax.jit
    def sample(rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        sub_rand_key1, sub_rand_key2 = jrnd.split(sub_rand_key)
        new_grid_points = grid_points.copy()
        new_boundary_points = boundary_points.copy()
        new_grid_points = new_grid_points.at[..., 1].add(jrnd.uniform(sub_rand_key1, (1,), minval=0, maxval=2 * jnp.pi))
        new_boundary_points = new_boundary_points.at[..., 1].add(jrnd.uniform(sub_rand_key2, (1,), minval=0, maxval=2 * jnp.pi))
        samples = jnp.stack([new_grid_points[..., 0] * jnp.cos(new_grid_points[..., 1]), new_grid_points[..., 0] * jnp.sin(new_grid_points[..., 1])], axis=-1)
        boundaries = jnp.stack([new_boundary_points[..., 0] * jnp.cos(new_boundary_points[..., 1]), new_boundary_points[..., 0] * jnp.sin(new_boundary_points[..., 1])], axis=-1)
        return samples, boundaries, rand_key
        # return grid_points

    @jax.jit
    def sample_deterministic(angles, angles_b):
        new_grid_points = grid_points.copy()
        new_grid_points = new_grid_points.at[..., 1].add(angles)
        new_boundary_points = boundary_points.copy()
        new_boundary_points = new_boundary_points.at[..., 1].add(angles_b)
        samples = jnp.stack([new_grid_points[..., 0] * jnp.cos(new_grid_points[..., 1]), new_grid_points[..., 0] * jnp.sin(new_grid_points[..., 1])], axis=-1)
        boundaries = jnp.stack([new_boundary_points[..., 0] * jnp.cos(new_boundary_points[..., 1]), new_boundary_points[..., 0] * jnp.sin(new_boundary_points[..., 1])], axis=-1)
        return samples, boundaries
        # return grid_points

    @jax.jit
    def sample_grid():
        new_grid_points = grid_points.copy()
        new_boundary_points = boundary_points.copy()
        samples = jnp.stack([new_grid_points[..., 0] * jnp.cos(new_grid_points[..., 1]), new_grid_points[..., 0] * jnp.sin(new_grid_points[..., 1])], axis=-1)
        boundaries = jnp.stack([new_boundary_points[..., 0] * jnp.cos(new_boundary_points[..., 1]), new_boundary_points[..., 0] * jnp.sin(new_boundary_points[..., 1])], axis=-1)
        return samples, boundaries
        # return grid_points

    return sample, sample_deterministic, sample_grid
class CircularQuadratureSampler(AbstractSampler):
    """
    samples at fixed intervals with appropriate quadrature weights for periodic boundary conditions
    the starting point is uniformly sampled
    does not support multiple devices now
    """
    def __init__(self, nb_sites, nb_samples, radius=1, quad_rule=None, rand_seed=1234):
        super().__init__()
        self.nb_sites = nb_sites
        self.radius = radius
        self.area = jnp.pi * radius ** 2
        # self.pure_funcs = PeriodicQuadratureSamplerPure(self.nb_sites)
        self.pure_funcs = namedtuple('CircularQuadratureSamplerPure', ['sample', 'sample_deterministic', 'sample_grid'])\
            (*gen_circular_quadrature_sample_funcs(self.nb_sites, nb_samples, self.radius))
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        assert self.nb_devices == 1, f"{self.nb_devices=} must be 1"
        self.nb_samples = nb_samples
        assert quad_rule is None, f"{quad_rule=} must be None for open quadrature sampler, will automatically pick the simplist ones"

        self.rand_key = jax.random.PRNGKey(rand_seed)


    def set_var_state(self, var_state):
        # we don't need this function in this case
        pass

    # def sample(self, start=None):
    #     if start is None:
    #         samples, rand_key = self.pure_funcs.sample(self.nb_samples, self.minvals, self.maxvals, self.rand_key)
    #     else:
    #         samples = self.pure_funcs.sample_deterministic(self.nb_samples, self.minvals, self.maxvals, start)
    #         rand_key = self.rand_key
    #     self.rand_key = rand_key
    #     return samples, None, self.sqrt_weights

    def sample(self, angles=None):
        if angles is None:
            samples, boundaries, rand_key = self.pure_funcs.sample(self.rand_key)
            # weights should depend on radius
            radii = jnp.linalg.norm(samples, axis=-1)
            sqrt_weights =  jnp.sqrt(radii * 2 * jnp.pi * self.radius / self.nb_samples)
            self.rand_key = rand_key
            return samples, (boundaries, jnp.ones((1, boundaries.shape[1])) * jnp.sqrt(2 * jnp.pi * self.radius / jnp.sqrt(self.nb_samples))), sqrt_weights
        elif angles == 0:
            samples, boundaries= self.pure_funcs.sample_grid()
            radii = jnp.linalg.norm(samples, axis=-1)
            sqrt_weights =  jnp.sqrt(radii * 2 * jnp.pi * self.radius / self.nb_samples)
            return samples, (boundaries, jnp.ones((1, boundaries.shape[1])) * jnp.sqrt(2 * jnp.pi * self.radius / jnp.sqrt(self.nb_samples))), sqrt_weights
        else:
            samples, boundaries = self.pure_funcs.sample_deterministic(angles)
            radii = jnp.linalg.norm(samples, axis=-1)
            sqrt_weights =  jnp.sqrt(radii * 2 * jnp.pi * self.radius / self.nb_samples)
            return samples, (boundaries, jnp.ones((1, boundaries.shape[1])) * jnp.sqrt(2 * jnp.pi * self.radius / jnp.sqrt(self.nb_samples))), sqrt_weights

    def get_state(self):
        return {'rand_key': self.rand_key}

    def set_state(self, state):
        self.rand_key = state['rand_key']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))


def gen_circular_quadrature_sample_funcs2(nb_sites, nb_samples, radius):
    assert nb_sites == 2, f"{nb_sites=} must be 2 for circular quadrature sampler"
    nb_points_each_site = (round(nb_samples ** (1 / nb_sites)),) * nb_sites
    nb_points_total = prod(nb_points_each_site)
    assert nb_points_total == nb_samples, "nb_samples must be a perfect power of nb_sites"
    minvals = jnp.array([-radius, -radius])
    maxvals = jnp.array([radius, radius])
    ranges = maxvals - minvals

    def gen_grid():
        grid_each_site = [jnp.linspace(minvals[i], maxvals[i], nb_points_each_site[i], endpoint=False) for i in
                                 range(nb_sites)]
        grid_points = jnp.stack(jnp.meshgrid(*grid_each_site, indexing='ij'), axis=-1)
        return grid_points.reshape((1, nb_points_total, nb_sites)) + ranges[0] / nb_points_each_site[0] / 2

    grid_points = gen_grid()
    grid_points_within = grid_points[jnp.linalg.norm(grid_points, axis=-1) <= radius].reshape(1, -1, nb_sites)
    boundary_points = jnp.stack(jnp.meshgrid(jnp.ones(1) * radius,
                                             jnp.linspace(0, 2 * jnp.pi, nb_points_each_site[1], endpoint=False),
                                             indexing='ij'), axis=-1).reshape(1, -1, nb_sites)
    boundary_points = jnp.stack((boundary_points[..., 0] * jnp.cos(boundary_points[..., 1]), boundary_points[..., 0] * jnp.sin(boundary_points[..., 1])), axis=-1)

    @jax.jit
    def sample(rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        sub_rand_key1, sub_rand_key2, sub_rand_key3 = jrnd.split(sub_rand_key, 3)
        starts = jrnd.uniform(sub_rand_key1, (nb_sites,), minval=0, maxval=ranges)
        samples = ((starts + grid_points) % ranges + minvals)  # ignore the last point
        # subselect samples within the circle
        radii = jnp.linalg.norm(samples, axis=-1)
        samples = samples[radii <= radius].reshape(1, -1, nb_sites)
        # randomly rotate the samples
        angles = jrnd.uniform(sub_rand_key2, (1,), minval=0, maxval=2 * jnp.pi)
        rot_mat = jnp.array([[jnp.cos(angles), -jnp.sin(angles)], [jnp.sin(angles), jnp.cos(angles)]])
        samples = jnp.tensordot(samples, rot_mat, axes=(-1, -2))
        angles = jrnd.uniform(sub_rand_key3, (1,), minval=0, maxval=2 * jnp.pi)
        rot_mat = jnp.array([[jnp.cos(angles), -jnp.sin(angles)], [jnp.sin(angles), jnp.cos(angles)]])
        samples_b = jnp.tensordot(boundary_points, rot_mat, axes=(-1, -2))
        return samples, samples_b, rand_key
        # return grid_points

    @jax.jit
    def sample_deterministic(starts, angles=0., angles_b=0.):
        samples = ((starts + grid_points) % ranges + minvals)
        # subselect samples within the circle
        radii = jnp.linalg.norm(samples, axis=-1)
        samples = samples[radii <= radius].reshape(1, -1, nb_sites)
        # rotate the samples
        rot_mat = jnp.array([[jnp.cos(angles), -jnp.sin(angles)], [jnp.sin(angles), jnp.cos(angles)]])
        samples = jnp.tensordot(samples, rot_mat, axes=(-1, -2))
        rot_mat = jnp.array([[jnp.cos(angles_b), -jnp.sin(angles_b)], [jnp.sin(angles_b), jnp.cos(angles_b)]])
        samples_b = jnp.tensordot(boundary_points, rot_mat, axes=(-1, -2))
        return samples, samples_b
        # return grid_points

    @jax.jit
    def sample_grid():
        return grid_points_within, boundary_points

    return sample, sample_deterministic, sample_grid
class CircularQuadratureSampler2(AbstractSampler):
    """
    samples at fixed intervals with appropriate quadrature weights for periodic boundary conditions
    the starting point is uniformly sampled
    does not support multiple devices now
    """
    def __init__(self, nb_sites, nb_samples, radius=1, quad_rule=None, rand_seed=1234):
        super().__init__()
        self.nb_sites = nb_sites
        self.radius = radius
        self.area = jnp.pi * radius ** 2
        # self.pure_funcs = PeriodicQuadratureSamplerPure(self.nb_sites)
        self.pure_funcs = namedtuple('CircularQuadratureSamplerPure', ['sample', 'sample_deterministic', 'sample_grid'])\
            (*gen_circular_quadrature_sample_funcs2(self.nb_sites, nb_samples, self.radius))
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        assert self.nb_devices == 1, f"{self.nb_devices=} must be 1"
        self.nb_samples = nb_samples
        assert quad_rule is None, f"{quad_rule=} must be None for open quadrature sampler, will automatically pick the simplist ones"
        self.sqrt_weights = (jnp.ones(nb_samples) * jnp.sqrt((2 * radius)**2 / nb_samples)).reshape(1, nb_samples)

        self.rand_key = jax.random.PRNGKey(rand_seed)


    def set_var_state(self, var_state):
        # we don't need this function in this case
        pass

    # def sample(self, start=None):
    #     if start is None:
    #         samples, rand_key = self.pure_funcs.sample(self.nb_samples, self.minvals, self.maxvals, self.rand_key)
    #     else:
    #         samples = self.pure_funcs.sample_deterministic(self.nb_samples, self.minvals, self.maxvals, start)
    #         rand_key = self.rand_key
    #     self.rand_key = rand_key
    #     return samples, None, self.sqrt_weights

    def sample(self, start=None, angles=0, angles_b=0):
        if start is None:
            samples, boundaries, rand_key = self.pure_funcs.sample(self.rand_key)
            self.rand_key = rand_key
            return samples, (boundaries, self.sqrt_weights[:, :boundaries.shape[1]]), self.sqrt_weights[:, :samples.shape[1]]
        elif start == 0:
            samples, boundaries = self.pure_funcs.sample_grid()
            return samples, (boundaries, self.sqrt_weights[:, :boundaries.shape[1]]), self.sqrt_weights[:, :samples.shape[1]]
        else:
            samples, boundaries = self.pure_funcs.sample_deterministic(start, angles, angles_b)
            return samples, (boundaries, self.sqrt_weights[:, :boundaries.shape[1]]), self.sqrt_weights[:, :samples.shape[1]]

    def get_state(self):
        return {'rand_key': self.rand_key}

    def set_state(self, state):
        self.rand_key = state['rand_key']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))



if __name__ == '__main__':
    from jax import random
    import matplotlib.pyplot as plt
    key = random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    nb_sites = 2
    nb_samples = 10000
    radius = 1
    sampler = CircularQuadratureSampler(nb_sites, nb_samples, radius)
    samples, boundaries, sqrt_weights = sampler.sample()
    print(samples.shape, sqrt_weights.shape)
    # plot where size means weight
    plt.scatter(samples[0, :, 0], samples[0, :, 1], s=sqrt_weights[0]**2 * nb_samples * 3, alpha=0.5)
    plt.axis('equal')
    plt.savefig('circular_quadrature.png', dpi=500)
    plt.show()

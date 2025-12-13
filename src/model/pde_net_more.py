import math
from typing import List, Tuple, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class DirichletPDENet(nn.Module):
    width: int
    depth: int
    minvals: Tuple[float]
    maxvals: Tuple[float]
    boundary_value: Optional[Callable[[jnp.array], jnp.array]] = None

    @nn.compact
    def __call__(self, x):
        boundary_r = self.boundary_restrict(x)
        boundary_v = self.boundary_value(x) if self.boundary_value is not None else 0.
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width, param_dtype=jnp.float64)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1) * boundary_r + boundary_v

    def boundary_restrict(self, x):
        minvals = jnp.array(self.minvals, dtype=x.dtype)
        maxvals = jnp.array(self.maxvals, dtype=x.dtype)
        intervals = maxvals - minvals
        return jnp.prod(4 * (x - minvals) * (maxvals - x) / intervals**2, -1)

class DirichletPDENet2(nn.Module):
    width: int
    depth: int
    minvals: Tuple[float]
    maxvals: Tuple[float]
    boundary_value: Optional[Callable[[jnp.array], jnp.array]] = None

    @nn.compact
    def __call__(self, x):
        p = self.param('p', lambda key, shape: jnp.ones(shape, dtype=jnp.float64) * jnp.log(jnp.e - 1), (x.shape[-1],))
        boundary_r = self.boundary_restrict(x, p)
        boundary_v = self.boundary_value(x) if self.boundary_value is not None else 0.
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width, param_dtype=jnp.float64)(x)
            # x = nn.tanh(x)
            x = jnp.arcsinh(x)
        return nn.Dense(1)(x).squeeze(-1) * boundary_r + boundary_v

    def boundary_restrict(self, x, p):
        minvals = jnp.array(self.minvals, dtype=x.dtype)
        maxvals = jnp.array(self.maxvals, dtype=x.dtype)
        intervals = maxvals - minvals
        x = 2 * (x - minvals) / intervals - 1
        return jnp.prod(1 - jnp.abs(x) ** (nn.softplus(p) + 4), -1)
        # return jnp.prod(1 - jnp.abs(x) ** (512), -1)

class CircularDirichletPDENet(nn.Module):
    width: int
    depth: int
    radius: float
    boundary_value: Optional[Callable[[jnp.array], jnp.array]] = None

    @nn.compact
    def __call__(self, x):
        boundary_r = self.boundary_restrict(x)
        boundary_v = self.boundary_value(x) if self.boundary_value is not None else 0.
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width, param_dtype=jnp.float64)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1) * boundary_r + boundary_v

    def boundary_restrict(self, x):
        r2 = jnp.sum(x**2, axis=-1)
        return 1 - r2 / self.radius**2

class CircularDirichletPDENet2(nn.Module):
    width: int
    depth: int
    radius: float
    boundary_value: Optional[Callable[[jnp.array], jnp.array]] = None

    @nn.compact
    def __call__(self, x):
        p = self.param('p', lambda key, shape: jnp.ones(shape, dtype=jnp.float64) * jnp.log(jnp.e-1), (1,))
        boundary_r = self.boundary_restrict(x, p)
        boundary_v = self.boundary_value(x) if self.boundary_value is not None else 0.
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width, param_dtype=jnp.float64)(x)
            # x = nn.tanh(x)
            x = jnp.arcsinh(x)
        return nn.Dense(1)(x).squeeze(-1) * boundary_r + boundary_v

    def boundary_restrict(self, x, p):
        r2 = jnp.sum(x ** 2, axis=-1)
        return 1 - (r2 / self.radius) ** (nn.softplus(p) + 4)
        # return 1 - (r2 / self.radius) ** 10

class Rational(nn.Module):
    p: int = 3

    @nn.compact
    def __call__(self, x):
        alpha_init_values = jnp.array([1.1915, 1.5957, 0.5, 0.0218][:self.p+1])
        beta_init_values = jnp.array([2.383, 0.0, 1.0][:self.p])

        alpha = self.param('alpha', lambda rng, shape: alpha_init_values, (self.p+1,))
        beta = self.param('beta', lambda rng, shape: beta_init_values, (self.p,))
        return (alpha[0] * x**3 + alpha[1] * x**2 + alpha[2] * x + alpha[3]) / (beta[0] * x**2 + beta[1] * x + beta[2])

class CircularDirichletPDENet3(nn.Module):
    width: int
    depth: int
    radius: float
    boundary_value: Optional[Callable[[jnp.array], jnp.array]] = None

    @nn.compact
    def __call__(self, x):
        p = self.param('p', lambda key, shape: jnp.ones(shape, dtype=jnp.float64) * jnp.log(jnp.e-1), (1,))
        boundary_r = self.boundary_restrict(x, p)
        boundary_v = self.boundary_value(x) if self.boundary_value is not None else 0.
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width, param_dtype=jnp.float64)(x)
            # x = nn.tanh(x)
            x = Rational()(x)
        return nn.Dense(1)(x).squeeze(-1) * boundary_r + boundary_v

    def boundary_restrict(self, x, p):
        r2 = jnp.sum(x ** 2, axis=-1)
        # return 1 - (r2 / self.radius) ** (nn.softplus(p) + 4)
        return 1 - (r2 / self.radius) ** 16

class PDENet(nn.Module):
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width, param_dtype=jnp.float64)(x)
            x = nn.tanh(x)

        return nn.Dense(1)(x).squeeze(-1)

class PeriodicLinear2(nn.Module):
    nodes: int
    periods: Tuple[float]

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        m = self.nodes // d
        # d, m = x.shape[-1], self.nodes
        a = self.param('a', nn.initializers.truncated_normal(1.0), (m, d))
        phi = self.param('phi', nn.initializers.truncated_normal(1.0), (m, d))
        c = self.param('c', nn.initializers.truncated_normal(1.0), (m, d))
        return (a[None, :, :] * jnp.cos((jnp.pi * 2 / jnp.array(self.periods)) * x[:, None, :] + phi[None, :, :]) + c[None, :, :]).reshape(x.shape[0], self.nodes)

class SimplePDENet4(nn.Module):
    width: int
    depth: int
    periods: Tuple[float]

    @nn.compact
    def __call__(self, x):
        x = PeriodicLinear2(self.width, self.periods)(x)
        x = nn.tanh(x)
        for _ in range(self.depth - 2):
            x = nn.Dense(self.width)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1)


if __name__ == '__main__':
    from jax import random
    key = random.PRNGKey(0)
    # x1 = jnp.linspace(0, 1, 101)
    # x2 = jnp.linspace(0, 1, 101)
    x1 = jnp.linspace(-1, 1, 101)
    x2 = jnp.linspace(-1, 1, 101)
    x = jnp.stack(jnp.meshgrid(x1, x2), axis=-1).reshape(-1, 2)
    model = CircularDirichletPDENet2(width=40, depth=7, radius=1)
    params = model.init(key, x)
    y = model.apply(params, x)
    # make a plot of y in heat map
    import matplotlib.pyplot as plt
    plt.imshow((y * ((x**2).sum(-1) <= 1)).reshape(101, 101), origin='lower')
    plt.colorbar()
    plt.savefig('pde_net.png')
    print(y.shape)

    # from jax import random
    #
    # key = random.PRNGKey(1)
    # # x1 = jnp.linspace(0, 1, 101)
    # # x2 = jnp.linspace(0, 1, 101)
    # x1 = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 101)
    # x2 = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 101)
    # x = jnp.stack(jnp.meshgrid(x1, x2), axis=-1).reshape(-1, 2)
    # model = SimplePDENet4(width=40, depth=7, periods=(2 * jnp.pi, 4 * jnp.pi))
    # params = model.init(key, x)
    # y = model.apply(params, x)
    # # make a plot of y in heat map
    # import matplotlib.pyplot as plt
    #
    # plt.imshow((y).reshape(101, 101), origin='lower')
    # plt.colorbar()
    # plt.savefig('pde_net.png')
    # print(y.shape)
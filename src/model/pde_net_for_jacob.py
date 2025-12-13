import flax.linen as nn
import jax
import jax.numpy as jnp


def build_nn(width, depth, period):
    net = SimplePDENet2(width, depth, period)
    return net


def unraveler(f, unravel, axis=0):
    """
    util to deal with pytrees, will wrap net function
    """
    def wrapper(*args, **kwargs):
        val = args[axis]
        if (type(val) != dict):
            args = list(args)
            args[axis] = unravel(val)
            args = tuple(args)
        return f(*args, **kwargs)
    return wrapper


def init_net(net, key, dim):
    theta_init = net.init(key, jnp.zeros(dim))
    theta_init_flat, unravel = jax.flatten_util.ravel_pytree(theta_init)
    u_scalar = unraveler(net.apply, unravel)
    return u_scalar, theta_init_flat, unravel


class Rational(nn.Module):
    p: int = 3

    @nn.compact
    def __call__(self, x):
        alpha_init_values = jnp.array([1.1915, 1.5957, 0.5, 0.0218][:self.p+1])
        beta_init_values = jnp.array([2.383, 0.0, 1.0][:self.p])

        alpha = self.param('alpha', lambda rng, shape: alpha_init_values, (self.p+1,))
        beta = self.param('beta', lambda rng, shape: beta_init_values, (self.p,))
        return (alpha[0] * x**3 + alpha[1] * x**2 + alpha[2] * x + alpha[3]) / (beta[0] * x**2 + beta[1] * x + beta[2])
        # return jnp.polyval(alpha, x) / jnp.polyval(beta, x)

# class PeriodicLinear0(nn.Module):
#     nodes: int
#     period: float
#
#     @nn.compact
#     def __call__(self, x):
#         d = x.shape[-1]
#         m = self.nodes
#         # d, m = x.shape[-1], self.nodes
#         a = self.param('a', nn.initializers.truncated_normal(1.0), (m, d))
#         phi = self.param('phi', nn.initializers.truncated_normal(1.0), (m, d))
#         c = self.param('c', nn.initializers.truncated_normal(1.0), (m, d))
#         return (a[None, :, :] * jnp.cos((jnp.pi * 2 / self.period) * x[:, None, :] + phi[None, :, :]) + c[None, :, :]).sum(axis=-1)

# class PeriodicLinear(nn.Module):
#     nodes: int
#     period: float
#
#     @nn.compact
#     def __call__(self, x):
#         d = x.shape[-1]
#         m = self.nodes // d
#         # d, m = x.shape[-1], self.nodes
#         a = self.param('a', nn.initializers.truncated_normal(1.0), (m, d))
#         phi = self.param('phi', nn.initializers.truncated_normal(1.0), (m, d))
#         c = self.param('c', nn.initializers.truncated_normal(1.0), (m, d))
#         return (a[None, :, :] * jnp.cos((jnp.pi * 2 / self.period) * x[:, None, :] + phi[None, :, :]) + c[None, :, :]).reshape(x.shape[0], self.nodes)

# class SimplePDENet(nn.Module):
#     width: int
#     depth: int
#     period: float
#
#     @nn.compact
#     def __call__(self, x):
#         x = PeriodicLinear0(self.width, self.period)(x)
#         x = Rational()(x)
#         for _ in range(self.depth - 2):
#             x = nn.Dense(self.width)(x)
#             x = Rational()(x)
#         return nn.Dense(1)(x).squeeze(-1)

class PeriodicLinear2(nn.Module):
    nodes: int
    period: float

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        m = self.nodes // d
        # d, m = x.shape[-1], self.nodes
        a = self.param('a', nn.initializers.truncated_normal(1.0), (m, d))
        phi = self.param('phi', nn.initializers.truncated_normal(1.0), (m, d))
        c = self.param('c', nn.initializers.truncated_normal(1.0), (m, d))
        if x.ndim == 1:
            return (a * jnp.cos((jnp.pi * 2 / self.period) * x + phi) + c).reshape(self.nodes)
        elif x.ndim == 2:
            return (a[None, :, :] * jnp.cos((jnp.pi * 2 / self.period) * x[:, None, :] + phi[None, :, :]) + c[None, :, :]).reshape(x.shape[0], self.nodes)

class SimplePDENet2(nn.Module):
    width: int
    depth: int
    period: float

    @nn.compact
    def __call__(self, x):
        x = PeriodicLinear2(self.width, self.period)(x)
        x = nn.tanh(x)
        for _ in range(self.depth - 2):
            x = nn.Dense(self.width)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1)

# if __name__ == '__main__':
#     from jax import random
#     key = random.PRNGKey(0)
#     x = jnp.ones((1, 5))
#     model = SimplePDENet(width=64, depth=4, period=2.0)
#     params = model.init(key, x)
#     y = model.apply(params, x)
#     print(y.shape)
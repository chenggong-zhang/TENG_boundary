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
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[
    None,
    str,
    lax.Precision,
    Tuple[str, str],
    Tuple[lax.Precision, lax.Precision],
]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]


class MLP(nn.Module):
    """
    MLP with residual connections
    """

    hidden_dim: int
    nb_layers: int
    out_dim: int
    hidden_act: Callable[..., Array] = nn.leaky_relu
    final_act: Callable[..., Array] = None
    use_resnet: bool = True
    use_layer_norm: bool = True
    use_layer_norm_final: bool = False
    kernel_init: Callable[..., Array] = initializers.kaiming_normal()
    bias_init: Callable[..., Array] = initializers.zeros_init()
    kernel_init_final: Callable[..., Array] = initializers.kaiming_normal()
    bias_init_final: Callable[..., Array] = initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        # zeroth layer cannot apply residual connection
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="dense_0",
        )(x)
        x = self.hidden_act(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name="layer_norm_0")(x)
        # first to last-1 layers can apply residual connection
        for i in range(1, self.nb_layers - 1):
            y = nn.Dense(
                self.hidden_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"dense_{i}",
            )(x)
            if self.use_resnet:
                y += x
            x = y
            x = self.hidden_act(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"layer_norm_{i}")(x)
        # last layer cannot apply residual connection
        x = nn.Dense(
            self.out_dim,
            kernel_init=self.kernel_init_final,
            bias_init=self.bias_init_final,
            name=f"dense_{self.nb_layers - 1}",
        )(x)
        if self.final_act is not None:
            x = self.final_act(x)
        if self.use_layer_norm_final:
            x = nn.LayerNorm(name=f"layer_norm_{self.nb_layers - 1}")(x)
        return x

class BiasOnlyNet(nn.Module):
    """
    output bias only, does not depend on input
    """
    out_dim: int
    final_act: Callable[..., Array] = None
    use_layer_norm_final: bool = False
    bias_init_final: Callable[..., Array] = initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        # zeroth layer cannot apply residual connection
        bias = self.param("bias", self.bias_init_final, (self.out_dim,))
        x = jnp.zeros_like(x, shape=(*x.shape[:-1], self.out_dim)) + bias
        if self.final_act is not None:
            x = self.final_act(x)
        if self.use_layer_norm_final:
            x = nn.LayerNorm(name=f"layer_norm")(x)
        return x


class FrequencyEmbedding(nn.Module):
    """
    embedding real numbers into frequencies similar to positional encoding
    """
    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, axis=-1)
        div_term = jnp.exp(jnp.arange(0, self.embedding_dim, 2) * (-jnp.log(100.0) / self.embedding_dim))
        sins = jnp.sin(x * div_term)
        coss = jnp.cos(x * div_term)
        x = jnp.concatenate([sins, coss], axis=-1)
        return x


class RQSNoBoundary(nn.Module):
    min_bin_size: float = 1e-3  # only applies to positive widths and heights
    min_derivative: float = 1e-3  # only applies to positive derivatives
    bin_size_pos_nonlinear: Callable[..., Array] = nn.softplus
    derivative_pos_nonlinear: Callable[..., Array] = nn.softplus
    center_nonlinear: Callable[..., Array] = None

    def searchsorted(self, bin_locations, inputs):
        return jnp.sum(inputs[..., None] >= bin_locations, axis=-1) - 1

    @nn.compact
    def __call__(self, inputs, unnormalized_widths, unnormalized_heights,
                 unnormalized_derivatives, center_x, center_y, inverse=False):
        min_bin_width = self.min_bin_size
        min_bin_height = self.min_bin_size
        min_derivative = self.min_derivative

        if self.center_nonlinear is not None:
            center_x = self.center_nonlinear(center_x)
            center_y = self.center_nonlinear(center_y)

        if self.bin_size_pos_nonlinear is not None:
            unnormalized_widths = self.bin_size_pos_nonlinear(unnormalized_widths)
            unnormalized_heights = self.bin_size_pos_nonlinear(unnormalized_heights)

        if self.derivative_pos_nonlinear is not None:
            unnormalized_derivatives = self.derivative_pos_nonlinear(unnormalized_derivatives)

        widths = min_bin_width + unnormalized_widths
        heights = min_bin_height + unnormalized_heights
        derivatives = min_derivative + unnormalized_derivatives

        cumwidths = jnp.cumsum(widths, axis=-1)
        num_dims = len(cumwidths.shape)
        pad_config = [(0, 0)] * (num_dims - 1) + [(1, 0)]
        cumwidths = jnp.pad(cumwidths, pad_width=pad_config, mode='constant', constant_values=0.0)
        cumwidths = cumwidths - cumwidths[..., -1:] / 2 + center_x[..., None]

        cumheights = jnp.cumsum(heights, axis=-1)
        num_dims = len(heights.shape)
        pad_config = [(0, 0)] * (num_dims - 1) + [(1, 0)]
        cumheights = jnp.pad(cumheights, pad_width=pad_config, mode='constant', constant_values=0.0)
        cumheights = cumheights - cumheights[..., -1:] / 2 + center_y[..., None]

        outputs = jnp.zeros_like(inputs)
        logabsdet = jnp.zeros_like(inputs)

        if inverse:
            left_mask = inputs < cumheights[..., 0]
            right_mask = inputs >= cumheights[..., -1]
            outputs = jnp.where(left_mask, cumwidths[..., 0] - (cumheights[..., 0] - inputs) / derivatives[..., 0],
                                outputs)
            outputs = jnp.where(right_mask, (inputs - cumheights[..., -1]) / derivatives[..., -1] + cumwidths[..., -1],
                                outputs)
            logabsdet = jnp.where(left_mask, jnp.log(derivatives[..., 0]), logabsdet)
            logabsdet = jnp.where(right_mask, jnp.log(derivatives[..., -1]), logabsdet)
        else:
            left_mask = inputs < cumwidths[..., 0]
            right_mask = inputs >= cumwidths[..., -1]
            outputs = jnp.where(left_mask, cumheights[..., 0] - (cumwidths[..., 0] - inputs) * derivatives[..., 0],
                                outputs)
            outputs = jnp.where(right_mask, (inputs - cumwidths[..., -1]) * derivatives[..., -1] + cumheights[..., -1],
                                outputs)
            logabsdet = jnp.where(left_mask, jnp.log(derivatives[..., 0]), logabsdet)
            logabsdet = jnp.where(right_mask, jnp.log(derivatives[..., -1]), logabsdet)

        inside_mask = ~(left_mask | right_mask)
        # inside_inds = jnp.nonzero(inside_mask)
        # inputs_inside = inputs[inside_inds]
        # heights_inside = heights[inside_inds]
        # widths_inside = widths[inside_inds]
        # cumheights_inside = cumheights[inside_inds]
        # cumwidths_inside = cumwidths[inside_inds]
        # derivatives_inside = derivatives[inside_inds]

        if inverse:
            # bin_idx = self.searchsorted(cumheights_inside, inputs_inside)[..., None]
            inputs_stable = jnp.where(inside_mask, inputs, center_y)
            # bin_idx = jnp.searchsorted(cumheights, inputs_stable)[..., None] - 1 # we use the convention with the zeroth interval
            bin_idx = self.searchsorted(cumheights, inputs_stable)[..., None]
        else:
            #bin_idx = self.searchsorted(cumwidths_inside, inputs_inside)[..., None]
            inputs_stable = jnp.where(inside_mask, inputs, center_x)
            # bin_idx = jnp.searchsorted(cumwidths, inputs_stable)[..., None] - 1 # we use the convention with the zeroth interval
            bin_idx = self.searchsorted(cumwidths, inputs_stable)[..., None]

        # input_cumwidths = jnp.take_along_axis(cumwidths_inside, bin_idx, -1).squeeze(-1)
        # input_bin_widths = jnp.take_along_axis(widths_inside, bin_idx, -1).squeeze(-1)

        input_cumwidths = jnp.take_along_axis(cumwidths, bin_idx, -1).squeeze(-1)
        input_bin_widths = jnp.take_along_axis(widths, bin_idx, -1).squeeze(-1)

        # input_cumheights = jnp.take_along_axis(cumheights_inside, bin_idx, -1).squeeze(-1)
        # delta = heights_inside / widths_inside
        # input_delta = jnp.take_along_axis(delta, bin_idx, -1).squeeze(-1)

        input_cumheights = jnp.take_along_axis(cumheights, bin_idx, -1).squeeze(-1)
        delta = heights / widths
        input_delta = jnp.take_along_axis(delta, bin_idx, -1).squeeze(-1)

        # input_derivatives = jnp.take_along_axis(derivatives_inside, bin_idx, -1).squeeze(-1)
        # input_derivatives_plus_one = jnp.take_along_axis(derivatives_inside[..., 1:], bin_idx, -1).squeeze(-1)
        # input_derivatives_plus_one = input_derivatives_plus_one.squeeze(-1)

        input_derivatives = jnp.take_along_axis(derivatives, bin_idx, -1).squeeze(-1)
        input_derivatives_plus_one = jnp.take_along_axis(derivatives[..., 1:], bin_idx, -1).squeeze(-1)
        # input_derivatives_plus_one = input_derivatives_plus_one.squeeze(-1)

        # input_heights = jnp.take_along_axis(heights_inside, bin_idx, -1).squeeze(-1)

        input_heights = jnp.take_along_axis(heights, bin_idx, -1).squeeze(-1)

        if inverse:
            a = (((inputs_stable - input_cumheights) * (input_derivatives \
                                                        + input_derivatives_plus_one - 2 * input_delta) \
                  + input_heights * (input_delta - input_derivatives)))
            b = (input_heights * input_derivatives - (inputs_stable - input_cumheights) \
                 * (input_derivatives + input_derivatives_plus_one \
                    - 2 * input_delta))
            c = - input_delta * (inputs_stable - input_cumheights)

            discriminant = b**2 - 4 * a * c
            # assert (discriminant >= 0).all()  # will not work when jitted

            root = (2 * c) / (-b - jnp.sqrt(discriminant))
            # outputs.at[inside_inds].set(root * input_bin_widths + input_cumwidths)
            outputs = jnp.where(inside_mask, root * input_bin_widths + input_cumwidths, outputs)

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta \
                          + ((input_derivatives + input_derivatives_plus_one \
                              - 2 * input_delta) * theta_one_minus_theta)
            derivative_numerator = input_delta**2 \
                                   * (input_derivatives_plus_one * root**2 \
                                      + 2 * input_delta * theta_one_minus_theta \
                                      + input_derivatives * (1 - root)**2)
            # logabsdet.at[inside_inds].set(jnp.log(derivative_numerator) - 2 * jnp.log(denominator))
            logabsdet = jnp.where(inside_mask, jnp.log(derivative_numerator) - 2 * jnp.log(denominator), logabsdet)
            return outputs, logabsdet
        else:
            theta = (inputs_stable - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (input_delta * theta**2 \
                                         + input_derivatives * theta_one_minus_theta)
            denominator = input_delta + ((input_derivatives \
                                          + input_derivatives_plus_one - 2 * input_delta) \
                                         * theta_one_minus_theta)
            # outputs.at[inside_inds].set(input_cumheights + numerator / denominator)
            outputs = jnp.where(inside_mask, input_cumheights + numerator / denominator, outputs)

            derivative_numerator = input_delta**2 \
                                   * (input_derivatives_plus_one * theta**2 \
                                      + 2 * input_delta * theta_one_minus_theta \
                                      + input_derivatives * (1 - theta)**2)
            # logabsdet.at[inside_inds].set(jnp.log(derivative_numerator) - 2 * jnp.log(denominator))
            logabsdet = jnp.where(inside_mask, jnp.log(derivative_numerator) - 2 * jnp.log(denominator), logabsdet)
            return outputs, logabsdet

def invsymlu(x: Array) -> Array:
    """
    inverse symmetrical linear unit
    """
    return (x + 1) * (x >= 0) + (1 / (1 - x)) * (x < 0)

class SimpleARRQSFlow(nn.Module):
    """
    autoregressive rational quadratic spline flow without tail bounds
    """
    nb_sites: int
    nb_intervals: int
    min_bin_size: float
    min_derivative: float
    hidden_dim: int
    nb_layers: int
    use_embedding: bool = True
    keep_input: bool = False  # whether to keep input as part of embedding
    embedding_dim: int = 8
    bin_size_pos_nonlinear: Callable[..., Array] = invsymlu
    derivative_pos_nonlinear: Callable[..., Array] = invsymlu
    center_nonlinear: Callable[..., Array] = None
    hidden_act: Callable[..., Array] = nn.gelu
    use_resnet: bool = True
    use_layer_norm: bool = True
    use_layer_norm_final: bool = False
    kernel_init: Callable[..., Array] = initializers.kaiming_uniform()
    bias_init: Callable[..., Array] = initializers.zeros_init()
    kernel_init_final: Callable[..., Array] = initializers.zeros_init()
    bias_init_final: Callable[..., Array] = initializers.uniform(scale=0.01)


    def setup(self):
        self.mlps = \
            (BiasOnlyNet(out_dim=3*self.nb_intervals+3,
                         final_act=None,
                         use_layer_norm_final=self.use_layer_norm_final,
                         bias_init_final=self.bias_init_final),) + \
            tuple(MLP(hidden_dim=self.hidden_dim,
                      nb_layers=self.nb_layers,
                      out_dim=3*self.nb_intervals+3,
                      hidden_act=self.hidden_act,
                      final_act=None,
                      use_resnet=self.use_resnet,
                      use_layer_norm=self.use_layer_norm,
                      use_layer_norm_final=self.use_layer_norm_final,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init,
                      kernel_init_final=self.kernel_init_final,
                      bias_init_final=self.bias_init_final) for _ in range(self.nb_sites-1))
        if self.use_embedding:
            self.embedding = FrequencyEmbedding(embedding_dim=self.embedding_dim)

        self.rqs = RQSNoBoundary(min_bin_size=self.min_bin_size,
                                 min_derivative=self.min_derivative,
                                 bin_size_pos_nonlinear=self.bin_size_pos_nonlinear,
                                 derivative_pos_nonlinear=self.derivative_pos_nonlinear,
                                 center_nonlinear=self.center_nonlinear)

    def embed(self, x):
        if self.use_embedding:
            if self.keep_input:
                x = jnp.concatenate((x, self.embedding(x)), axis=-1)
            else:
                x = self.embedding(x)
        else:
            x = x[..., None]
        return x

    def __call__(self, x):
        assert x.shape == (x.shape[0], self.nb_sites)
        x_embedded = self.embed(x)
        mlp_outs = []
        for i, mlp in enumerate(self.mlps):
            mlp_out = mlp(x_embedded[:, :i].reshape(x_embedded.shape[0], -1))
            mlp_outs.append(mlp_out)
        mlp_outs = jnp.stack(mlp_outs, axis=1)

        widths = mlp_outs[..., :self.nb_intervals]
        heights = mlp_outs[..., self.nb_intervals:2*self.nb_intervals]
        derivatives = mlp_outs[..., 2*self.nb_intervals:3*self.nb_intervals+1]
        centerx = mlp_outs[..., -2]
        centery = mlp_outs[..., -1]

        z, logJ = self.rqs(x, widths, heights, derivatives, centerx, centery, inverse=True)
        log_px = jax.scipy.stats.norm.logpdf(z).sum(-1) - logJ.sum(-1)
        return log_px / 2

    def sample(self, shape, rng, z_init=None):
        if z_init is None:
            z = jax.random.normal(rng, shape=shape)
        else:
            z = z_init
        assert z.shape == (z.shape[0], self.nb_sites)

        log_pz = jax.scipy.stats.norm.logpdf(z).sum(-1)
        logJ = jnp.zeros(shape[0])

        # mlp_outs = [self.zeroth_site_bias]
        #
        # widths = mlp_outs[-1][..., :self.nb_intervals]
        # heights = mlp_outs[-1][..., self.nb_intervals:2 * self.nb_intervals]
        # derivatives = mlp_outs[-1][..., 2 * self.nb_intervals:3 * self.nb_intervals + 1]
        # centerx = mlp_outs[-1][..., -2]
        # centery = mlp_outs[-1][..., -1]
        #
        # x0, logJ_new = self.rqs(z[:, 0], widths, heights, derivatives, centerx, centery, inverse=False)
        #
        # logJ += logJ_new
        #
        # if self.use_embedding:
        #     if self.keep_input:
        #         x0_embedded = jnp.concatenate((x0, self.embedding(x0)), axis=-1)
        #     else:
        #         x0_embedded  = self.embedding(x0)
        # else:
        #     x0_embedded = x0

        # x_list = [x0]
        # x_embedded_list = [x0_embedded]

        # mlp_outs = []
        # x_list = []
        # x_embedded_list = [self.embed(z[..., :0])] # just a zero element place holder for batch dim

        mlp_outs = jnp.zeros((shape[0], self.nb_sites, 3*self.nb_intervals+3))
        x = jnp.zeros((shape[0], self.nb_sites))
        x_embedded = jnp.zeros((shape[0], self.nb_sites, self.embed(z[..., :0, :]).shape[-1]))

        for i, mlp in enumerate(self.mlps):
            mlp_out = mlp(x_embedded[:, :i].reshape(x_embedded.shape[0], -1))
            mlp_outs = mlp_outs.at[..., i, :].set(mlp_out)

            widths = mlp_outs[..., i, :self.nb_intervals]
            heights = mlp_outs[..., i, self.nb_intervals:2 * self.nb_intervals]
            derivatives = mlp_outs[..., i, 2 * self.nb_intervals:3 * self.nb_intervals + 1]
            centerx = mlp_outs[..., i, -2]
            centery = mlp_outs[..., i, -1]

            x_new, logJ_new = self.rqs(z[..., i], widths, heights, derivatives, centerx, centery, inverse=False)

            logJ += logJ_new

            x_new_embedded = self.embed(x_new)

            x = x.at[..., i].set(x_new)
            x_embedded = x_embedded.at[..., i, :].set(x_new_embedded)

        # x = jnp.concatenate(x_list, axis=-1)
        log_px = log_pz - logJ

        return x, log_px / 2


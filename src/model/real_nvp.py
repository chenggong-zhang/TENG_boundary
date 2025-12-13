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

class SimpleRealNVP(nn.Module):
    flows: Tuple[nn.Module]  # A tuple of flows (each a nn.Module) that should be applied on the images.
    import_samples: int = 8  # Number of importance samples to use during testing (see explanation below).

    def __call__(self, x, testing=False):
        if not testing:
            bpd = self._get_log_wf(x)
        else:
            # Perform importance sampling during testing => estimate likelihood M times for each image
            img_ll, rng = self._get_likelihood(x.repeat(self.import_samples, 0),
                                               return_ll=True)
            img_ll = img_ll.reshape(-1, self.import_samples)

            # To average the probabilities, we need to go from log-space to exp, and back to log.
            # Logsumexp provides us a stable implementation for this
            img_ll = jax.nn.logsumexp(img_ll, axis=-1) - np.log(self.import_samples)

            # Calculate final bpd
            bpd = -img_ll * np.log2(np.exp(1)) / np.prod(x.shape[1:])
            bpd = bpd.mean()
        return bpd

    def encode(self, imgs):
        # Given a batch of images, return the latent representation z and
        # log-determinant jacobian (ldj) of the transformations
        z, ldj = imgs, jnp.zeros(imgs.shape[0])
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_log_wf(self, imgs, return_ll=True):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        imgs = imgs[:, :, :, None] # add channel dimension
        z, ldj = self.encode(imgs)
        log_pz = jax.scipy.stats.norm.logpdf(z).sum(axis=(1,2,3))
        log_px = ldj + log_pz
        return log_px / 2 # divide by 2 because we are dealing with wavefunction
        # nll = -log_px
        # # Calculating bits per dimension
        # bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        # return (bpd.mean() if not return_ll else log_px), rng

    def sample(self, img_shape, rng, z_init=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        img_shape = img_shape + (1,) # add channel dimension
        if z_init is None:
            z = jax.random.normal(rng, shape=img_shape)
        else:
            z = z_init

        log_pz = jax.scipy.stats.norm.logpdf(z).sum(axis=(1, 2, 3))

        # Transform z to x by inverting the flows
        # The log-determinant jacobian (ldj) is usually not of interest during sampling
        ldj = jnp.zeros(img_shape[0])
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        log_px = -ldj + log_pz
        return z.squeeze(-1), log_px / 2 # remove channel dimension and divide by 2 because we are dealing with wavefunction


class CouplingLayer(nn.Module):
    network: nn.Module  # NN to use in the flow for predicting mu and sigma
    mask_init: Callable[[PRNGKey, Shape, Dtype], Array]  # Binary mask where 0 denotes that the element should be transformed, and 1 not.
    c_in: int  # Number of input channels

    # def setup(self):
    #     self.scaling_factor = self.param('scaling_factor',
    #                                      nn.initializers.zeros,
    #                                      (self.c_in,))
    #     self.mask = self.variable('mask', 'checkerboard_mask', ())

    @nn.compact
    def __call__(self, z, ldj, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            rng - PRNG state
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        mask = self.variable('mask', 'checkerboard_mask', self.mask_init)
        mask_value = mask.value
        z_in = z * mask_value
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(jnp.concatenate([z_in, orig_img], axis=-1))
        s, t = jnp.split(nn_out, 2, axis=-1)

        # Stabilize scaling output
        scaling_factor = self.param('scaling_factor',
                                    nn.initializers.zeros,
                                    (self.c_in,))
        s_fac = jnp.exp(scaling_factor).reshape(1, 1, 1, -1)
        s = nn.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - mask_value)
        t = t * (1 - mask_value)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * jnp.exp(s)
            ldj += s.sum(axis=[1,2,3])
        else:
            z = (z * jnp.exp(-s)) - t
            ldj -= s.sum(axis=[1,2,3])

        return z, ldj


class ScaleShiftLayer(nn.Module):

    # def setup(self):
    #     self.scaling_factor = self.param('scaling_factor',
    #                                      nn.initializers.zeros,
    #                                      (self.c_in,))
    #     self.mask = self.variable('mask', 'checkerboard_mask', ())

    @nn.compact
    def __call__(self, z, ldj, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            rng - PRNG state
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        s = self.param('scaling_param', nn.initializers.zeros, (1, *z.shape[-3:]))
        t = self.param('shift_param', nn.initializers.zeros, (1, *z.shape[-3:]))

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * jnp.exp(s)
            ldj += s.sum(axis=[1,2,3])
        else:
            z = (z * jnp.exp(-s)) - t
            ldj -= s.sum(axis=[1,2,3])

        return z, ldj


def create_checkerboard_mask(h, w, invert=False):
    x, y = jnp.arange(h, dtype=jnp.int32), jnp.arange(w, dtype=jnp.int32)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    mask = jnp.fmod(xx + yy, 2)
    mask = mask.astype(jnp.float32).reshape(1, h, w, 1)
    if invert:
        mask = 1 - mask
    return mask

def create_channel_mask(c_in, invert=False):
    mask = jnp.concatenate([
                jnp.ones((c_in//2,), dtype=jnp.float32),
                jnp.zeros((c_in-c_in//2,), dtype=jnp.float32)
            ])
    mask = mask.reshape(1, 1, 1, c_in)
    if invert:
        mask = 1 - mask
    return mask


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def __call__(self, x):
        return jnp.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)


class GatedConv(nn.Module):
    """ This module applies a two-layer convolutional ResNet block with input gate """
    c_in: int  # Number of input channels
    c_hidden: int  # Number of hidden dimensions
    kernel_size: Shape = (3, 3)  # Kernel size to use in the convolution

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential([
            ConcatELU(),
            nn.Conv(self.c_hidden, kernel_size=self.kernel_size, padding='SAME'),
            ConcatELU(),
            nn.Conv(2*self.c_in, kernel_size=(1, 1))
        ])(x)
        val, gate = jnp.split(out, 2, axis=-1)
        return x + val * nn.sigmoid(gate)


# class GatedConvNet(nn.Module):
#     c_hidden: int  # Number of hidden dimensions to use within the network
#     c_out: int  # Number of output channels
#     num_layers: int = 3 # Number of gated ResNet blocks to apply
#     kernel_size: Shape = (3, 3)  # Kernel size to use in the convolution
#
#     def setup(self):
#         layers = []
#         layers += [nn.Conv(self.c_hidden, kernel_size=self.kernal_size, padding='SAME')]
#         for layer_index in range(self.num_layers):
#             layers += [GatedConv(self.c_hidden, self.c_hidden, self.kernel_size),
#                        nn.LayerNorm()]
#         layers += [ConcatELU(),
#                    nn.Conv(self.c_out, kernel_size=self.kernel_size, padding='SAME',
#                            kernel_init=nn.initializers.zeros)]
#         self.nn = nn.Sequential(layers)
#
#     def __call__(self, x):
#         return self.nn(x)

def normal_bias_init(key, shape, dtype=jnp.float32):
    # mean = 0, std_dev = 0.1
    return jax.random.normal(key, shape, dtype) * 0.1

class GatedConvNet(nn.Module):
    c_hidden: int  # Number of hidden dimensions to use within the network
    c_out: int  # Number of output channels
    num_layers: int = 3  # Number of gated ResNet blocks to apply
    kernel_size: Tuple[int, int] = (3, 3)  # Kernel size to use in the convolution

    @nn.compact
    def __call__(self, x):
        # First convolutional layer
        x = nn.Conv(self.c_hidden, kernel_size=self.kernel_size, padding='SAME')(x)

        # Gated ResNet blocks
        for _ in range(self.num_layers):
            x = GatedConv(self.c_hidden, self.c_hidden, self.kernel_size)(x)
            x = nn.LayerNorm()(x)

        # Final layers
        x = ConcatELU()(x)
        x = nn.Conv(self.c_out, kernel_size=self.kernel_size, padding='SAME', kernel_init=nn.initializers.zeros)(x)
        # x = nn.Conv(self.c_out, kernel_size=self.kernel_size, padding='SAME', bias_init=normal_bias_init)(x)

        return x


# use upper case letter since this creates a neural network
def CreateSimpleRealNVP(nb_rows, nb_columns, hidden_dim=32, nb_layers=3, kernel_size=(3, 3), nb_flow_layers=8, scale_and_shift_before=False, scale_and_shift_after=False):
    flow_layers = []
    if scale_and_shift_before:
        flow_layers += [ScaleShiftLayer()]

    for i in range(nb_flow_layers):
        mask = create_checkerboard_mask(h=nb_rows, w=nb_columns, invert=(i%2==1))
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=hidden_dim, num_layers=nb_layers, kernel_size=kernel_size),
                                      mask_init=lambda m=mask: m, c_in=1)] # the lambda function is defined (with a default value) such that the mask is not shared between layers

    if scale_and_shift_after:
        flow_layers += [ScaleShiftLayer()]
        
    flow_model = SimpleRealNVP(tuple(flow_layers))
    return flow_model
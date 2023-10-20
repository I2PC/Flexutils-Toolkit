# **************************************************************************
# *
# * Authors:  David Herreros Calero (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.init_ops_v2 import _compute_fans
from tf_siren.meta.meta_siren import HyperNetBlock


class Sine(tf.keras.layers.Layer):
    def __init__(self, w0: float = 1.0, **kwargs):
        """
        Sine activation function with w0 scaling support.
        Args:
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`
        """
        super(Sine, self).__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        return tf.sin(self.w0 * inputs)

    def get_config(self):
        config = {'w0': self.w0}
        base_config = super(Sine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SIRENFirstLayerInitializer(tf.keras.initializers.Initializer):

    def __init__(self, scale=1.0, seed=None):
        super().__init__()
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, fan_out = _compute_fans(shape)
        limit = self.scale / max(1.0, float(fan_in))
        return tf.random.uniform(shape, -limit, limit, seed=self.seed)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'scale': self.scale,
            'seed': self.seed
        }
        return dict(list(base_config.items()) + list(config.items()))


class SIRENInitializer(tf.keras.initializers.VarianceScaling):

    def __init__(self, w0: float = 1.0, c: float = 6.0, seed: int = None):
        # Uniform variance scaler multiplies by 3.0 for limits, so scale down here to compensate
        self.w0 = w0
        self.c = c
        scale = c / (3.0 * w0 * w0)
        super(SIRENInitializer, self).__init__(scale=scale, mode='fan_in', distribution='uniform', seed=seed)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'w0': self.w0,
            'c': self.c
        }
        return dict(list(base_config.items()) + list(config.items()))


class MetaDenseWrapper(tf.keras.layers.Layer):
    def __init__(self,
                 input_units: int,
                 output_units: int,
                 hyper_units: int,
                 num_hyper_layers: int = 1,
                 w0: float = 1.0,
                 hyper_activation: str = 'relu',
                 use_bias: bool = True,
                 meta_kernel_initializer=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.layer = MetaDense(input_units, output_units, hyper_units, num_hyper_layers, w0=w0,
                               hyper_activation=hyper_activation, use_bias=use_bias,
                               meta_kernel_initializer=meta_kernel_initializer, **kwargs)

    @tf.function
    def call(self, inputs):
        param_list = self.layer(inputs)
        return self.layer.inner_call(inputs, param_list)


class _MetaDense(layers.Dense):
    """
    A Meta wrapper over a Dense.
    Does not have its own weights, accepts parameters during forward call via `params`.
    Unpacks these params and reshapes them for use in batched call of multiple
    kernels and biases over individual samples in the batch.
    """
    @tf.function
    def __call__(self, inputs, params=None):
        # input = [batch, input_dim]
        # kernel = [batch, input_dim, output_dim]
        # bias = [batch, output_dim]

        if self.use_bias:
            kernel, bias = params
        else:
            kernel = params

        outputs = tf.matmul(inputs[:, None, :], kernel)

        if self.use_bias:
            bias = tf.expand_dims(bias, axis=1)
            outputs += bias

        outputs = outputs[:, 0, :]

        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable

        return outputs


class MetaDense(tf.keras.layers.Layer):
    def __init__(self,
                 input_units: int,
                 output_units: int,
                 hyper_units: int,
                 num_hyper_layers: int = 1,
                 w0: float = 1.0,
                 hyper_activation: str = 'relu',
                 use_bias: bool = True,
                 meta_kernel_initializer=None,
                 **kwargs):

        super().__init__(**kwargs)

        self.input_units = input_units
        self.output_units = output_units
        self.hyper_units = hyper_units

        total_param_count = input_units * output_units
        if use_bias:
            total_param_count += output_units

        if meta_kernel_initializer is None:
            meta_kernel_initializer = SIRENInitializer()

        self.total_param_count = total_param_count
        self.kernel_param_count = input_units * output_units
        self.bias_param_count = output_units

        # Model which provides parameters for inner model
        self.hyper_net = HyperNetBlock(
            input_units=input_units, output_units=total_param_count, hyper_units=hyper_units,
            activation=hyper_activation, num_hyper_layers=num_hyper_layers,
            hyper_final_activation='linear', use_bias=use_bias)

        # Weights won't be generated for this meta layer, just its forward method will be used
        self.inner_siren = _MetaDense(output_units, activation=Sine(w0=w0),
                                      kernel_initializer=meta_kernel_initializer, use_bias=use_bias)

        # Don't allow to build weights
        self.inner_siren.built = True

    @tf.function
    def call(self, context, **kwargs):
        parameters = self.hyper_net(context)  # [B, total_parameter_count]

        # Unpack kernel weights from generated parameters
        kernel = tf.reshape(parameters[:, :self.kernel_param_count],
                            [-1, self.input_units, self.output_units])

        # Unpack bias parameters if available
        if self.hyper_net.use_bias:
            bias = tf.reshape(parameters[:, self.kernel_param_count:], [-1, self.output_units])
        else:
            bias = None

        if self.hyper_net.use_bias:
            return kernel, bias
        else:
            return kernel

    @tf.function
    def inner_call(self, inputs, params, **kwargs):
        """
        A convenience method to perform a forward pass over the meta layer.
        Requires the weights generated from the call() method to be passed as `params`.

        Args:
            inputs: Input tensors to the meta layer.
            params: Parameters of this meta layer.
        """
        outputs = self.inner_siren(inputs, params=params)
        return outputs

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
from tensorflow.keras import layers, initializers
from tensorflow.python.ops.init_ops_v2 import _compute_fans
from tf_siren.meta.meta_siren import HyperNetBlock

# 1) A serializable Sine activation
@tf.keras.utils.register_keras_serializable()
class Sine(layers.Layer):
    def __init__(self, w0: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        return tf.sin(self.w0 * inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'w0': self.w0})
        return config

# 2) Initializer for the very first SIREN layer
@tf.keras.utils.register_keras_serializable()
class SIRENFirstLayerInitializer(initializers.Initializer):
    def __init__(self, scale=1.0, seed=None):
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        fan_in, _ = _compute_fans(shape)
        limit = self.scale / max(1.0, float(fan_in))
        return tf.random.uniform(shape, -limit, limit, seed=self.seed)

    def get_config(self):
        return {'scale': self.scale, 'seed': self.seed}

# 3) General SIREN initializer
@tf.keras.utils.register_keras_serializable()
class SIRENInitializer(initializers.VarianceScaling):
    def __init__(self, w0: float = 1.0, c: float = 6.0, seed: int = None):
        self.w0 = w0
        self.c = c
        # compensate for uniform's 3x factor
        scale = c / (3.0 * w0 * w0)
        super().__init__(scale=scale, mode='fan_in', distribution='uniform', seed=seed)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'w0': self.w0, 'c': self.c})
        return cfg

# 4) The “inner” Meta-Dense that applies per-sample weights
@tf.keras.utils.register_keras_serializable()
class _MetaDense(layers.Dense):
    def __call__(self, inputs, params=None):
        # unpack
        if self.use_bias:
            kernel, bias = params
        else:
            kernel = params

        x = tf.matmul(inputs[:, None, :], kernel)
        if self.use_bias:
            bias = tf.expand_dims(bias, 1)
            x = x + bias
        x = x[:, 0, :]
        if self.activation:
            return self.activation(x)
        return x

# 5) The MetaDense itself, which builds a little hyper-network
@tf.keras.utils.register_keras_serializable()
class MetaDense(layers.Layer):
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
        self.num_hyper_layers = num_hyper_layers
        self.w0 = w0
        self.hyper_activation = hyper_activation
        self.use_bias = use_bias

        # count parameters
        self.kernel_param_count = input_units * output_units
        self.bias_param_count = output_units if use_bias else 0
        self.total_param_count = self.kernel_param_count + self.bias_param_count

        if meta_kernel_initializer is None:
            meta_kernel_initializer = SIRENInitializer(w0=w0)

        # the hyper-network that spits out [batch, total_param_count]
        self.hyper_net = HyperNetBlock(
            input_units=input_units,
            output_units=self.total_param_count,
            hyper_units=hyper_units,
            num_hyper_layers=num_hyper_layers,
            activation=hyper_activation,
            hyper_final_activation='linear',
            use_bias=use_bias
        )

        # a dummy Dense that we'll build _only_ to get tracked variables,
        # but we will actually call it with our own `params`
        self.inner_siren = _MetaDense(
            units=output_units,
            activation=Sine(w0=w0),
            kernel_initializer=meta_kernel_initializer,
            use_bias=use_bias
        )

    def build(self, input_shape):
        # build hyper-net to create its weights
        self.hyper_net.build(input_shape)
        # also build the inner Dense (so its weights show up in checkpoints)
        # we just need a fake shape: (None, input_units)
        self.inner_siren.build((None, self.input_units))
        super().build(input_shape)

    def call(self, context):
        # 1) generate all weights
        params = self.hyper_net(context)
        # 2) split
        k = tf.reshape(params[:, :self.kernel_param_count],
                       [-1, self.input_units, self.output_units])
        if self.use_bias:
            b = tf.reshape(params[:, self.kernel_param_count:],
                           [-1, self.output_units])
            params = (k, b)
        else:
            params = k
        # 3) apply per-sample
        return self.inner_siren(context, params=params)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'input_units': self.input_units,
            'output_units': self.output_units,
            'hyper_units': self.hyper_units,
            'num_hyper_layers': self.num_hyper_layers,
            'w0': self.w0,
            'hyper_activation': self.hyper_activation,
            'use_bias': self.use_bias,
            'meta_kernel_initializer': tf.keras.initializers.serialize(self.inner_siren.kernel_initializer)
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        # rehydrate initializer
        init_conf = config.pop('meta_kernel_initializer')
        config['meta_kernel_initializer'] = tf.keras.initializers.deserialize(init_conf)
        return cls(**config)

# 6) A convenient wrapper that both generates and applies in one call
@tf.keras.utils.register_keras_serializable()
class MetaDenseWrapper(layers.Layer):
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
        self.meta = MetaDense(
            input_units=input_units,
            output_units=output_units,
            hyper_units=hyper_units,
            num_hyper_layers=num_hyper_layers,
            w0=w0,
            hyper_activation=hyper_activation,
            use_bias=use_bias,
            meta_kernel_initializer=meta_kernel_initializer
        )

    def call(self, inputs):
        return self.meta(inputs)

    def get_config(self):
        cfg = super().get_config()
        # pull everything out of self.meta
        mc = self.meta.get_config()
        cfg.update({k: mc[k] for k in mc if k not in cfg})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

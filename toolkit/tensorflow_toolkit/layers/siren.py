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
from tensorflow.python.ops.init_ops_v2 import _compute_fans


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

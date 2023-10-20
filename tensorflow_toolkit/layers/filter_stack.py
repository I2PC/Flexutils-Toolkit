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


class FilterStack(tf.keras.layers.Layer):
    def __init__(self, kernel_size=11, variance=0.01, num_filters=4,
                 scaling=10, activation='linear', **args):
        super(FilterStack, self).__init__()

        self.kernel_size = kernel_size
        self.variance = variance
        self.num_filters = num_filters
        self.scaling = scaling
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "variance": self.variance,
            "num_filters": self.num_filters,
            "scaling": self.scaling,
            "activation": self.act,
        })
        return config

    def generateKernels(self, size, var, scales=1, scaling=2):
        coords = tf.range(-(size // 2), size // 2 + 1, dtype=tf.float32)
        xy = tf.stack(tf.meshgrid(coords, coords), axis=0)
        kernels = [tf.exp(-tf.reduce_sum(xy ** 2, axis=0) / (2 * var * scaling ** i)) for i in range(scales)]
        kernels = tf.stack(kernels, axis=2)
        kernels /= tf.reduce_sum(kernels, axis=(0, 1), keepdims=True)
        kernels = kernels[:, :, None, :]
        return kernels

    def build(self, shp):
        self.w = self.generateKernels(self.kernel_size, self.variance,
                                      self.num_filters + 1, self.scaling)

    def call(self, inp):
        return tf.nn.conv2d(inp, self.w, strides=(1, 1), padding="SAME")

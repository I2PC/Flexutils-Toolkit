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


import numpy as np

import prody as pd

import tensorflow as tf

from toolkit.tensorflow_toolkit.generators.generator_template import DataGeneratorBase
from toolkit.tensorflow_toolkit.utils import euler_matrix_batch


class Generator(DataGeneratorBase):
    def __init__(self, n_modes=20, refinePose=True, struct_file="", sr=1.0, **kwargs):
        super().__init__(**kwargs)

        self.refinePose = refinePose

        # Inverse Sampling rate
        self.i_sr = 1 / sr

        # # Read structure data
        # self.coords, self.values, self.pd_struct = self.readStructureData(struct_file)

        # NMA vector size
        self.n_modes = n_modes

        # Precompute Zernike3D basis
        self.U = self.computeBasis(self.coords, n_modes=n_modes)

        # Initialize pose information
        if refinePose:
            self.rot_batch = np.zeros(self.batch_size)
            self.tilt_batch = np.zeros(self.batch_size)
            self.psi_batch = np.zeros(self.batch_size)


    # ----- Utils -----#

    def computeBasis(self, coords, n_modes=20, **kwargs):
        filename = kwargs.pop('filename', None)

        if filename is None:
            anm_start = pd.ANM(coords)
            anm_start.buildHessian(coords, **kwargs)
            kwargs.pop('cutoff', None)
            anm_start.calcModes(n_modes=n_modes, **kwargs)
        else:
            # Code to read ANM files generated by ProDy
            anm_start = pd.loadModel(filename)

        basis = anm_start.getEigvecs()

        return basis

    # def readStructureData(self, filename):
    #     pd_struct = pd.parsePDB(filename, subset='ca', compressed=False)
    #     coords = pd_struct.getCoords()
    #     center_mass = np.mean(coords, axis=0)
    #     values = np.ones(coords.shape[0])
    #     return coords - center_mass, values, pd_struct

    def computeDeformationField(self, c_nma):
        batch_size_scope = tf.shape(c_nma)[0]
        U = tf.constant(self.U, dtype=tf.float32)
        d = tf.reshape(tf.matmul(U, tf.transpose(c_nma)), (-1, 3, batch_size_scope))
        return d[:, 0, :], d[:, 1, :], d[:, 2, :]

    def applyDeformationField(self, d, axis):
        coords = tf.constant(self.coords, dtype=tf.float32)
        coords_axis = tf.transpose(tf.gather(coords, axis, axis=1))
        return tf.add(coords_axis[:, None], d)

    def applyAlignmentMatrix(self, c, axis):
        c_r_1 = tf.multiply(c[0], tf.cast(tf.gather(self.r[axis], 0, axis=1), dtype=tf.float32))
        c_r_2 = tf.multiply(c[1], tf.cast(tf.gather(self.r[axis], 1, axis=1), dtype=tf.float32))
        c_r_3 = tf.multiply(c[2], tf.cast(tf.gather(self.r[axis], 2, axis=1), dtype=tf.float32))
        return self.i_sr * tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyAlignmentDeltaEuler(self, inputs, axis):
        r = euler_matrix_batch(self.rot_batch + inputs[3][:, 0],
                               self.tilt_batch + inputs[3][:, 1],
                               self.psi_batch + inputs[3][:, 2])

        c_r_1 = tf.multiply(inputs[0], tf.cast(tf.gather(r[axis], 0, axis=1), dtype=tf.float32))
        c_r_2 = tf.multiply(inputs[1], tf.cast(tf.gather(r[axis], 1, axis=1), dtype=tf.float32))
        c_r_3 = tf.multiply(inputs[2], tf.cast(tf.gather(r[axis], 2, axis=1), dtype=tf.float32))
        return self.i_sr * tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyShifts(self, c, axis):
        shifts_batch = tf.gather(self.shifts[axis], self.indexes, axis=0)
        return tf.add(tf.subtract(c, shifts_batch[None, :]), self.xmipp_origin[axis])

    def applyDeltaShifts(self, c, axis):
        shifts_batch = tf.gather(self.shifts[axis], self.indexes, axis=0) + c[1][:, axis]
        return tf.add(tf.subtract(c[0], shifts_batch[None, :]), self.xmipp_origin[axis])

    def scatterImgByPass(self, c):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(c[0])[1]

        c_x = tf.reshape(tf.transpose(c[0]), [batch_size_scope, -1, 1])
        c_y = tf.reshape(tf.transpose(c[1]), [batch_size_scope, -1, 1])
        c_sampling = tf.concat([c_y, c_x], axis=2)

        imgs = tf.zeros((batch_size_scope, self.xsize, self.xsize), dtype=tf.float32)

        bamp = tf.constant(self.values, dtype=tf.float32)

        bposf = tf.floor(c_sampling)
        bposi = tf.cast(bposf, tf.int32)
        bposf = c_sampling - bposf

        # Bilinear interpolation to provide forward mapping gradients
        bamp0 = bamp[None, :] * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp1 = bamp[None, :] * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp2 = bamp[None, :] * (bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp3 = bamp[None, :] * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])
        bampall = tf.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposall = tf.concat([bposi, bposi + (1, 0), bposi + (1, 1), bposi + (0, 1)], 1)
        # images = tf.stack([tf.tensor_scatter_nd_add(imgs[i], bposall[i], bampall[i]) for i in range(imgs.shape[0])])

        # bposf = tf.round(c_sampling)
        # bposall = tf.cast(bposf, tf.int32)
        #
        # num = tf.reduce_sum(((bposf - c_sampling) ** 2.), axis=-1)
        # sigma = 1.
        # bampall = bamp[None, :] * tf.exp(-num / (2. * sigma ** 2.))

        fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
        images = tf.map_fn(fn, [imgs, bposall, bampall], fn_output_signature=tf.float32)
        # images = tf.vectorized_map(fn, [imgs, bposall, bampall])

        images = tf.reshape(images, [-1, self.xsize, self.xsize, 1])

        return images

    def centerMassShift(self):
        coords_o = tf.constant(self.coords, dtype=tf.float32)
        coords_o_x = tf.transpose(tf.gather(coords_o, 0, axis=1))
        coords_o_y = tf.transpose(tf.gather(coords_o, 1, axis=1))
        coords_o_z = tf.transpose(tf.gather(coords_o, 2, axis=1))

        diff_x = self.def_coords[0] - coords_o_x[:, None]
        diff_y = self.def_coords[1] - coords_o_y[:, None]
        diff_z = self.def_coords[2] - coords_o_z[:, None]

        mean_diff_x = tf.reduce_mean(diff_x)
        mean_diff_y = tf.reduce_mean(diff_y)
        mean_diff_z = tf.reduce_mean(diff_z)

        cm = tf.sqrt(mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z)

        return cm
        # return tf.keras.activations.relu(cm, threshold=0.2)

    def averageDeformation(self):
        d_x = self.def_coords[0]
        d_y = self.def_coords[1]
        d_z = self.def_coords[2]

        rmsdef = tf.reduce_mean(tf.sqrt(tf.reduce_mean(d_x * d_x + d_y * d_y + d_z * d_z, axis=0)))

        return rmsdef

    # ----- -------- -----#

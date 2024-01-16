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
import mrcfile
from pathlib import Path

import tensorflow as tf

from tensorflow_toolkit.generators.generator_template import DataGeneratorBase
from tensorflow_toolkit.utils import basisDegreeVectors, computeBasis, euler_matrix_batch


class Generator(DataGeneratorBase):
    def __init__(self, L1=3, L2=2, refinePose=True, cap_def=False, **kwargs):
        super().__init__(**kwargs)

        self.refinePose = refinePose
        self.cap_def = cap_def

        # Get coords group
        mask_file = Path(Path(kwargs.get("md_file")).parent, 'mask.mrc')
        if self.ref_is_struct:
            groups, centers = None, None
        else:
            groups, centers = self.getCoordsGroup(mask_file)

        # Get Zernike3D vector size
        self.zernike_size = basisDegreeVectors(L1, L2)

        # Precompute Zernike3D basis
        self.Z = computeBasis(self.coords, L1=L1, L2=L2, r=0.5 * self.xsize,
                              groups=groups, centers=centers)

        if self.ref_is_struct:
            self.Z_atoms = computeBasis(self.atom_coords, L1=L1, L2=L2, r=0.5 * self.xsize,
                                        groups=groups, centers=centers)

        # Initialize zernike information
        size = self.zernike_size.shape[0]
        if self.metadata.isMetaDataLabel('zernikeCoefficients'):
            z_space = np.asarray([np.fromstring(item, sep=',')
                                  for item in self.metadata[:, 'zernikeCoefficients']])
            self.z_x_space = tf.constant(z_space[:, :size], dtype=tf.float32)
            self.z_y_space = tf.constant(z_space[:, size:2 * size], dtype=tf.float32)
            self.z_z_space = tf.constant(z_space[:, 2 * size:], dtype=tf.float32)
        else:
            self.z_x_space = tf.zeros((len(self.metadata), size), dtype=tf.float32)
            self.z_y_space = tf.zeros((len(self.metadata), size), dtype=tf.float32)
            self.z_z_space = tf.zeros((len(self.metadata), size), dtype=tf.float32)
        self.z_x_batch = np.zeros(self.batch_size)
        self.z_y_batch = np.zeros(self.batch_size)
        self.z_z_batch = np.zeros(self.batch_size)

        # Initialize pose information
        if refinePose:
            self.rot_batch = np.zeros(self.batch_size)
            self.tilt_batch = np.zeros(self.batch_size)
            self.psi_batch = np.zeros(self.batch_size)

        # Initial bonds and angles
        if self.ref_is_struct:
            coords = [self.atom_coords[:, 0][..., None], self.atom_coords[:, 1][..., None], self.atom_coords[:, 2][..., None]]
            self.angle0 = self.calcAngle(coords)
            self.bond0 = self.calcBond(coords)
        else:
            self.angle0 = 0.0
            self.bond0 = 0.0

    # ----- Initialization methods -----#
    def getCoordsGroup(self, mask):
        with mrcfile.open(mask) as mrc:
            indices = self.coords + self.xmipp_origin
            groups = mrc.data[indices[:, 2], indices[:, 1], indices[:, 0]]

        if np.unique(groups).size > 1:
            centers = []
            for group in np.unique(groups):
                centers.append(np.mean(self.coords[groups == group], axis=0))
            centers = np.asarray(centers)
        else:
            groups, centers = None, None

        return groups, centers
    # ----- -------- -----#


    # ----- Utils -----#

    def computeDeformationFieldVol(self, z):
        Z = tf.constant(self.Z, dtype=tf.float32)
        d = tf.matmul(Z, tf.transpose(z))
        return d

    def computeDeformationFieldAtoms(self, z):
        Z = tf.constant(self.Z_atoms, dtype=tf.float32)
        d = tf.matmul(Z, tf.transpose(z))
        return d

    def applyDeformationFieldVol(self, d, axis):
        coords = tf.constant(self.coords, dtype=tf.float32)
        coords_axis = tf.transpose(tf.gather(coords, axis, axis=1))
        return tf.add(coords_axis[:, None], d)

    def applyDeformationFieldAtoms(self, d, axis):
        coords = tf.constant(self.atom_coords, dtype=tf.float32)
        coords_axis = tf.transpose(tf.gather(coords, axis, axis=1))
        return tf.add(coords_axis[:, None], d)

    def applyAlignmentMatrix(self, c, axis):
        c_r_1 = tf.multiply(c[0], tf.cast(tf.gather(self.r[axis], 0, axis=1), dtype=tf.float32))
        c_r_2 = tf.multiply(c[1], tf.cast(tf.gather(self.r[axis], 1, axis=1), dtype=tf.float32))
        c_r_3 = tf.multiply(c[2], tf.cast(tf.gather(self.r[axis], 2, axis=1), dtype=tf.float32))
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyAlignmentDeltaEuler(self, inputs, axis):
        r = euler_matrix_batch(self.rot_batch + inputs[3][:, 0],
                               self.tilt_batch + inputs[3][:, 1],
                               self.psi_batch + inputs[3][:, 2])

        c_r_1 = tf.multiply(inputs[0], tf.cast(tf.gather(r[axis], 0, axis=1), dtype=tf.float32))
        c_r_2 = tf.multiply(inputs[1], tf.cast(tf.gather(r[axis], 1, axis=1), dtype=tf.float32))
        c_r_3 = tf.multiply(inputs[2], tf.cast(tf.gather(r[axis], 2, axis=1), dtype=tf.float32))
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

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

    def averageDeformation(self, d_f):
        d_x, d_y, d_z = d_f[0], d_f[1], d_f[2]

        rmsdef = tf.reduce_mean(tf.sqrt(tf.reduce_mean(d_x * d_x + d_y * d_y + d_z * d_z, axis=0)))

        return rmsdef

    def calcBond(self, coords):
        coords = [tf.transpose(coords[0]), tf.transpose(coords[1]), tf.transpose(coords[2])]
        coords = tf.stack(coords, axis=2)
        px = tf.gather(coords, self.connectivity[:, 0], axis=1)
        py = tf.gather(coords, self.connectivity[:, 1], axis=1)
        pz = tf.gather(coords, self.connectivity[:, 2], axis=1)
        dst_1 = tf.sqrt(tf.nn.relu(tf.reduce_sum((px - py) ** 2, axis=2)))
        dst_2 = tf.sqrt(tf.nn.relu(tf.reduce_sum((py - pz) ** 2, axis=2)))

        return tf.stack([dst_1, dst_2], axis=2)

    # def calcAngle(self, coords):
    #     coords = [tf.transpose(coords[0]), tf.transpose(coords[1]), tf.transpose(coords[2])]
    #     coords = tf.stack(coords, axis=2)
    #     p0 = tf.gather(coords, self.connectivity[:, 0], axis=1)
    #     p1 = tf.gather(coords, self.connectivity[:, 1], axis=1)
    #     p2 = tf.gather(coords, self.connectivity[:, 2], axis=1)
    #     b0 = p0 - p1
    #     b1 = p2 - p1
    #     b0 = b0 / tf.sqrt(tf.reduce_sum(b0 ** 2.0, axis=2, keepdims=True))
    #     b1 = b1 / tf.reduce_sum(b1 ** 2.0, axis=2, keepdims=True)
    #     ang = tf.reduce_sum(b0 * b1, axis=2)
    #     n0 = tf.linalg.norm(b0, axis=2) * tf.linalg.norm(b1, axis=2)
    #     ang = tf.math.divide_no_nan(ang, n0)
    #     epsilon = 1.0 - 1e-6
    #     ang = tf.minimum(tf.maximum(ang, -epsilon), epsilon)
    #     # ang = np.min(np.max(ang, axis=1), axis=0)
    #     ang = tf.acos(ang)
    #     ang *= 180 / np.pi
    #
    #     return ang

    def calcAngle(self, coords):
        coords = [tf.transpose(coords[0]), tf.transpose(coords[1]), tf.transpose(coords[2])]
        coords = tf.stack(coords, axis=2)

        p0 = tf.gather(coords, self.connectivity[:, 0], axis=1)
        p1 = tf.gather(coords, self.connectivity[:, 1], axis=1)
        p2 = tf.gather(coords, self.connectivity[:, 2], axis=1)
        p3 = tf.gather(coords, self.connectivity[:, 3], axis=1)

        # Assuming positions is a 4x3 array: [a, b, c, d]
        b1 = p1 - p0  # b - a
        b2 = p2 - p1  # c - b
        b3 = p3 - p2  # d - c

        # Compute normals
        n1 = tf.linalg.cross(b1, b2)
        n2 = tf.linalg.cross(b2, b3)

        # Normalize normals
        n1 = tf.nn.l2_normalize(n1)
        n2 = tf.nn.l2_normalize(n2)

        # Compute angle
        angle = tf.acos(tf.reduce_sum(n1 * n2, axis=2))

        # Adjust sign
        angle = tf.where(tf.reduce_sum(n2 * n1, axis=2) < 0.0, tf.abs(angle), angle)

        # Convert to degrees
        angle *= 180.0 / np.pi

        return angle

    # ----- -------- -----#

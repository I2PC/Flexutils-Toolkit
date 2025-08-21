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
import tensorflow_addons as tfa

from tensorflow_toolkit.generators.generator_template import DataGeneratorBase
from tensorflow_toolkit.utils import gramSchmidt


def random_unit_quaternion(num_samples):
    # Sampling from a normal distribution
    quaternions = np.random.normal(size=(num_samples, 4))

    # Normalize to unit quaternions
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions


def quaternion_to_rotation_matrix(quaternion):
    # Unpack the quaternion
    w, x, y, z = quaternion

    # Compute the rotation matrix
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])
    return R


class Generator(DataGeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Random numbers
        np.random.seed(0)
        self.rnd = np.zeros(4)
        # self.rnd = np.random.normal(size=120)

        # Save mask map and indices
        mask_path = Path(self.filename.parent, 'mask.mrc')
        with mrcfile.open(mask_path) as mrc:
            self.mask_map = mrc.data
            coords = np.asarray(np.where(mrc.data == 1))
            self.indices = coords.T

        # Read symmetry matrices
        sym_matrices_path = Path(self.filename.parent, "sym_matrices.npy")
        sym_matrices = np.load(sym_matrices_path)
        self.noSym = sym_matrices.shape[0]
        self.sym_matrices = tf.constant(sym_matrices, tf.float32)

        # Scale factor
        self.scale_factor = 0.5 * self.xsize

        # Coords preprocessing
        self.group_coords = self.getCoordsGroups()
        self.indices = tf.constant(self.indices[::self.step], dtype=tf.int32)
        if self.step > 1:
            self.all_coords = self.coords / self.scale_factor
        self.coords = self.coords[::self.step] / self.scale_factor
        self.values = self.values[::self.step]
        self.indices = self.indices[::self.step]
        self.total_voxels = self.coords.shape[0]

        # Initialize pose information
        self.rot_batch = np.zeros(self.batch_size, dtype=np.float32)
        self.tilt_batch = np.zeros(self.batch_size, dtype=np.float32)
        self.psi_batch = np.zeros(self.batch_size, dtype=np.float32)
        self.shifts_batch = [np.zeros(self.batch_size, dtype=np.float32), np.zeros(self.batch_size, dtype=np.float32)]
        if np.all(self.angle_rot == 0.0) and np.all(self.angle_tilt == 0.0) and np.all(self.angle_psi == 0.0):
            self.refinement = False
        else:
            self.refinement = True

        # Check if input volume
        self.hasInputVolume = not np.all(self.values == 0.0)

        # Cost functions
        cost = kwargs.get("cost")
        if cost == "mae":
            self.cost_function = tf.keras.metrics.mae
        elif cost == "mse":
            self.cost_function = tf.keras.metrics.mse
        elif cost == "corr":
            self.cost_function = self.correlation_coefficient_loss
        elif cost == "fpc":
            self.cost_function = self.fourier_phase_correlation


    # ----- Utils -----#

    def getCoordsGroups(self):
        indices = np.arange(np.sum(self.mask_map), dtype=int)
        indices = [indices[idx::self.step] for idx in range(self.step)]
        dim = max([len(v) for v in indices])
        return [np.pad(v, (0, dim - v.shape[0])) for v in indices]

    def getAllCoordsMask(self):
        split = [self.all_coords[self.group_coords[idx], ...] for idx in range(len(self.group_coords))]
        return split

    def getRandomCoordsGroup(self):
        if self.step > 1:
            idx = np.random.randint(0, len(self.group_coords), size=1)[0]
            coords_idx = self.group_coords[idx]
            self.coords = self.all_coords[coords_idx, ...]

    def applyAlignment(self, c_x, c_y, c_z, delta_angles, axis):
        # Get rotation matrix
        r = gramSchmidt(delta_angles)

        # Apply alignment
        c_r_1 = tf.multiply(c_x[None, :], tf.cast(tf.gather(r[axis], 0, axis=1), dtype=tf.float32)[:, None])
        c_r_2 = tf.multiply(c_y[None, :], tf.cast(tf.gather(r[axis], 1, axis=1), dtype=tf.float32)[:, None])
        c_r_3 = tf.multiply(c_z[None, :], tf.cast(tf.gather(r[axis], 2, axis=1), dtype=tf.float32)[:, None])
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyShifts(self, coord, delta_shifts, axis):
        shifts_batch = delta_shifts[:, axis]
        return tf.add(tf.subtract(coord, shifts_batch[:, None]), self.xmipp_origin[axis])

    def getRotatedGrid(self, angles):
        # Get coords to move
        c_x = tf.constant(self.coords[:, 0], dtype=tf.float32)
        c_y = tf.constant(self.coords[:, 1], dtype=tf.float32)
        c_z = tf.constant(self.coords[:, 2], dtype=tf.float32)

        # Apply alignment
        c_x_r = self.applyAlignment(c_x, c_y, c_z, angles, 0)
        c_y_r = self.applyAlignment(c_x, c_y, c_z, angles, 1)
        c_z_r = self.applyAlignment(c_x, c_y, c_z, angles, 2)

        return (c_x_r, c_y_r, c_z_r), (c_x[None, :], c_y[None, :], c_z[None, :])

    def scatterImgByPass(self, c):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(c[0][0])[0]

        # Apply shifts
        c_x_2d = self.applyShifts(self.scale_factor * c[0][0], c[1], 0)
        c_y_2d = self.applyShifts(self.scale_factor * c[0][1], c[1], 1)

        c_x_2d = c_x_2d[:, :, None]
        c_y_2d = c_y_2d[:, :, None]
        c_sampling = tf.concat([c_y_2d, c_x_2d], axis=2)

        imgs = tf.zeros((batch_size_scope, self.xsize, self.xsize), dtype=tf.float32)

        bamp = self.values[None, :] + c[2]

        bposf = tf.round(c_sampling)
        bposi = tf.cast(bposf, tf.int32)

        num = tf.reduce_sum(((bposf - c_sampling) ** 2.), axis=-1)
        sigma = 1.
        bamp = bamp * tf.exp(-num / (2. * sigma ** 2.))

        fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
        imgs = tf.map_fn(fn, [imgs, bposi, bamp], fn_output_signature=tf.float32)

        imgs = tf.reshape(imgs, [-1, self.xsize, self.xsize, 1])

        return imgs

    def gaussianFilterImage(self, images):
        # This method is redifined as we will fix the step values to 1 or 2 (experimental)
        # For this values, sigma=1 works better
        images = tfa.image.gaussian_filter2d(images, 3, 1)
        return images

    # ----- -------- -----#

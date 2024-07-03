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


import os
import sys

import numpy as np
import mrcfile
from pathlib import Path
from xmipp_metadata.image_handler import ImageHandler

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_toolkit.generators.generator_template import DataGeneratorBase
from tensorflow_toolkit.utils import euler_matrix_batch


### TO BE USED IN FUTURE NETWORKS ###
def trilinear_interpolation(input_volumes, query_points):
    """
    Perform trilinear interpolation on a batch of 3D volumes with coordinates ranging from
    -size//2 to size//2.

    Parameters:
    - input_volumes: A 5D tensor of shape (batch_size, depth, height, width, channels)
    - query_points: A 3D tensor of shape (batch_size, num_queries, 3) representing the
      coordinates to sample from. Each coordinate is of the form (z, y, x).

    Returns:
    - A 3D tensor of shape (batch_size, num_queries, channels) containing the interpolated values.
    """
    batch_size, depth, height, width, channels = (tf.shape(input_volumes)[0], tf.shape(input_volumes)[1],
                                                  tf.shape(input_volumes)[2], tf.shape(input_volumes)[3],
                                                  tf.shape(input_volumes)[4])
    depth, height, width = tf.cast(depth, tf.float32), tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    # Adjust query points from [-size//2, size//2] to [0, size]
    query_points = query_points + tf.stack([depth // 2, height // 2, width // 2], 0)

    # Split the query points into z, y, x coordinates
    z = query_points[..., 0]
    y = query_points[..., 1]
    x = query_points[..., 2]

    # Ensure coordinates are within bounds
    z = tf.clip_by_value(z, 0, depth - 1)
    y = tf.clip_by_value(y, 0, height - 1)
    x = tf.clip_by_value(x, 0, width - 1)

    # Get the integer and fractional parts of the coordinates
    z0 = tf.floor(z)
    y0 = tf.floor(y)
    x0 = tf.floor(x)

    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1

    z0 = tf.clip_by_value(z0, 0, depth - 1)
    y0 = tf.clip_by_value(y0, 0, height - 1)
    x0 = tf.clip_by_value(x0, 0, width - 1)

    z1 = tf.clip_by_value(z1, 0, depth - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)
    x1 = tf.clip_by_value(x1, 0, width - 1)

    z0 = tf.cast(z0, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    x0 = tf.cast(x0, tf.int32)

    z1 = tf.cast(z1, tf.int32)
    y1 = tf.cast(y1, tf.int32)
    x1 = tf.cast(x1, tf.int32)

    # Compute the interpolation weights
    dz = z - tf.cast(z0, tf.float32)
    dy = y - tf.cast(y0, tf.float32)
    dx = x - tf.cast(x0, tf.float32)

    # Gather values from the 8 corners of the cube surrounding each query point
    def gather_values(z, y, x):
        return tf.squeeze(tf.gather_nd(input_volumes,
                                       tf.stack(
                                           [tf.range(batch_size)[:, None] * tf.ones_like(z, dtype=tf.int32), z, y, x],
                                           axis=-1)))

    c000 = gather_values(z0, y0, x0)
    c001 = gather_values(z0, y0, x1)
    c010 = gather_values(z0, y1, x0)
    c011 = gather_values(z0, y1, x1)
    c100 = gather_values(z1, y0, x0)
    c101 = gather_values(z1, y0, x1)
    c110 = gather_values(z1, y1, x0)
    c111 = gather_values(z1, y1, x1)

    # Perform trilinear interpolation
    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    c = c0 * (1 - dz) + c1 * dz

    return c


class Generator(DataGeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(keepMap=True, **kwargs)

        # Save mask map and indices
        mask_path = Path(self.filename.parent, 'mask.mrc')
        values_in_mask = np.zeros((self.xsize, self.xsize, self.xsize))
        with mrcfile.open(mask_path) as mrc:
            self.mask_map = mrc.data
            coords = np.asarray(np.where(mrc.data == 1))
            self.indices = coords.T
            self.flat_indices = self.convert_indices_to_flat(self.indices)
            coords = np.asarray(np.where(mrc.data >= 0))
            self.full_indices = coords.T
            coords = np.transpose(np.asarray([coords[2, :], coords[1, :], coords[0, :]]))
            self.coords = coords - self.xmipp_origin
            combined_masks = values_in_mask + mrc.data

        # TEST: Reconstruction instead of refinement in the mask?
        inverted_mask = 1. - self.mask_map
        vol = self.vol * inverted_mask
        ImageHandler().write(vol, "masked_volume.mrc")
        ImageHandler().write(self.vol, "np_masked_volume.mrc")

        r = np.linalg.norm(self.coords, axis=1)
        indices_r = self.full_indices[r <= 0.5 * self.xsize]
        combined_masks[indices_r[:, 0], indices_r[:, 1], indices_r[:, 2]] = 1.0
        # ImageHandler().write(combined_masks, "combined_masks.mrc")
        combined_masks = combined_masks.flatten().astype(bool)
        self.coords = self.coords[combined_masks]
        self.full_indices = self.full_indices[combined_masks]
        self.values = vol.flatten()[combined_masks]
        self.values_no_masked = self.vol.flatten()[combined_masks]
        values_in_mask[self.indices[:, 0], self.indices[:, 1], self.indices[:, 2]] = 1.0
        values_in_mask = values_in_mask.flatten()[combined_masks].astype(bool)
        self.values_in_mask = tf.squeeze(tf.constant(np.argwhere(values_in_mask), dtype=tf.int32))
        self.full_voxels = self.coords.shape[0]
        self.cube = self.xsize * self.xsize * self.xsize
        self.mask = tf.squeeze(tf.constant(np.argwhere(combined_masks), dtype=tf.int32))

        # Checks for losses
        volume_path = Path(self.filename.parent, 'volume.mrc')
        if os.path.isfile(volume_path):
            self.null_ref = False
        else:
            self.null_ref = True

        # Scale factor
        self.scale_factor = 0.5 * self.xsize

        # Scale values
        volume_path = Path(self.filename.parent, 'volume.mrc')
        if os.path.isfile(volume_path):
            min_values = 0.1 * tf.reduce_min(self.values)
            max_values = 0.1 * tf.reduce_max(self.values)
            self.weight_initializer = tf.keras.initializers.RandomUniform(minval=min_values, maxval=max_values,
                                                                          seed=None)
        else:
            self.weight_initializer = "glorot_uniform"
        # self.values = -1.0 + 2.0 * (self.values - min_values) / (max_values - min_values)
        # self.values *= 0.1

        # Coords preprocessing
        self.group_coords = self.getCoordsGroups()
        self.indices = self.indices[::self.step].astype(int)
        if self.step > 1:
            self.all_coords = self.coords / self.scale_factor
        self.coords = self.coords[::self.step] / self.scale_factor
        self.total_voxels = self.indices.shape[0]

        # Initialize pose information
        self.rot_batch = np.zeros(self.batch_size)
        self.tilt_batch = np.zeros(self.batch_size)
        self.psi_batch = np.zeros(self.batch_size)
        self.shifts_batch = [np.zeros(self.batch_size), np.zeros(self.batch_size)]

        # B-Spline kernel (order 1)
        # b_spline_1d = np.asarray([0.0, 0.5, 1.0, 0.5, 1.0])
        b_spline_1d = np.asarray([0.0, 0.25, 0.5, 1.0, 0.5, 0.25, 1.0])
        b_spline_kernel = np.einsum('i,j->ij', b_spline_1d, b_spline_1d)
        self.b_spline_kernel = tf.constant(b_spline_kernel, dtype=tf.float32)[..., None, None]

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
        r = euler_matrix_batch(self.rot_batch + delta_angles[:, 0],
                               self.tilt_batch + delta_angles[:, 1],
                               self.psi_batch + delta_angles[:, 2])

        # Apply alignment
        c_r_1 = tf.multiply(c_x[None, :], tf.cast(tf.gather(r[axis], 0, axis=1), dtype=tf.float32)[:, None])
        c_r_2 = tf.multiply(c_y[None, :], tf.cast(tf.gather(r[axis], 1, axis=1), dtype=tf.float32)[:, None])
        c_r_3 = tf.multiply(c_z[None, :], tf.cast(tf.gather(r[axis], 2, axis=1), dtype=tf.float32)[:, None])
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyShifts(self, coord, delta_shifts, axis):
        shifts_batch = self.shifts_batch[axis] + delta_shifts[:, axis]
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

        return c_x_r, c_y_r, c_z_r

    def concatCoordsHet(self, c):
        coords = tf.stack(c[0], axis=2)
        coords_het = tf.concat([coords, tf.tile(c[1][:, None, :], (1, coords.shape[1], 1))], axis=2)
        return coords_het

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

        # Update values within mask
        flat_indices = tf.constant(self.flat_indices, dtype=tf.int32)[:, None]
        fn = lambda inp: tf.scatter_nd(flat_indices, inp, [self.cube])
        updates = tf.map_fn(fn, c[2], fn_output_signature=tf.float32)
        updates = tf.gather(updates, self.mask, axis=1)
        bamp = tf.tile(self.values[None, :], [batch_size_scope, 1]) + updates

        bposf = tf.round(c_sampling)
        bposi = tf.cast(bposf, tf.int32)

        num = tf.reduce_sum(((bposf - c_sampling) ** 2.), axis=-1)
        sigma = 1.
        bamp = bamp * tf.exp(-num / (2. * sigma ** 2.))

        fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
        imgs = tf.map_fn(fn, [imgs, bposi, bamp], fn_output_signature=tf.float32)

        imgs = tf.reshape(imgs, [-1, self.xsize, self.xsize, 1])

        # Create projection mask (to improve cost accuracy)
        self.mask_imgs = tf.zeros((batch_size_scope, self.xsize, self.xsize), dtype=tf.float32)
        mask_values = tf.ones([batch_size_scope, self.coords.shape[0]], dtype=tf.float32)
        self.mask_imgs = tf.map_fn(fn, [self.mask_imgs, bposi, mask_values], fn_output_signature=tf.float32)
        self.mask_imgs = tf.reshape(self.mask_imgs, [-1, self.xsize, self.xsize, 1])
        self.mask_imgs = tfa.image.gaussian_filter2d(self.mask_imgs, 3, 1)
        self.mask_imgs = tf.math.divide_no_nan(self.mask_imgs, self.mask_imgs)

        # TODO: Do we need to comply with line integral?
        # self.mask_imgs = tf.zeros((batch_size_scope, self.xsize, self.xsize), dtype=tf.float32)
        # mask_values = tf.exp(-num / (sigma ** 2.))
        # # mask_values = tf.ones([batch_size_scope, self.coords.shape[0]], dtype=tf.float32)
        # self.mask_imgs = tf.map_fn(fn, [self.mask_imgs, bposi, mask_values], fn_output_signature=tf.float32)
        # self.mask_imgs = tf.reshape(self.mask_imgs, [-1, self.xsize, self.xsize, 1])
        # self.mask_imgs = tfa.image.gaussian_filter2d(self.mask_imgs, 3, 1)
        # # self.mask_imgs = self.mask_imgs ** 2.0
        # # self.mask_imgs = tf.math.divide_no_nan(self.mask_imgs, self.mask_imgs)
        # imgs = tf.math.divide_no_nan(imgs, self.mask_imgs)

        return imgs

    def gaussianFilterImage(self, images):
        # This method is redifined as we will fix the step values to 1 or 2 (experimental)
        # For this values, sigma=1 works better
        images = tfa.image.gaussian_filter2d(images, 3, 1)
        return images

    def bSplineFilterImage(self, images):
        images = tf.nn.conv2d(images, self.b_spline_kernel, 1, "SAME")
        return images

    ### TO BE USED IN FUTURE NETWORKS ###
    def create_3d_gaussian_kernel(self, size, mean, cov):
        """Creates a batch of 3D Gaussian kernels using TensorFlow."""
        coords = tf.linspace(-tf.cast(size // 2, tf.float32), tf.cast(size // 2, tf.float32), size)
        z, y, x = tf.meshgrid(coords, coords, coords, indexing='ij')
        zyx = tf.stack([z, y, x], axis=-1)  # Shape (size, size, size, 3)
        zyx = tf.expand_dims(zyx, axis=0)  # Shape (1, size, size, size, 3)

        diff = tf.tile(zyx - mean, (tf.shape(cov)[0], 1, 1, 1, 1))  # Shape (1, size, size, size, 3)
        det = (cov[..., 0, 0] * (cov[..., 1, 1] * cov[..., 2, 2] - cov[..., 2, 1] * cov[..., 1, 2]) -
               cov[..., 1, 1] * (cov[..., 0, 1] * cov[..., 2, 2] - cov[..., 2, 0] * cov[..., 1, 2]) +
               cov[..., 2, 2] * (cov[..., 0, 1] * cov[..., 2, 1] - cov[..., 2, 0] * cov[..., 2, 1]) + 1e-6)
        inv_cov = tf.transpose(cov, (0, 2, 1)) / det[..., None, None]

        # Reshape for broadcasting
        inv_cov = tf.reshape(inv_cov, (-1, 1, 1, 1, 3, 3))
        det = tf.reshape(det, (-1, 1, 1, 1))

        # Perform einsum with correctly shaped tensors
        diff_expanded = tf.expand_dims(diff, -1)  # Shape: (1, size_x, size_y, size_z, 3, 1)
        exponent = -0.5 * tf.reduce_sum(
            diff_expanded * tf.matmul(inv_cov, diff_expanded), axis=[-2, -1]
        )  # Shape: (batch_size, size_x, size_y, size_z)

        # Compute the Gaussian kernel
        norm_factor = 1.0 / (tf.pow(2 * np.pi, 1.5) * tf.sqrt(det))
        norm_factor = tf.reshape(norm_factor, (-1, 1, 1, 1))
        gauss_kernel = norm_factor * tf.exp(exponent)

        # Normalize the kernel
        gauss_kernel /= tf.reduce_sum(gauss_kernel, axis=[1, 2, 3], keepdims=True)

        return gauss_kernel

    def project_3d_to_2d(self, kernel_3d, delta_angles):
        """Projects a 3D Gaussian kernel to 2D using a rotation matrix."""
        batch_size = tf.shape(delta_angles)[0]
        size = tf.shape(kernel_3d)[1]
        coords = tf.linspace(-tf.cast(size // 2, tf.float32), tf.cast(size // 2, tf.float32), size)
        x, y, z = tf.meshgrid(coords, coords, coords)
        xyz = tf.cast(tf.stack([x, y, z], axis=-1), tf.float32)
        xyz = tf.reshape(xyz, [-1, size * size * size, 3])  # Shape (B, size*size*size, 3)

        # Define rotation matrix
        rotation_matrix = euler_matrix_batch(self.rot_batch + delta_angles[:, 0],
                                             self.tilt_batch + delta_angles[:, 1],
                                             self.psi_batch + delta_angles[:, 2])
        rotation_matrix = tf.stack(rotation_matrix, axis=2)
        i_rotation_matrix = tf.transpose(rotation_matrix, (0, 2, 1))

        # Rotate coordinates
        r_xyz = tf.matmul(xyz, tf.transpose(i_rotation_matrix, perm=[0, 2, 1]))
        r_zyx = tf.stack([r_xyz[..., 2], r_xyz[..., 1], r_xyz[..., 0]], axis=-1)

        # Interpolate values
        kernel_3d = tf.reshape(trilinear_interpolation(kernel_3d[..., None], r_zyx),
                               [batch_size, size, size, size])

        return tf.reduce_sum(kernel_3d, axis=1)

    def convert_indices_to_flat(self, indices):
        """
        Convert 3D indices to linear indices for a flattened array.

        Args:
        indices (numpy.ndarray): A numpy array of shape (N, 3) containing the 3D indices.
        M (int): The size of the dimension of the volume.

        Returns:
        numpy.ndarray: A numpy array of shape (N,) containing the linear indices.
        """
        return indices[:, 0] * (self.xsize ** 2) + indices[:, 1] * self.xsize + indices[:, 2]

    def multivariate_gaussian_3d_filter(self, inputs):
        """Applies a 3D Gaussian filter projected to 2D to a batch of images."""
        images, kernel_size, mean, cov, delta_angles = inputs

        # Recover covariance matrix by Cholesky decomposition
        L = tf.stack([cov[:, 0], tf.zeros_like(cov[:, 0]), tf.zeros_like(cov[:, 0]),
                      cov[:, 1], cov[:, 2], tf.zeros_like(cov[:, 0]),
                      cov[:, 3], cov[:, 4], cov[:, 5]], axis=-1)
        L = tf.reshape(L, (tf.shape(cov)[0], 3, 3))
        cov = tf.matmul(L, tf.transpose(L, (0, 2, 1)))

        kernel_3d = self.create_3d_gaussian_kernel(kernel_size, mean, cov)
        kernel_2d = self.project_3d_to_2d(kernel_3d, delta_angles)
        kernel_2d = tf.transpose(kernel_2d, [1, 2, 0])[..., None]
        images = tf.transpose(images[..., 0], [1, 2, 0])[None, ...]

        # Create a depthwise convolutional layer
        gauss_filter = tf.nn.depthwise_conv2d(
            images,
            kernel_2d,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        tf.print(tf.reduce_sum(cov), output_stream=sys.stdout)
        tf.print(tf.reduce_sum(kernel_3d), output_stream=sys.stdout)
        tf.print(tf.reduce_sum(kernel_2d), output_stream=sys.stdout)
        tf.print(tf.reduce_sum(gauss_filter), output_stream=sys.stdout)

        return tf.transpose(gauss_filter[0], [2, 0, 1])[..., None]

    # ----- -------- -----#

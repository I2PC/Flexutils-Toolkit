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
    def __init__(self, L1=3, L2=2, refinePose=True, cap_def=False, strides=2, **kwargs):
        super().__init__(**kwargs)

        self.refinePose = refinePose
        self.cap_def = cap_def
        self.strides = strides

        # Get coords group
        mask_file = Path(Path(kwargs.get("md_file")).parent, 'mask.mrc')
        groups, centers = self.getCoordsGroup(mask_file)

        # Get Zernike3D vector size
        self.zernike_size = basisDegreeVectors(L1, L2)

        # Precompute Zernike3D basis
        self.Z = computeBasis(self.coords, L1=L1, L2=L2, r=0.5 * self.xsize,
                              groups=groups, centers=centers)

        # Index for finite differences
        self.order_x = np.argsort(self.coords[:, 0])
        self.order_y = np.argsort(self.coords[:, 1])
        self.order_z = np.argsort(self.coords[:, 2])

        # Initialize pose information
        if refinePose:
            self.rot_batch = np.zeros(self.batch_size)
            self.tilt_batch = np.zeros(self.batch_size)
            self.psi_batch = np.zeros(self.batch_size)


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

    def computeDeformationField(self, z):
        Z = tf.constant(self.Z, dtype=tf.float32)
        d = tf.matmul(Z, tf.transpose(z))
        return d

    def applyDeformationField(self, d, axis):
        coords = tf.constant(self.coords, dtype=tf.float32)
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

    def fieldGrad(self, d_f):
        d_x, d_y, d_z = d_f[0], d_f[1], d_f[2]

        batch_size_scope = tf.shape(d_x)[1]
        fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
        Gx = tf.zeros([batch_size_scope, self.xsize, self.xsize, self.xsize], dtype=tf.float32)
        Gy = tf.zeros([batch_size_scope, self.xsize, self.xsize, self.xsize], dtype=tf.float32)
        Gz = tf.zeros([batch_size_scope, self.xsize, self.xsize, self.xsize], dtype=tf.float32)
        
        # XYZ gradients
        coords = tf.constant(self.coords.astype(np.int32)[None, ...], dtype=tf.int32)
        coords = tf.repeat(coords, batch_size_scope, axis=0)
        Gx = tf.map_fn(fn, [Gx, coords, tf.transpose(d_x)], fn_output_signature=tf.float32)
        Gy = tf.map_fn(fn, [Gy, coords, tf.transpose(d_y)], fn_output_signature=tf.float32)
        Gz = tf.map_fn(fn, [Gz, coords, tf.transpose(d_z)], fn_output_signature=tf.float32)
        
        # MaxPool3D (for speed up purposes)
        Gx = tf.nn.max_pool3d(Gx[..., None], ksize=self.strides, strides=self.strides, padding="VALID")
        Gy = tf.nn.max_pool3d(Gy[..., None], ksize=self.strides, strides=self.strides, padding="VALID")
        Gz = tf.nn.max_pool3d(Gz[..., None], ksize=self.strides, strides=self.strides, padding="VALID")
        
        Gx = tf.abs(self.sobel_edge_3d(Gx))
        Gy = tf.abs(self.sobel_edge_3d(Gy))
        Gz = tf.abs(self.sobel_edge_3d(Gz))

        # Divergence and rotational
        divG = Gx[:, 0] + Gy[:, 1] + Gz[:, 2]
        rotGx = Gz[:, 1] - Gy[:, 2]
        rotGy = Gz[:, 2] - Gz[:, 0]
        rotGz = Gy[:, 0] - Gx[:, 1]
        
        # Gradient divergence and rotational
        G_divG = tf.abs(self.sobel_edge_3d(divG[..., None]))
        G_rotGx = tf.abs(self.sobel_edge_3d(rotGx[..., None]))
        G_rotGy = tf.abs(self.sobel_edge_3d(rotGy[..., None]))
        G_rotGz = tf.abs(self.sobel_edge_3d(rotGz[..., None]))

        # G_divG = tf.gather_nd(G_divG, coords, batch_dims=1)
        # G_rotGx = tf.gather_nd(G_rotGx, coords, batch_dims=1)
        # G_rotGy = tf.gather_nd(G_rotGy, coords, batch_dims=1)
        # G_rotGz = tf.gather_nd(G_rotGz, coords, batch_dims=1)
        # G = tf.concat([Gx[..., None], Gy[..., None], Gz[..., None]], axis=4)

        G_divG = tf.reduce_mean(G_divG)
        G_rotGx = tf.reduce_mean(G_rotGx)
        G_rotGy = tf.reduce_mean(G_rotGy)
        G_rotGz = tf.reduce_mean(G_rotGz)

        return (G_divG + G_rotGx + G_rotGy + G_rotGz) / 4.0

    # def computeGrad(self, d_x, d_y, d_z):
    #     diff_x = tf.abs(d_x[1:] - d_x[:-1])
    #     diff_y = tf.abs(d_y[1:] - d_y[:-1])
    #     diff_z = tf.abs(d_z[1:] - d_z[:-1])
    #     return tf.reduce_mean(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z, axis=0)

    def sobel_edge_3d(self, inputTensor):
        # This function computes Sobel edge maps on 3D images
        # inputTensor: input 3D images, with size of [batchsize,W,H,D,1]
        # output: output 3D edge maps, with size of [batchsize,W-2,H-2,D-2,3], each channel represents edge map in one dimension
        sobel1 = tf.constant([1, 0, -1], tf.float32)  # 1D edge filter
        sobel2 = tf.constant([1, 2, 1], tf.float32)  # 1D blur weight

        # generate sobel1 and sobel2 on x- y- and z-axis, saved in sobel1xyz and sobel2xyz
        sobel1xyz = [sobel1, sobel1, sobel1]
        sobel2xyz = [sobel2, sobel2, sobel2]
        for xyz in range(3):
            newShape = [1, 1, 1, 1, 1]
            newShape[xyz] = 3
            sobel1xyz[xyz] = tf.reshape(sobel1, newShape)
            sobel2xyz[xyz] = tf.reshape(sobel2, newShape)

        # outputTensor_x will be the Sobel edge map in x-axis
        outputTensor_x = tf.nn.conv3d(inputTensor, sobel1xyz[0], strides=[1, 1, 1, 1, 1],
                                      padding='VALID')  # edge filter in x-axis
        outputTensor_x = tf.nn.conv3d(outputTensor_x, sobel2xyz[1], strides=[1, 1, 1, 1, 1],
                                      padding='VALID')  # blur filter in y-axis
        outputTensor_x = tf.nn.conv3d(outputTensor_x, sobel2xyz[2], strides=[1, 1, 1, 1, 1],
                                      padding='VALID')  # blur filter in z-axis

        outputTensor_y = tf.nn.conv3d(inputTensor, sobel1xyz[1], strides=[1, 1, 1, 1, 1], padding='VALID')
        outputTensor_y = tf.nn.conv3d(outputTensor_y, sobel2xyz[0], strides=[1, 1, 1, 1, 1], padding='VALID')
        outputTensor_y = tf.nn.conv3d(outputTensor_y, sobel2xyz[2], strides=[1, 1, 1, 1, 1], padding='VALID')

        outputTensor_z = tf.nn.conv3d(inputTensor, sobel1xyz[2], strides=[1, 1, 1, 1, 1], padding='VALID')
        outputTensor_z = tf.nn.conv3d(outputTensor_z, sobel2xyz[0], strides=[1, 1, 1, 1, 1], padding='VALID')
        outputTensor_z = tf.nn.conv3d(outputTensor_z, sobel2xyz[1], strides=[1, 1, 1, 1, 1], padding='VALID')

        return tf.concat([outputTensor_x, outputTensor_y, outputTensor_z], 4)

    # ----- -------- -----#

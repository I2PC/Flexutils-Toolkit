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
from scipy.ndimage import gaussian_filter
import mrcfile
import os
from pathlib import Path
from xmipp_metadata.metadata import XmippMetaData

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

from tensorflow_toolkit.utils import getXmippOrigin, fft_pad, ifft_pad, full_fft_pad, full_ifft_pad


class DataGeneratorBase(tf.keras.utils.Sequence):
    def __init__(self, md_file, batch_size=32, shuffle=True, step=1, splitTrain=None,
                 radius_mask=2, smooth_mask=True, cost="corr", keepMap=False, pad_factor=2,
                 sr=1., applyCTF=1, xsize=128, mode=None):
        # Attributes
        self.step = step
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = np.arange(self.batch_size)
        self.pad_factor = pad_factor
        self.filename = Path(md_file)
        # self.cap_def = 3.

        # Read metadata
        metadata = XmippMetaData(file_name=str(md_file))
        mask, volume, structure = self.readMetadata(metadata)
        self.sr = tf.constant(sr, dtype=tf.float32)
        self.applyCTF = applyCTF
        if metadata.binaries:
            self.xsize = metadata.getMetaDataImage(0).shape[1]
        else:
            self.xsize = xsize
        self.xmipp_origin = getXmippOrigin(self.xsize)

        if os.path.isfile(volume):
            # Read volume data
            self.readVolumeData(mask, volume, keepMap)
        elif os.path.isfile(mask):
            # Read default volume data
            self.readDefaultVolumeData(mask)

        if os.path.isfile(structure):
            # Read structure data
            self.readStructureData(structure)

        # Get train dataset
        if splitTrain is not None:
            self.getTrainDataset(splitTrain)

        # Prepare CTF
        self.ctf = np.zeros([self.batch_size, self.pad_factor * self.xsize,
                             int(0.5 * self.pad_factor * self.xsize + 1)])
        # self.ctf = np.zeros([self.batch_size, self.xsize,
        #                      int(0.5 * self.xsize + 1)])

        # Prepare alignment
        self.r = [np.zeros([self.batch_size, 3]), np.zeros([self.batch_size, 3]), np.zeros([self.batch_size, 3])]

        # Prepare circular mask
        radius_mask = radius_mask * 0.5 * self.xsize
        circular_mask = self.create_circular_mask(self.xsize, self.xsize, radius_mask=radius_mask,
                                                  smooth_mask=smooth_mask)
        circular_mask = tf.constant(circular_mask, dtype=tf.float32)
        circular_mask = tf.signal.ifftshift(circular_mask[None, :, :])
        size = int(0.5 * self.xsize + 1)
        # size = int(0.5 * self.pad_factor * self.xsize + 1)
        circular_mask = tf.signal.fftshift(circular_mask[:, :, :size])
        self.circular_mask = circular_mask[0, :, :]

        # Cap deformation vals
        # self.inv_sqrt_N = tf.constant(1. / np.sqrt(self.coords.shape[0]), dtype=tf.float32)
        # self.inv_bs = tf.constant(1. / self.batch_size, dtype=tf.float32)

        # Fourier rings
        # self.getFourierRings()
        # self.radial_masks, self.spatial_freq = self.get_radial_masks()

        # Cost function
        if cost == "corr":
            self.cost_function = self.loss_correlation
        elif cost == "fpc":
            self.cost_function = self.fourier_phase_correlation
        elif cost == "mae":
            self.cost_function = self.mae
        elif cost == "mse":
            self.cost_function = self.mse

        # Generator mode
        if mode is None:
            if metadata.isMetaDataLabel("subtomo_labels"):
                self.mode = "tomo"
            else:
                self.mode = "spa"
        else:
            self.mode = mode

        # Positional encoding of subtomo labels (Tomo only)
        if self.mode == "tomo":
            unique_labels = np.unique(metadata[:, "subtomo_labels"]).astype(int)
            # self.get_sinusoid_encoding_table(len(unique_labels), 10)
            self.get_sinusoid_encoding_table(np.amax(unique_labels), 100)


    #----- Initialization methods -----#

    def readMetadata(self, metadata):
        mask = Path(self.filename.parent, 'mask.mrc')
        volume = Path(self.filename.parent, 'volume.mrc')
        structure = Path(self.filename.parent, 'structure.txt')
        self.angle_rot = tf.constant(np.asarray(metadata[:, 'angleRot']), dtype=tf.float32)
        self.angle_tilt = tf.constant(np.asarray(metadata[:, 'angleTilt']), dtype=tf.float32)
        self.angle_psi = tf.constant(np.asarray(metadata[:, 'anglePsi']), dtype=tf.float32)
        self.shift_x = tf.constant(metadata[:, 'shiftX'], dtype=tf.float32)
        self.shift_y = tf.constant(np.asarray(metadata[:, 'shiftY']), dtype=tf.float32)
        self.shift_z = tf.constant(np.zeros(self.shift_x.shape), dtype=tf.float32)
        self.shifts = [self.shift_x, self.shift_y, self.shift_z]
        self.defocusU = tf.constant(metadata[:, 'ctfDefocusU'], dtype=tf.float32)
        self.defocusV = tf.constant(metadata[:, 'ctfDefocusV'], dtype=tf.float32)
        self.defocusAngle = tf.constant(metadata[:, 'ctfDefocusAngle'], dtype=tf.float32)
        self.cs = tf.constant(metadata[:, 'ctfSphericalAberration'], dtype=tf.float32)
        self.kv = tf.constant(metadata[:, 'ctfVoltage'][0], dtype=tf.float32)
        self.file_idx = np.arange(len(metadata))

        return mask, volume, structure

    def readVolumeData(self, mask, volume, keepMap=False):
        with mrcfile.open(mask) as mrc:
            # self.xsize = mrc.data.shape[0]
            # self.xmipp_origin = getXmippOrigin(self.xsize)
            coords = np.asarray(np.where(mrc.data > 0))
            coords = np.transpose(np.asarray([coords[2, :], coords[1, :], coords[0, :]]))
            self.coords = coords - self.xmipp_origin

            if keepMap:
                self.mask_map = mrc.data

        # Apply step to coords and values
        coords = coords[::self.step]
        self.coords = self.coords[::self.step]

        with mrcfile.open(volume) as mrc:
            self.values = mrc.data[coords[:, 2], coords[:, 1], coords[:, 0]]

            if keepMap:
                self.vol = mrc.data

        # Flag (reference is map)
        self.ref_is_struct = False

    def readStructureData(self, structure):
        pdb_info = np.loadtxt(structure)

        # Get structure coords
        self.atom_coords = pdb_info[:, :-1]

        # Values for every atom
        # self.values = np.ones(self.coords.shape[0])
        # self.values = (pdb_info[:, -1]).reshape(-1)

        # Bonds
        bonds_file = str(Path(structure.parent, 'bonds.txt'))
        if os.path.isfile(bonds_file):
            self.bonds = np.loadtxt(bonds_file).astype(int)

        # Dihedrals
        dihedrals_file = str(Path(structure.parent, 'dihedrals.txt'))
        if os.path.isfile(dihedrals_file):
            self.dihedrals = np.loadtxt(dihedrals_file).astype(int)

        # CA indices
        ca_file = str(Path(structure.parent, 'ca_indices.txt'))
        if os.path.isfile(ca_file):
            self.ca_indices = tf.constant(np.loadtxt(ca_file).astype(int), tf.int32)

        # Flag (reference is structure)
        self.ref_is_struct = True

    def readDefaultVolumeData(self, mask):
        with mrcfile.open(mask) as mrc:
            coords = np.asarray(np.where(mrc.data > 0))
            coords = np.transpose(np.asarray([coords[2, :], coords[1, :], coords[0, :]]))
            self.coords = coords - self.xmipp_origin
            xsize = mrc.data.shape[0]

        # Apply step to coords and values
        self.coords = self.coords[::self.step]
        self.values = np.zeros(self.coords.shape[0], dtype=np.float32)
        self.vol = np.zeros((xsize, xsize, xsize), dtype=np.float32)

        # Flag (reference is map)
        self.ref_is_struct = False

    def getTrainDataset(self, splitTrain):
        indexes = np.arange(self.file_idx.size)
        if splitTrain > 0:
            self.file_idx = self.file_idx[indexes[:int(splitTrain * indexes.size)]]
        elif splitTrain < 0:
            self.file_idx = self.file_idx[indexes[int(splitTrain * indexes.size):]]
        else:
            raise ValueError("The variable splitTrain must not be equal to 0")

    # ----- -------- -----#


    # ----- Data generation methods -----#
    def return_tf_dataset(self):
        metadata = XmippMetaData(file_name=str(self.filename))
        images = metadata.getMetaDataImage(self.file_idx)[..., None]
        if self.mode == "tomo":
            subtomo_labels = metadata[self.file_idx, "subtomo_labels"].astype(int) - 1
            dataset = tf.data.Dataset.from_tensor_slices(((images, subtomo_labels), (self.file_idx, self.file_idx)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((images, self.file_idx))
        if self.shuffle:
            dataset = dataset.shuffle(len(self.file_idx))
        return dataset.batch(self.batch_size).prefetch(2)

    # ----- -------- -----#


    # ----- Utils -----#

    def gaussianFilterImage(self, images):
        # Both 4 and 3 are ok for the kernel size (same time and results)
        # images = tfa.image.gaussian_filter2d(images, 4 * self.step, self.step)
        images = tfa.image.gaussian_filter2d(images, 3 * self.step, self.step)
        return images

    def wiener2DFilter(self, images):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(images)[0]

        # Sizes
        pad_size = tf.constant(int(self.pad_factor * self.xsize), dtype=tf.int32)
        size = tf.constant(int(self.xsize), dtype=tf.int32)

        ctf_2 = self.ctf * self.ctf
        # epsilon = 1e-5
        epsilon = 0.1 * tf.reduce_mean(ctf_2)

        ft_images = fft_pad(images, pad_size, pad_size)
        ft_w_images_real = tf.math.real(ft_images) * self.ctf / (ctf_2 + epsilon)
        ft_w_images_imag = tf.math.imag(ft_images) * self.ctf / (ctf_2 + epsilon)
        ft_w_images = tf.complex(ft_w_images_real, ft_w_images_imag)
        w_images = ifft_pad(ft_w_images, size, size)
        return tf.reshape(w_images, [batch_size_scope, self.xsize, self.xsize, 1])

    def ctfFilterImage(self, images):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(images)[0]

        # Sizes
        pad_size = tf.constant(int(self.pad_factor * self.xsize), dtype=tf.int32)
        size = tf.constant(int(self.xsize), dtype=tf.int32)

        # ft_images = tf.signal.fftshift(tf.signal.rfft2d(images[:, :, :, 0]))
        ft_images = fft_pad(images, pad_size, pad_size)
        ft_ctf_images_real = tf.multiply(tf.math.real(ft_images), self.ctf)
        ft_ctf_images_imag = tf.multiply(tf.math.imag(ft_images), self.ctf)
        ft_ctf_images = tf.complex(ft_ctf_images_real, ft_ctf_images_imag)
        # ctf_images = tf.signal.irfft2d(tf.signal.ifftshift(ft_ctf_images))
        ctf_images = ifft_pad(ft_ctf_images, size, size)
        return tf.reshape(ctf_images, [batch_size_scope, self.xsize, self.xsize, 1])

    def resizeImageFourier(self, images, out_size):
        # Sizes
        xsize = tf.shape(images)[1]
        pad_size = self.pad_factor * xsize
        pad_out_size = self.pad_factor * out_size

        # Fourier transform
        ft_images = full_fft_pad(images, pad_size, pad_size)

        # Normalization constant
        norm = tf.cast(pad_out_size, dtype=tf.float32) / tf.cast(pad_size, dtype=tf.float32)

        # Resizing
        ft_images = tf.image.resize_with_crop_or_pad(ft_images[..., None], pad_out_size, pad_out_size)[..., 0]

        # Inverse transform
        images = full_ifft_pad(ft_images, out_size, out_size)
        images *= norm * norm

        return images

    def create_circular_mask(self, h, w, center=None, radius_mask=None, smooth_mask=True):

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius_mask is None:  # use the smallest distance between the center and image walls
            radius_mask = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = (dist_from_center <= radius_mask).astype(np.float32)

        if smooth_mask:
            mask = gaussian_filter(mask, sigma=2.)

        return mask

    def applyRealMask(self, images):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(images)[0]

        # Sizes
        pad_size = tf.constant(int(self.pad_factor * self.xsize), dtype=tf.int32)
        size = tf.constant(int(self.xsize), dtype=tf.int32)

        # ft_images = tf.signal.fftshift(tf.signal.rfft2d(images[:, :, :, 0]))
        ft_images = fft_pad(images, pad_size, pad_size)
        ft_masked_images_real = tf.multiply(tf.math.real(ft_images), self.circular_mask[None, :, :])
        ft_masked_images_imag = tf.multiply(tf.math.imag(ft_images), self.circular_mask[None, :, :])
        ft_masked_images = tf.complex(ft_masked_images_real, ft_masked_images_imag)
        # masked_images = tf.signal.irfft2d(tf.signal.ifftshift(ft_masked_images))
        masked_images = ifft_pad(ft_masked_images, size, size)
        return tf.reshape(masked_images, [batch_size_scope, self.xsize, self.xsize, 1])

    def applyFourierMask(self, ft_images):
        # FT images must be shifted with tf.signal.fftshift
        ft_masked_images_real = tf.multiply(tf.math.real(ft_images), self.circular_mask[None, :, :])
        ft_masked_images_imag = tf.multiply(tf.math.imag(ft_images), self.circular_mask[None, :, :])
        ft_masked_images = tf.complex(ft_masked_images_real, ft_masked_images_imag)
        return ft_masked_images

    # def capDeformation(self, d_x, d_y, d_z):
    #     num = tf.sqrt(tf.reduce_sum(d_x * d_x + d_y * d_y + d_z * d_z))
    #     rmsdef = self.inv_sqrt_N * self.inv_bs * num
    #     return tf.keras.activations.relu(rmsdef, threshold=self.cap_def)
    #     # return tf.math.pow(1000., rmsdef - self.cap_def)

    def getFourierRings(self):
        idx = np.indices((self.xsize, self.xsize)) - self.xsize // 2
        idx = np.fft.fftshift(idx)
        idx = idx[:, :, :self.xsize // 2 + 1]

        # self.idxft = (idx / self.xsize).astype(np.float32)
        # self.rrft = np.sqrt(np.sum(idx ** 2, axis=0)).astype(np.float32)  ## batch, npts, x-y

        rr = np.round(np.sqrt(np.sum(idx ** 2, axis=0))).astype(int)
        self.rings = np.zeros((self.xsize, self.xsize // 2 + 1, self.xsize // 2), dtype=np.float32)  #### Fourier rings
        for i in range(self.xsize // 2):
            self.rings[:, :, i] = (rr == i)

        # self.xvec = tf.constant(np.fromfunction(lambda i, j: 1.0 - 2 * ((i + j) % 2),
        #                                         (self.xsize, self.xsize // 2 + 1), dtype=np.float32))

    def radial_mask(self, r, cx=64, cy=64, sx=np.arange(0, 128), sy=np.arange(0, 128), delta=1):
        ind = (sx[np.newaxis, :] - cx) ** 2 + (sy[:, np.newaxis] - cy) ** 2
        ind1 = ind <= ((r[0] + delta) ** 2)  # one liner for this and below?
        ind2 = ind > (r[0] ** 2)
        return ind1 * ind2

    @tf.function
    def get_radial_masks(self):
        bxsize = self.xsize
        half_bxsize = self.xsize // 2
        freq_nyq = int(np.floor(int(bxsize) / 2.0))
        radii = np.arange(half_bxsize).reshape(half_bxsize, 1)  # image size 256, binning = 3
        radial_masks = np.apply_along_axis(self.radial_mask, 1, radii, half_bxsize, half_bxsize,
                                           np.arange(0, bxsize), np.arange(0, bxsize), 1)
        radial_masks = np.expand_dims(radial_masks, 1)
        radial_masks = np.expand_dims(radial_masks, 1)

        spatial_freq = radii.astype(np.float32) / freq_nyq
        spatial_freq = spatial_freq / max(spatial_freq)

        return radial_masks, spatial_freq

    def downSampleImages(self, images, size):
        return tf.image.resize(images, size=size)

    # ----- -------- -----#


    # ----- Losses -----#

    def loss_correlation(self, y_true, y_pred):
        return self.correlation_coefficient_loss(y_true, y_pred)

    def correlation_coefficient_loss(self, x, y):
        epsilon = 10e-5
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(xm * ym)
        x_square_sum = K.sum(xm * xm)
        y_square_sum = K.sum(ym * ym)
        r_den = K.sqrt(x_square_sum * y_square_sum)
        r = r_num / (r_den + epsilon)
        return 1 - K.mean(r)

    def fourier_phase_correlation(self, x, y):
        x = tf.signal.fftshift(tf.signal.rfft2d(x[:, :, :, 0]))
        y = tf.signal.fftshift(tf.signal.rfft2d(y[:, :, :, 0]))

        # In case we want to exclude some high (noisy) frequencies from the cost (using hard or
        # soft circular masks in Fourier space)
        x = self.applyFourierMask(x)
        y = self.applyFourierMask(y)

        epsilon = 10e-5
        num = tf.abs(tf.reduce_sum(x * tf.math.conj(y), axis=(1, 2)))
        d_1 = tf.reduce_sum(tf.abs(x) ** 2, axis=(1, 2))
        d_2 = tf.reduce_sum(tf.abs(y) ** 2, axis=(1, 2))
        den = tf.sqrt(d_1 * d_2)
        cross_power_spectrum = num / (den + epsilon)

        return 1 - K.mean(cross_power_spectrum)

    def frc_loss(self, y_true, y_pred, minpx=1, maxpx=-1):
        y_true = tf.signal.rfft2d(y_true[:, :, :, 0])
        y_pred = tf.signal.rfft2d(y_pred[:, :, :, 0])
        mreal, mimag = tf.math.real(y_pred), tf.math.imag(y_pred)
        dreal, dimag = tf.math.real(y_true), tf.math.imag(y_true)
        #### normalization per ring
        nrm_img = mreal ** 2 + mimag ** 2
        nrm_data = dreal ** 2 + dimag ** 2

        nrm0 = tf.tensordot(nrm_img, self.rings, [[1, 2], [0, 1]])
        nrm1 = tf.tensordot(nrm_data, self.rings, [[1, 2], [0, 1]])

        nrm = tf.sqrt(nrm0) * tf.sqrt(nrm1)
        nrm = tf.maximum(nrm, 1e-4)  #### so we do not divide by 0

        #### average FRC per batch
        ccc = mreal * dreal + mimag * dimag
        frc = tf.tensordot(ccc, self.rings, [[1, 2], [0, 1]]) / nrm

        frcval = tf.reduce_mean(frc[:, minpx:maxpx], axis=1)
        return frcval

    def mae(self, y_true, y_pred):
        cost = tf.keras.metrics.mae(y_true, y_pred)
        axis = tf.range(1, tf.rank(cost))
        return tf.reduce_mean(cost, axis=axis)

    def mse(self, y_true, y_pred):
        cost = tf.keras.metrics.mse(y_true, y_pred)
        axis = tf.range(1, tf.rank(cost))
        return tf.reduce_mean(cost, axis=axis)

    # @tf.function
    # def fourier_ring_correlation(self, image1, image2, rn, spatial_freq):
    #     # we need the channels first format for this loss
    #     image1 = tf.transpose(image1, perm=[0, 3, 1, 2])
    #     image2 = tf.transpose(image2, perm=[0, 3, 1, 2])
    #     image1 = tf.cast(image1, tf.complex64)
    #     image2 = tf.cast(image2, tf.complex64)
    #     rn = tf.cast(rn, tf.complex64)
    #     fft_image1 = tf.signal.fftshift(tf.signal.fft2d(image1), axes=[2, 3])
    #     fft_image2 = tf.signal.fftshift(tf.signal.fft2d(image2), axes=[2, 3])
    #
    #     t1 = tf.multiply(fft_image1, rn)  # (128, BS?, 3, 256, 256)
    #     t2 = tf.multiply(fft_image2, rn)
    #     c1 = tf.math.real(tf.reduce_sum(tf.multiply(t1, tf.math.conj(t2)), [2, 3, 4]))
    #     c2 = tf.reduce_sum(tf.math.abs(t1) ** 2, [2, 3, 4])
    #     c3 = tf.reduce_sum(tf.math.abs(t2) ** 2, [2, 3, 4])
    #     frc = tf.math.divide(c1, tf.math.sqrt(tf.math.multiply(c2, c3)))
    #     frc = tf.where(tf.compat.v1.is_inf(frc), tf.zeros_like(frc), frc)  # inf
    #     frc = tf.where(tf.compat.v1.is_nan(frc), tf.zeros_like(frc), frc)  # nan
    #
    #     t = spatial_freq
    #     y = frc
    #     riemann_sum = tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[:-1] + y[1:]) / 2.), 0)
    #     return riemann_sum
    #
    # @tf.function
    # def compute_loss_frc(self, y_true, y_pred):
    #     loss = -self.fourier_ring_correlation(y_true, y_pred, self.radial_masks, self.spatial_freq)
    #
    #     loss = tf.math.reduce_mean(loss)
    #     return loss

    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        """
        numpy sinusoid position encoding of Transformer model.
        params:
            n_position(n):number of positions
            d_hid(m): dimension of embedding vector
            padding_idx:set 0 dimension
        return:
            sinusoid_table(n*m):numpy array
        """

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        self.sinusoid_table = sinusoid_table

    def softThresholdImage(self, images):
        images = (images - tf.cast(images > 1e-6, dtype=tf.float32)
                  * 1e-6 + tf.cast(images < -1e-6, dtype=tf.float32)
                  * 1e-6 - tf.cast(images == 1e-6, dtype=tf.float32) * images)
        return images

    # ----- -------- -----#
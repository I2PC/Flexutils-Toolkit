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
from tensorflow.keras import layers, models, Input, Model

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import signal

from tensorflow_toolkit.utils import computeCTF
from tensorflow_toolkit.layers.siren import SIRENFirstLayerInitializer, SIRENInitializer, MetaDenseWrapper


##### Extra functions for HetSIREN network #####
def richardsonLucyDeconvolver(volume, iter=5):
    original_volume = volume.copy()
    volume = tf.constant(volume, dtype=tf.float32)
    original_volume = tf.constant(original_volume, dtype=tf.float32)

    std = np.pi * np.sqrt(volume.shape[1])
    gauss_1d = signal.windows.gaussian(volume.shape[1], std)
    kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)

    def applyKernelFourier(x):
        x = tf.cast(x, dtype=tf.complex64)
        ft_x = tf.signal.fftshift(tf.signal.fft3d(x))
        ft_x_real = tf.math.real(ft_x) * kernel
        ft_x_imag = tf.math.imag(ft_x) * kernel
        ft_x = tf.complex(ft_x_real, ft_x_imag)
        return tf.math.real(tf.signal.ifft3d(tf.signal.fftshift(ft_x)))

    for _ in range(iter):
        # Deconvolve image (update)
        conv_1 = applyKernelFourier(volume)
        conv_1_2 = conv_1 * conv_1
        epsilon = 0.1 * np.mean(conv_1_2[:])
        update = original_volume * conv_1 / (conv_1_2 + epsilon)
        update = applyKernelFourier(update)
        volume = volume * update

        volume = volume.numpy()
        thr = 1e-6
        volume = volume - (volume > thr) * thr + (volume < -thr) * thr - (volume == thr) * volume
        volume = tf.constant(volume, dtype=tf.float32)

    return volume.numpy()

def filterVol(volume):
    size = volume.shape[1]
    volume = tf.constant(volume, dtype=tf.float32)

    b_spline_1d = np.asarray([0.0, 0.5, 1.0, 0.5, 0.0])

    pad_before = (size - len(b_spline_1d)) // 2
    pad_after = size - pad_before - len(b_spline_1d)

    kernel = np.einsum('i,j,k->ijk', b_spline_1d, b_spline_1d, b_spline_1d)
    kernel = np.pad(kernel, (pad_before, pad_after), 'constant', constant_values=(0.0,))
    kernel = tf.constant(kernel, dtype=tf.complex64)
    ft_kernel = tf.abs(tf.signal.fftshift(tf.signal.fft3d(kernel)))

    def applyKernelFourier(x):
        x = tf.cast(x, dtype=tf.complex64)
        ft_x = tf.signal.fftshift(tf.signal.fft3d(x))
        ft_x_real = tf.math.real(ft_x) * ft_kernel
        ft_x_imag = tf.math.imag(ft_x) * ft_kernel
        ft_x = tf.complex(ft_x_real, ft_x_imag)
        return tf.math.real(tf.signal.ifft3d(tf.signal.fftshift(ft_x)))

    volume = applyKernelFourier(volume).numpy()
    thr = 1e-6
    volume = volume - (volume > thr) * thr + (volume < -thr) * thr - (volume == thr) * volume

    return volume


class Encoder(Model):
    def __init__(self, latent_dim, input_dim, architecture="convnn", refPose=True,
                 mode="spa"):
        super(Encoder, self).__init__()

        images = Input(shape=(input_dim, input_dim, 1))
        subtomo_pe = Input(shape=(100,))

        if architecture == "convnn":
            x = tf.keras.layers.Flatten()(images)
            x = tf.keras.layers.Dense(64 * 64)(x)
            x = tf.keras.layers.Reshape((64, 64, 1))(x)

            x = tf.keras.layers.Conv2D(4, 5, activation="relu", strides=(2, 2), padding="same")(x)
            b1_out = tf.keras.layers.Conv2D(8, 5, activation="relu", strides=(2, 2), padding="same")(x)

            b2_x = tf.keras.layers.Conv2D(8, 1, activation="relu", strides=(1, 1), padding="same")(b1_out)
            b2_x = tf.keras.layers.Conv2D(8, 1, activation="linear", strides=(1, 1), padding="same")(b2_x)
            b2_add = layers.Add()([b1_out, b2_x])
            b2_add = layers.ReLU()(b2_add)

            for _ in range(1):
                b2_x = tf.keras.layers.Conv2D(8, 1, activation="linear", strides=(1, 1), padding="same")(b2_add)
                b2_add = layers.Add()([b2_add, b2_x])
                b2_add = layers.ReLU()(b2_add)

            b2_out = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(b2_add)

            b3_x = tf.keras.layers.Conv2D(16, 1, activation="relu", strides=(1, 1), padding="same")(b2_out)
            b3_x = tf.keras.layers.Conv2D(16, 1, activation="linear", strides=(1, 1), padding="same")(b3_x)
            b3_add = layers.Add()([b2_out, b3_x])
            b3_add = layers.ReLU()(b3_add)

            for _ in range(1):
                b3_x = tf.keras.layers.Conv2D(16, 1, activation="linear", strides=(1, 1), padding="same")(b3_add)
                b3_add = layers.Add()([b3_add, b3_x])
                b3_add = layers.ReLU()(b3_add)

            b3_out = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(b3_add)
            x = tf.keras.layers.Flatten()(b3_out)

            x = layers.Flatten()(x)
            for _ in range(4):
                x = layers.Dense(256, activation='relu')(x)

        elif architecture == "mlpnn":
            x = layers.Flatten()(images)
            x = layers.Dense(1024, activation='relu')(x)
            for _ in range(2):
                x = layers.Dense(1024, activation='relu')(x)

        if mode == "spa":
            latent = layers.Dense(256, activation="relu")(x)
        elif mode == "tomo":
            latent = layers.Dense(1024, activation="relu")(subtomo_pe)
            for _ in range(2):  # TODO: Is it better to use 12 hidden layers as in Zernike3Deep?
                latent = layers.Dense(1024, activation="relu")(latent)
        for _ in range(2):
            latent = layers.Dense(256, activation="relu")(latent)
        latent = layers.Dense(latent_dim, activation="linear")(latent)  # Tanh [-1,1] as needed by SIREN?

        rows = layers.Dense(256, activation="relu", trainable=refPose)(x)
        for _ in range(2):
            rows = layers.Dense(256, activation="relu", trainable=refPose)(rows)
        rows = layers.Dense(3, activation="linear", trainable=refPose)(rows)

        shifts = layers.Dense(256, activation="relu", trainable=refPose)(x)
        for _ in range(2):
            shifts = layers.Dense(256, activation="relu", trainable=refPose)(shifts)
        shifts = layers.Dense(2, activation="linear", trainable=refPose)(shifts)

        if mode == "spa":
            self.encoder = Model(images, [rows, shifts, latent], name="encoder")
        elif mode == "tomo":
            self.encoder = Model([images, subtomo_pe], [rows, shifts, latent], name="encoder")
            self.encoder_latent = Model(subtomo_pe, latent, name="encode_latent")

    def call(self, x):
        encoded = self.encoder(x)
        return encoded


class Decoder(Model):
    def __init__(self, latent_dim, generator, CTF="apply"):
        super(Decoder, self).__init__()
        self.generator = generator
        self.CTF = CTF
        w0_first = 30.0 if generator.step == 1 else 30.0

        rows = Input(shape=(3,))
        shifts = Input(shape=(2,))
        latent = Input(shape=(latent_dim,))

        coords = layers.Lambda(self.generator.getRotatedGrid)(rows)

        # Volume decoder
        count = 0
        delta_het = MetaDenseWrapper(latent_dim, latent_dim, latent_dim, w0=w0_first,
                                     meta_kernel_initializer=SIRENFirstLayerInitializer(scale=6.0),
                                     name=f"het_{count}")(latent)  # activation=Sine(w0=1.0)
        for _ in range(3):
            count += 1
            aux = MetaDenseWrapper(latent_dim, latent_dim, latent_dim, w0=1.0,
                                   meta_kernel_initializer=SIRENInitializer(),
                                   name=f"het_{count}")(delta_het)
            delta_het = layers.Add()([delta_het, aux])
        count += 1
        delta_het = layers.Dense(self.generator.total_voxels, activation='linear',
                                 name=f"het_{count}")(delta_het)

        # Scatter image and bypass gradient
        decoded_het = layers.Lambda(self.generator.scatterImgByPass)([coords, shifts, delta_het])

        # Gaussian filter image
        decoded_het = layers.Lambda(self.generator.gaussianFilterImage)(decoded_het)

        # Soft threshold image
        decoded_het = layers.Lambda(self.generator.softThresholdImage)(decoded_het)

        if self.CTF == "apply":
            # CTF filter image
            decoded_het = layers.Lambda(self.generator.ctfFilterImage)(decoded_het)

        self.decode_het = Model(latent, delta_het, name="decoder_het")
        self.decoder = Model([rows, shifts, latent], decoded_het, name="decoder")

    def eval_volume_het(self, x_het, filter=True):
        batch_size = x_het.shape[0]

        values = self.generator.values[None, :] + self.decode_het(x_het)

        # Coords indices
        o_z, o_y, o_x = (self.generator.indices[:, 0].astype(int), self.generator.indices[:, 1].astype(int),
                         self.generator.indices[:, 2].astype(int))

        # Get numpy volumes
        values = values.numpy()
        volume_grids = np.zeros((batch_size, self.generator.xsize, self.generator.xsize, self.generator.xsize))
        for idx in range(batch_size):
            volume_grids[idx, o_z, o_y, o_x] = values[idx]
            volume_grids[idx] = volume_grids[idx] * (volume_grids[idx] >= 0.0)
            if filter:
                volume_grids[idx] = gaussian_filter(volume_grids[idx], sigma=1)
                volume_grids[idx] = filterVol(volume_grids[idx])
                volume_grids[idx] = richardsonLucyDeconvolver(volume_grids[idx])

        return volume_grids.astype(np.float32)

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class AutoEncoder(Model):
    def __init__(self, generator, het_dim=10, architecture="convnn", CTF="wiener", refPose=True,
                 l1_lambda=0.5, mode=None, train_size=None, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.mode = generator.mode if mode is None else mode
        self.xsize = generator.metadata.getMetaDataImage(0).shape[1] if generator.metadata.binaries else generator.xsize
        self.encoder = Encoder(het_dim, self.xsize, architecture=architecture,
                               refPose=refPose, mode=self.mode)
        self.decoder = Decoder(het_dim, generator, CTF=CTF)
        self.refPose = 1.0 if refPose else 0.0
        self.l1_lambda = l1_lambda
        self.het_dim = het_dim
        self.train_size = train_size if train_size is not None else self.xsize
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_loss_tracker = tf.keras.metrics.Mean(name="test_loss")
        self.loss_het_tracker = tf.keras.metrics.Mean(name="rec_het")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.test_loss_tracker,
            self.loss_het_tracker,
        ]

    def train_step(self, data):
        inputs = data[0]

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

        self.decoder.generator.indexes = indexes
        self.decoder.generator.current_images = images

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(images)[0]

        # Precompute batch aligments
        self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, indexes, axis=0)
        self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, indexes, axis=0)
        self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, indexes, axis=0)

        # Precompute batch shifts
        self.decoder.generator.shifts_batch = [tf.gather(self.decoder.generator.shift_x, indexes, axis=0),
                                               tf.gather(self.decoder.generator.shift_y, indexes, axis=0)]

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.decoder.generator.defocusU, indexes, axis=0)
        defocusV_batch = tf.gather(self.decoder.generator.defocusV, indexes, axis=0)
        defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, indexes, axis=0)
        cs_batch = tf.gather(self.decoder.generator.cs, indexes, axis=0)
        kv_batch = self.decoder.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.decoder.generator.sr, self.decoder.generator.pad_factor,
                         [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
                         batch_size_scope, self.decoder.generator.applyCTF)
        self.decoder.generator.ctf = ctf

        # Wiener filter
        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)
            if self.mode == "spa":
                inputs = images
            elif self.mode == "tomo":
                inputs[0] = images

        with tf.GradientTape() as tape:
            rows, shifts, het = self.encoder(inputs)
            decoded_het = self.decoder([self.refPose * rows, self.refPose * shifts, het])

            # L1 penalization delta_het
            delta_het = self.decoder.decode_het(het) + self.decoder.generator.values[None, :]
            l1_loss_het = tf.reduce_mean(tf.reduce_sum(tf.abs(delta_het), axis=1))
            l1_loss_het = self.l1_lambda * l1_loss_het / self.decoder.generator.total_voxels

            # Negative loss
            mask = tf.less(delta_het, 0.0)
            delta_neg = tf.boolean_mask(delta_het, mask)
            delta_neg_size = tf.cast(tf.shape(delta_neg)[-1], dtype=tf.float32)
            delta_neg = tf.reduce_mean(tf.abs(delta_neg))
            neg_loss_het = self.l1_lambda * delta_neg / delta_neg_size

            # Reconstruction mask for projections (Decoder size)
            mask_imgs = self.decoder.generator.resizeImageFourier(self.decoder.generator.mask_imgs,
                                                                  self.decoder.generator.xsize)
            mask_imgs = tf.abs(mask_imgs)
            mask_imgs = tf.math.divide_no_nan(mask_imgs, mask_imgs)

            # Reconstruction loss for original size images
            images_masked = mask_imgs * self.decoder.generator.resizeImageFourier(images, self.decoder.generator.xsize)
            loss_het_ori = self.decoder.generator.cost_function(images_masked, decoded_het)

            # Reconstruction mask for projections (Train size)
            mask_imgs = self.decoder.generator.resizeImageFourier(self.decoder.generator.mask_imgs, self.train_size)
            mask_imgs = tf.abs(mask_imgs)
            mask_imgs = tf.math.divide_no_nan(mask_imgs, mask_imgs)

            # Reconstruction loss for downscaled images
            images_masked = mask_imgs * self.decoder.generator.resizeImageFourier(images, self.train_size)
            decoded_het_scl = self.decoder.generator.resizeImageFourier(decoded_het, self.train_size)
            loss_het_scl = self.decoder.generator.cost_function(images_masked, decoded_het_scl)

            # Final losses
            rec_loss = loss_het_ori + loss_het_scl
            reg_loss = l1_loss_het + neg_loss_het

            total_loss = 0.5 * rec_loss + 0.5 * reg_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.loss_het_tracker.update_state(rec_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.loss_het_tracker.result(),
        }

    def test_step(self, data):
        inputs = data[0]

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

        self.decoder.generator.indexes = indexes
        self.decoder.generator.current_images = images

            # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(images)[0]

        # Precompute batch aligments
        self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, indexes, axis=0)
        self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, indexes, axis=0)
        self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, indexes, axis=0)

        # Precompute batch shifts
        self.decoder.generator.shifts_batch = [tf.gather(self.decoder.generator.shift_x, indexes, axis=0),
                                               tf.gather(self.decoder.generator.shift_y, indexes, axis=0)]

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.decoder.generator.defocusU, indexes, axis=0)
        defocusV_batch = tf.gather(self.decoder.generator.defocusV, indexes, axis=0)
        defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, indexes, axis=0)
        cs_batch = tf.gather(self.decoder.generator.cs, indexes, axis=0)
        kv_batch = self.decoder.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.decoder.generator.sr, self.decoder.generator.pad_factor,
                         [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
                         batch_size_scope, self.decoder.generator.applyCTF)
        self.decoder.generator.ctf = ctf

        # Wiener filter
        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)
            if self.mode == "spa":
                inputs = images
            elif self.mode == "tomo":
                inputs[0] = images

        rows, shifts, het = self.encoder(inputs)
        decoded_het = self.decoder([self.refPose * rows, self.refPose * shifts, het])

        # L1 penalization delta_het
        delta_het = self.decoder.decode_het(het) + self.decoder.generator.values[None, :]
        l1_loss_het = tf.reduce_mean(tf.reduce_sum(tf.abs(delta_het), axis=1))
        l1_loss_het = self.l1_lambda * l1_loss_het / self.decoder.generator.total_voxels

        # Negative loss
        mask = tf.less(delta_het, 0.0)
        delta_neg = tf.boolean_mask(delta_het, mask)
        delta_neg_size = tf.cast(tf.shape(delta_neg)[-1], dtype=tf.float32)
        delta_neg = tf.reduce_mean(tf.abs(delta_neg))
        neg_loss_het = self.l1_lambda * delta_neg / delta_neg_size

        # Reconstruction mask for projections (Decoder size)
        mask_imgs = self.decoder.generator.resizeImageFourier(self.decoder.generator.mask_imgs,
                                                              self.decoder.generator.xsize)
        mask_imgs = tf.abs(mask_imgs)
        mask_imgs = tf.math.divide_no_nan(mask_imgs, mask_imgs)

        # Reconstruction loss for original size images
        images_masked = mask_imgs * self.decoder.generator.resizeImageFourier(images, self.decoder.generator.xsize)
        loss_het_ori = self.decoder.generator.cost_function(images_masked, decoded_het)

        # Reconstruction mask for projections (Train size)
        mask_imgs = self.decoder.generator.resizeImageFourier(self.decoder.generator.mask_imgs, self.train_size)
        mask_imgs = tf.abs(mask_imgs)
        mask_imgs = tf.math.divide_no_nan(mask_imgs, mask_imgs)

        # Reconstruction loss for downscaled images
        images_masked = mask_imgs * self.decoder.generator.resizeImageFourier(images, self.train_size)
        decoded_het_scl = self.decoder.generator.resizeImageFourier(decoded_het, self.train_size)
        loss_het_scl = self.decoder.generator.cost_function(images_masked, decoded_het_scl)

        # Final losses
        rec_loss = loss_het_ori + loss_het_scl
        reg_loss = l1_loss_het + neg_loss_het

        total_loss = 0.5 * rec_loss + 0.5 * reg_loss

        self.total_loss_tracker.update_state(total_loss)
        self.loss_het_tracker.update_state(rec_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.loss_het_tracker.result(),
        }

    def eval_encoder(self, x):
        # Precompute batch aligments
        self.decoder.generator.rot_batch = x[1]
        self.decoder.generator.tilt_batch = x[1]
        self.decoder.generator.psi_batch = x[1]

        # Precompute batch shifts
        self.decoder.generator.shifts_batch = [x[2][:, 0], x[2][:, 1]]

        # Precompute batch CTFs
        self.decoder.generator.ctf = x[3]

        # Wiener filter
        if self.CTF == "wiener":
            x[0] = self.decoder.generator.wiener2DFilter(x[0])

        rot, shift, het = self.encoder.forward(x[0])

        return self.refPose * rot.numpy(), self.refPose * shift.numpy(), het.numpy()

    def eval_volume_het(self, x_het, allCoords=False, filter=True):
        batch_size = x_het.shape[0]

        if allCoords and self.decoder.generator.step > 1:
            new_coords, prev_coords = self.decoder.generator.getAllCoordsMask(), \
                                      self.decoder.generator.coords
        else:
            new_coords = [self.decoder.generator.coords]

        # Volume
        volume = np.zeros((batch_size, self.decoder.generator.xsize,
                           self.decoder.generator.xsize,
                           self.decoder.generator.xsize), dtype=np.float32)
        for coords in new_coords:
            self.decoder.generator.coords = coords
            volume += self.decoder.eval_volume_het(x_het, filter=filter)

        if allCoords and self.decoder.generator.step > 1:
            self.decoder.generator.coords = prev_coords

        return volume
    def predict(self, data, predict_mode="het", applyCTF=False):
        self.predict_mode, self.applyCTF = predict_mode, applyCTF
        self.predict_function = None
        decoded = super().predict(data)
        return decoded

    def predict_step(self, data):
        inputs = data[0]

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

        self.decoder.generator.indexes = indexes
        self.decoder.generator.current_images = images

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(images)[0]

        # Precompute batch aligments
        self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, indexes, axis=0)
        self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, indexes, axis=0)
        self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, indexes, axis=0)

        # Precompute batch shifts
        self.decoder.generator.shifts_batch = [tf.gather(self.decoder.generator.shift_x, indexes, axis=0),
                                               tf.gather(self.decoder.generator.shift_y, indexes, axis=0)]

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.decoder.generator.defocusU, indexes, axis=0)
        defocusV_batch = tf.gather(self.decoder.generator.defocusV, indexes, axis=0)
        defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, indexes, axis=0)
        cs_batch = tf.gather(self.decoder.generator.cs, indexes, axis=0)
        kv_batch = self.decoder.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.decoder.generator.sr, self.decoder.generator.pad_factor,
                         [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
                         batch_size_scope, self.decoder.generator.applyCTF)
        self.decoder.generator.ctf = ctf

        # Wiener filter
        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)
            if self.mode == "spa":
                inputs = images
            elif self.mode == "tomo":
                inputs[0] = images

        # Predict images with CTF applied?
        if self.applyCTF == 1:
            self.decoder.generator.CTF = "apply"
        else:
            self.decoder.generator.CTF = None

        if self.predict_mode == "het":
            return self.encoder(inputs)
        elif self.predict_mode == "particles":
            return self.decoder(self.encoder(inputs))
        else:
            raise ValueError("Prediction mode not understood!")

    def call(self, input_features):
        decoded = self.decoder(self.encoder(input_features))
        return decoded

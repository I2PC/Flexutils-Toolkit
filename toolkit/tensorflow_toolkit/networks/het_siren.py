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

from tensorflow_toolkit.utils import computeCTF
from tensorflow_toolkit.layers.siren import SIRENFirstLayerInitializer, SIRENInitializer, MetaDenseWrapper


class Encoder(Model):
    def __init__(self, latent_dim, input_dim, architecture="convnn", refPose=True):
        super(Encoder, self).__init__()

        images = Input(shape=(input_dim, input_dim, 1))

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

        latent = layers.Dense(256, activation="relu")(x)
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

        self.encoder = Model(images, [rows, shifts, latent], name="encoder")

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

        if self.CTF == "apply":
            # CTF filter image
            decoded_het = layers.Lambda(self.generator.ctfFilterImage)(decoded_het)

        self.decode_het = Model(latent, delta_het, name="decoder_het")
        self.decoder = Model([rows, shifts, latent], decoded_het, name="decoder")

    def eval_volume_het(self, x_het, filter=True):
        batch_size = x_het.shape[0]

        values = self.generator.values[None, :] + self.decode_het(x_het)

        # Coords indices
        coords = self.generator.scale_factor * self.generator.coords + self.generator.xmipp_origin
        o_x, o_y, o_z = coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)

        # Get numpy volumes
        values = values.numpy()
        volume_grids = np.zeros((batch_size, self.generator.xsize, self.generator.xsize, self.generator.xsize))
        for idx in range(batch_size):
            volume_grids[idx, o_z, o_y, o_x] = values[idx]
            if filter:
                volume_grids[idx] = gaussian_filter(volume_grids[idx], sigma=1)

        return volume_grids.astype(np.float32)

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class AutoEncoder(Model):
    def __init__(self, generator, het_dim=10, architecture="convnn", CTF="wiener", refPose=True,
                 l1_lambda=0.5, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.encoder = Encoder(het_dim, generator.xsize, architecture=architecture, refPose=refPose)
        self.decoder = Decoder(het_dim, generator, CTF=CTF)
        self.refPose = 1.0 if refPose else 0.0
        self.l1_lambda = l1_lambda
        self.het_dim = het_dim
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.loss_het_tracker = tf.keras.metrics.Mean(name="rec_het")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.loss_het_tracker,
        ]

    def train_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch aligments
        self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
        self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
        self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)

        # Precompute batch shifts
        self.decoder.generator.shifts_batch = [tf.gather(self.decoder.generator.shift_x, data[1], axis=0),
                                               tf.gather(self.decoder.generator.shift_y, data[1], axis=0)]

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.decoder.generator.defocusU, data[1], axis=0)
        defocusV_batch = tf.gather(self.decoder.generator.defocusV, data[1], axis=0)
        defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, data[1], axis=0)
        cs_batch = tf.gather(self.decoder.generator.cs, data[1], axis=0)
        kv_batch = self.decoder.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.decoder.generator.sr, self.decoder.generator.pad_factor,
                         [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
                         batch_size_scope, self.decoder.generator.applyCTF)
        self.decoder.generator.ctf = ctf

        # Wiener filter
        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)

        with tf.GradientTape() as tape:
            rows, shifts, het = self.encoder(images)
            decoded_het = self.decoder([self.refPose * rows, self.refPose * shifts, het])

            # L1 penalization delta_het
            delta_het = self.decoder.decode_het(het)
            l1_loss_het = tf.reduce_mean(tf.reduce_sum(tf.abs(delta_het), axis=1))
            l1_loss_het = self.l1_lambda * l1_loss_het / self.decoder.generator.total_voxels

            loss_het = self.decoder.generator.cost_function(images, decoded_het)

            total_loss = 0.5 * loss_het + 0.5 * l1_loss_het

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.loss_het_tracker.update_state(loss_het)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_het": self.loss_het_tracker.result(),
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
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch aligments
        self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
        self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
        self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)

        # Precompute batch shifts
        self.decoder.generator.shifts_batch = [tf.gather(self.decoder.generator.shift_x, data[1], axis=0),
                                               tf.gather(self.decoder.generator.shift_y, data[1], axis=0)]

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.decoder.generator.defocusU, data[1], axis=0)
        defocusV_batch = tf.gather(self.decoder.generator.defocusV, data[1], axis=0)
        defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, data[1], axis=0)
        cs_batch = tf.gather(self.decoder.generator.cs, data[1], axis=0)
        kv_batch = self.decoder.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.decoder.generator.sr, self.decoder.generator.pad_factor,
                         [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
                         batch_size_scope, self.decoder.generator.applyCTF)
        self.decoder.generator.ctf = ctf

        # Wiener filter
        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)

        # Predict images with CTF applied?
        if self.applyCTF == 1:
            self.decoder.generator.CTF = "apply"
        else:
            self.decoder.generator.CTF = None

        if self.predict_mode == "het":
            return self.encoder(images)
        elif self.predict_mode == "particles":
            return self.decoder(self.encoder(images))
        else:
            raise ValueError("Prediction mode not understood!")

    def call(self, input_features):
        decoded = self.decoder(self.encoder(input_features))
        return decoded

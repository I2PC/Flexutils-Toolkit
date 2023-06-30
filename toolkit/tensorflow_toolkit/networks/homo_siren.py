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

from tensorflow_toolkit.utils import computeCTF, gramSchmidt, euler_matrix_batch
from tensorflow_toolkit.layers.siren import Sine, SIRENFirstLayerInitializer, SIRENInitializer


def sigmoid_diff(x):
    return 2.0 / (1.0 + tf.exp(-x)) - 1.0

class Encoder(Model):
    def __init__(self, input_dim, architecture="convnn", refPose=True, maxAngleDiff=5., maxShiftDiff=2.):
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

        rows = layers.Dense(256, activation="relu", trainable=refPose)(x)
        for _ in range(2):
            rows = layers.Dense(256, activation="relu", trainable=refPose)(rows)
        # rows = layers.Dense(3, activation=sigmoid_diff, trainable=refPose)(rows)
        rows = layers.Dense(3, activation=Sine(w0=1.0), kernel_initializer=SIRENInitializer())(rows)
        # rows = layers.Dense(6, activation="linear", trainable=refPose)(rows)

        shifts = layers.Dense(256, activation="relu", trainable=refPose)(x)
        for _ in range(2):
            shifts = layers.Dense(256, activation="relu", trainable=refPose)(shifts)
        # shifts = layers.Dense(2, activation=sigmoid_diff, trainable=refPose)(shifts)
        shifts = layers.Dense(2, activation=Sine(w0=1.0), kernel_initializer=SIRENInitializer())(shifts)

        self.encoder = Model(images, [maxAngleDiff * rows, maxShiftDiff * shifts], name="encoder")

    def call(self, x):
        encoded = self.encoder(x)
        return encoded


class Decoder(Model):
    def __init__(self, generator, CTF="apply"):
        super(Decoder, self).__init__()
        self.generator = generator
        self.CTF = CTF
        w0_first = 30.0 if generator.step == 1 else 30.0

        rows = Input(shape=(3,))
        shifts = Input(shape=(2,))

        coords, coords_fix = layers.Lambda(self.generator.getRotatedGrid)(rows)

        # Volume decoder
        delta_vol = layers.Concatenate()(coords_fix)
        delta_vol = layers.Flatten()(delta_vol)
        delta_vol = layers.Dense(8, activation=Sine(w0=w0_first),
                                 kernel_initializer=SIRENFirstLayerInitializer(scale=6.0))(delta_vol)  # activation=Sine(w0=1.0)
        for _ in range(3):
            delta_vol = layers.Dense(8, activation=Sine(w0=1.0),
                                     kernel_initializer=SIRENInitializer())(delta_vol)
        delta_vol = layers.Dense(self.generator.total_voxels, activation='linear')(delta_vol)

        # Scatter image and bypass gradient
        decoded = layers.Lambda(self.generator.scatterImgByPass)([coords, shifts, delta_vol])

        # Gaussian filter image
        decoded = layers.Lambda(self.generator.gaussianFilterImage)(decoded)

        if self.CTF == "apply":
            # CTF filter image
            decoded = layers.Lambda(self.generator.ctfFilterImage)(decoded)

        self.decode_delta = Model(rows, delta_vol, name="delta_decoder")
        self.decoder = Model([rows, shifts], decoded, name="decoder")

    def eval_volume(self, x_rows, filter=True):
        values = self.generator.values[None, :] + self.decode_delta(x_rows)

        # Coords indices
        coords = self.generator.scale_factor * self.generator.coords + self.generator.xmipp_origin
        o_x, o_y, o_z = coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)

        # Get numpy volumes
        values = values.numpy()
        volume_grids = np.zeros((1, self.generator.xsize, self.generator.xsize, self.generator.xsize))
        volume_grids[0, o_z, o_y, o_x] = values[0]
        if filter:
            volume_grids[0] = gaussian_filter(volume_grids[0], sigma=1)

        return volume_grids.astype(np.float32)

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class AutoEncoder(Model):
    def __init__(self, generator, architecture="convnn", CTF="wiener", refPose=True,
                 l1_lambda=0.1, maxAngleDiff=5., maxShiftDiff=2., **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.encoder = Encoder(generator.xsize, architecture=architecture, refPose=refPose,
                               maxAngleDiff=maxAngleDiff, maxShiftDiff=maxShiftDiff)
        self.decoder = Decoder(generator, CTF=CTF)
        self.refPose = 1.0 if refPose else 0.0
        self.l1_lambda = l1_lambda
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
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
            rows, shifts = self.encoder(images)
            decoded = self.decoder([self.refPose * rows, self.refPose * shifts])

            # L1 penalization delta_vol
            delta_vol = self.decoder.decode_delta(rows)
            l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(delta_vol), axis=1))
            l1_loss = self.l1_lambda * l1_loss / self.decoder.generator.total_voxels

            total_loss = self.decoder.generator.cost_function(images, decoded) + l1_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result(),
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

    def eval_volume(self, allCoords=False, filter=True):
        if allCoords and self.decoder.generator.step > 1:
            new_coords, prev_coords = self.decoder.generator.getAllCoordsMask(), \
                                      self.decoder.generator.coords
        else:
            new_coords = [self.decoder.generator.coords]

        # No rot
        self.decoder.generator.rot_batch = tf.zeros((1, 1), dtype=tf.float32)
        self.decoder.generator.tilt_batch = tf.zeros((1, 1), dtype=tf.float32)
        self.decoder.generator.psi_batch = tf.zeros((1, 1), dtype=tf.float32)
        
        # Identity rows
        i_rows = np.zeros([1, 3], dtype=np.float32)
        # i_rows = np.eye(3, dtype=np.float32)[:-1, :].reshape([1, -1])

        # Volume
        volume = np.zeros((1, self.decoder.generator.xsize,
                           self.decoder.generator.xsize,
                           self.decoder.generator.xsize), dtype=np.float32)
        for coords in new_coords:
            self.decoder.generator.coords = coords
            volume += self.decoder.eval_volume(i_rows, filter=filter)

        if allCoords and self.decoder.generator.step > 1:
            self.decoder.generator.coords = prev_coords

        return volume

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

        # Rows and shifts
        rows, shifts = self.encoder(images)
        pred_imgs = self.decoder([rows, shifts])

        # Rows to matrix
        # r = gramSchmidt(rows)
        # r_ori = euler_matrix_batch(self.decoder.generator.rot_batch,
        #                            self.decoder.generator.tilt_batch,
        #                            self.decoder.generator.psi_batch)
        # rows = tf.matmul(tf.stack(r, axis=2), tf.stack(r_ori, axis=2))
        # rows = tf.stack(r_ori, axis=2)

        return rows, shifts, pred_imgs

    def call(self, input_features):
        decoded = self.decoder(self.encoder(input_features))
        return decoded

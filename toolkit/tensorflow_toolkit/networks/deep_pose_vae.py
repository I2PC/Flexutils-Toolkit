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

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from toolkit.tensorflow_toolkit.utils import computeCTF, euler_matrix_batch, gramSchmidt


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_vae_loss(images, decoded, z, mean, logvar):
    # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoded, labels=images)
    # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return logqz_x - logpz
    # return -(logpx_z + logpz - logqz_x)

    # kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
    # return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

class Encoder(tf.keras.Model):
    def __init__(self, input_dim, refine, architecture="convnn"):
        super(Encoder, self).__init__()

        encoder_inputs = tf.keras.Input(shape=(input_dim, input_dim, 1))

        x = tf.keras.layers.Flatten()(encoder_inputs)

        if architecture == "mlpnn":
            pass

        elif architecture == "convnn":

            l2 = tf.keras.regularizers.l2(1e-3)
            for _ in range(12):
                x = layers.Dense(1024, activation='relu', kernel_regularizer=l2)(x)

            y = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2)(x)
            y = layers.Dropout(0.3)(y)
            y = layers.BatchNormalization()(y)

            z = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=l2)(x)
            z = layers.Dropout(0.3)(z)
            z = layers.BatchNormalization()(z)

        algn_mean = layers.Dense(3, activation="linear")(y) if refine \
                    else layers.Dense(6, activation="linear")(y)
        algn_log = layers.Dense(3, activation="linear", kernel_initializer=tf.keras.initializers.Zeros())(y) if refine \
                   else layers.Dense(6, activation="linear", kernel_initializer=tf.keras.initializers.Zeros())(y)
        algn = layers.Concatenate()([algn_mean, algn_log])

        shifts_mean = layers.Dense(2, activation="linear")(z)
        shifts_log = layers.Dense(2, activation="linear", kernel_initializer=tf.keras.initializers.Zeros())(z)
        shifts = layers.Concatenate()([shifts_mean, shifts_log])

        self.encoder = tf.keras.Model(encoder_inputs, [algn, shifts], name="encoder")

    def call(self, x):
        return self.encoder(x)


class Decoder(tf.keras.Model):
    def __init__(self, generator):
        super(Decoder, self).__init__()
        self.generator = generator

        alignnment = tf.keras.Input(shape=(3,)) if self.generator.refinePose else tf.keras.Input(shape=(6,))
        shifts = tf.keras.Input(shape=(2,))

        # Apply alignment
        if self.generator.refinePose:
            c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentEuler(inp, 0), trainable=False)(alignnment)
            c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentEuler(inp, 1), trainable=False)(alignnment)
        else:
            c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentMatrix(inp, 0), trainable=False)(alignnment)
            c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentMatrix(inp, 1), trainable=False)(alignnment)

        # Apply shifts
        c_r_s_x = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 0), trainable=False)([c_r_x, shifts])
        c_r_s_y = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 1), trainable=False)([c_r_y, shifts])

        # Scatter image and bypass gradient
        scatter_images = layers.Lambda(self.generator.scatterImgByPass, trainable=False)([c_r_s_x, c_r_s_y])

        if self.generator.step > 1 or self.generator.ref_is_struct:
            # Gaussian filter image
            decoded = layers.Lambda(self.generator.gaussianFilterImage)(scatter_images)

            # CTF filter image
            decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(decoded)
        else:
            # CTF filter image
            decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(scatter_images)

        self.decoder = tf.keras.Model([alignnment, shifts], decoded_ctf, name="decoder")

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, architecture="convnn", n_gradients=1, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.encoder = Encoder(generator.xsize, generator.refinePose, architecture=architecture)
        self.decoder = Decoder(generator)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in
                                      self.trainable_variables]

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
        ]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
        if self.decoder.generator.refinePose:
            self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
            self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
            self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)

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

        images = data[0]
        images_rot = tfa.image.rotate(images, np.pi, interpolation="bilinear")

        images = tf.image.per_image_standardization(images)
        images_rot = tf.image.per_image_standardization(images_rot)

        with tf.GradientTape() as tape:
            ml_algn_1, ml_shift_1 = self.encode(images)
            z_algn_1 = self.reparameterize(ml_algn_1[0], ml_algn_1[1])
            z_shift_1 = self.reparameterize(ml_shift_1[0], ml_shift_1[1])
            decoded = self.decode([z_algn_1, z_shift_1])

            ml_algn_2, ml_shift_2 = self.encode(images_rot)
            z_algn_2 = self.reparameterize(ml_algn_2[0], ml_algn_2[0])
            z_shift_2 = self.reparameterize(ml_shift_2[0], ml_shift_2[0])
            decoded_rot = self.decode([z_algn_2, z_shift_2])

            vae_loss_1_algn = compute_vae_loss(images, decoded, z_algn_1, ml_algn_1[0], ml_algn_1[1])
            vae_loss_1_shift = compute_vae_loss(images, decoded, z_shift_1, ml_shift_1[0], ml_shift_1[1])
            vae_loss_1 = vae_loss_1_algn + vae_loss_1_shift

            vae_loss_2_algn = compute_vae_loss(images, decoded, z_algn_2, ml_algn_2[0], ml_algn_2[1])
            vae_loss_2_shift = compute_vae_loss(images, decoded, z_shift_2, ml_shift_2[0], ml_shift_2[1])
            vae_loss_2 = vae_loss_2_algn + vae_loss_2_shift

            vae_loss = tf.reduce_min(tf.concat([vae_loss_1[:, None], vae_loss_2[:, None]], axis=1), axis=1)

            # mse_algn = tf.losses.mse(z_algn_1, z_algn_2)
            # mse_shift = tf.losses.mse(z_shift_1, z_shift_2)
            # mse_pose = mse_algn + mse_shift

            loss_1 = self.decoder.generator.cost_function(images, decoded)
            loss_2 = self.decoder.generator.cost_function(images_rot, decoded_rot)
            loss = tf.reduce_min(tf.concat([loss_1, loss_2], axis=1), axis=1)

            total_loss = loss + vae_loss / tf.cast(batch_size_scope, tf.float32) #+ mse_pose

        # Calculate batch gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result(),
        }

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        ml_algn, ml_shift = self.encoder(x)
        mean_algn, logvar_algn = tf.split(ml_algn, num_or_size_splits=2, axis=1)
        mean_shift, logvar_shift = tf.split(ml_shift, num_or_size_splits=2, axis=1)

        # logvar_algn = tf.clip_by_value(logvar_algn, -0.08, 0.08)
        # logvar_shift = tf.clip_by_value(logvar_shift, -0.08, 0.08)

        return [[mean_algn, logvar_algn], [mean_shift, logvar_shift]]

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, input_features):
        input_features = tf.image.per_image_standardization(input_features)
        ml_algn, ml_shift = self.encode(input_features)
        z_algn = self.reparameterize(ml_algn[0], ml_algn[1])
        z_shift = self.reparameterize(ml_shift[0], ml_shift[1])
        decoded = self.decode([z_algn, z_shift])
        return decoded

    def predict_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
        if self.decoder.generator.refinePose:
            self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1],
                                                         axis=0)
            self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1],
                                                          axis=0)
            self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1],
                                                         axis=0)

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

        # Get symmetric inputs
        images = data[0]
        images_rot = tfa.image.rotate(images, np.pi, interpolation="bilinear")

        # Standarize inputs
        images = tf.image.per_image_standardization(images)
        images_rot = tf.image.per_image_standardization(images_rot)

        # Decode inputs
        ml_algn, ml_shift = self.encode(images)
        z_algn = self.reparameterize(ml_algn[0], ml_algn[1])
        z_shift = self.reparameterize(ml_shift[0], ml_shift[1])
        decoded = self.decode([z_algn, z_shift])

        # Decode rotated inputs
        ml_algn, ml_shift = self.encode(images_rot)
        z_algn_rot = self.reparameterize(ml_algn[0], ml_algn[1])
        z_shift_rot = self.reparameterize(ml_shift[0], ml_shift[1])
        decoded_rot = self.decode([z_algn, z_shift])

        # Convert to matrix if ab initio
        if not self.decoder.generator.refinePose:
            z_algn = gramSchmidt(z_algn)
            z_algn_rot = gramSchmidt(z_algn_rot)

        # Correct encoded rot
        if self.decoder.generator.refinePose:
            z_algn_rot += tf.constant([0, 0, np.pi], tf.float32)[None, :]
        else:
            A = euler_matrix_batch(tf.zeros([batch_size_scope], tf.float32),
                                   tf.zeros([batch_size_scope], tf.float32),
                                   tf.zeros([batch_size_scope], tf.float32) + 180.)
            A = tf.stack(A, axis=1)
            z_algn_rot = tf.matmul(A, z_algn_rot)

        # Stack encoder outputs
        pred_algn = tf.stack([z_algn, z_algn_rot], axis=-1)
        pred_shifts = tf.stack([z_shift, z_shift_rot], axis=-1)

        # Compute losses between symmetric inputs and predictions
        loss_1 = self.decoder.generator.cost_function(images, decoded)
        loss_2 = self.decoder.generator.cost_function(images_rot, decoded_rot)
        loss = tf.concat([loss_1, loss_2], axis=1)

        # Get only best predictions according to symmetric loss
        # index_min = tf.math.argmin(loss, axis=1)
        # pred_algn = pred_algn[tf.range(batch_size_scope, dtype=tf.int32), ..., index_min]
        # pred_shifts = pred_shifts[tf.range(batch_size_scope, dtype=tf.int32), ..., index_min]

        return pred_algn, pred_shifts, loss

# class AutoEncoder(tf.keras.Model):
#     def __init__(self, generator, architecture="convnn", **kwargs):
#         super(AutoEncoder, self).__init__(**kwargs)
#         self.generator = generator
#         self.encoder = Encoder(generator.xsize, generator.refinePose, architecture=architecture)
#         self.decoder = Decoder(generator)
#         self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#
#     @property
#     def metrics(self):
#         return [
#             self.total_loss_tracker,
#         ]
#
#     def train_step(self, data):
#         self.decoder.generator.indexes = data[1]
#         self.decoder.generator.current_images = data[0]
#
#         # Precompute batch aligments
#         if self.decoder.generator.refinePose:
#             self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
#             self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
#             self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)
#
#         # Precompute batch CTFs
#         defocusU_batch = tf.gather(self.decoder.generator.defocusU, data[1], axis=0)
#         defocusV_batch = tf.gather(self.decoder.generator.defocusV, data[1], axis=0)
#         defocusAngle_batch = tf.gather(self.decoder.generator.defocusAngle, data[1], axis=0)
#         cs_batch = tf.gather(self.decoder.generator.cs, data[1], axis=0)
#         kv_batch = self.decoder.generator.kv
#         ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch, self.decoder.generator.sr,
#                          [self.decoder.generator.xsize, int(0.5 * self.decoder.generator.xsize + 1)],
#                          self.decoder.generator.batch_size, self.decoder.generator.applyCTF)
#         self.decoder.generator.ctf = ctf
#
#         images = data[0]
#         images_rot = tfa.image.rotate(images, np.pi, interpolation="bilinear")
#
#         with tf.GradientTape() as tape:
#             encoded = self.encoder(data[0])
#             decoded = self.decoder(encoded)
#
#             encoded = self.encoder(images_rot)
#             decoded_rot = self.decoder(encoded)
#
#             loss_1 = self.decoder.generator.cost_function(images, decoded)
#             loss_2 = self.decoder.generator.cost_function(images_rot, decoded_rot)
#             total_loss = tf.reduce_min(tf.concat([loss_1, loss_2], axis=1), axis=1)
#
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)
#         return {
#             "loss": self.total_loss_tracker.result(),
#         }
#
#     def call(self, input_features):
#         return self.decoder(self.encoder(input_features))

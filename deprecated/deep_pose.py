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
import sys

import numpy as np
import scipy.stats as st

import tensorflow as tf
import tensorflow_addons as tfa
from keras.applications.vgg16 import VGG16
from keras.initializers.initializers import RandomUniform, RandomNormal
from tensorflow.keras import layers


from tensorflow_toolkit.utils import computeCTF, gramSchmidt, euler_matrix_batch, full_fft_pad, full_ifft_pad
from tensorflow_toolkit.layers.siren import Sine, SIRENInitializer


def resizeImageFourier(images, out_size, pad_factor=1):
    # Sizes
    xsize = tf.shape(images)[1]
    pad_size = pad_factor * xsize
    pad_out_size = pad_factor * out_size

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

def gaussian_kernel(size: int, std: float):
    """
    Creates a 2D Gaussian kernel with specified size and standard deviation.

    Args:
    - size: The size of the kernel (will be square).
    - std: The standard deviation of the Gaussian.

    Returns:
    - A 2D numpy array representing the Gaussian kernel.
    """
    interval = (2 * std + 1.) / size
    x = np.linspace(-std - interval / 2., std + interval / 2., size)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def create_blur_filters(num_filters, max_std, filter_size):
    """
    Create a set of Gaussian blur filters with varying standard deviations.

    Args:
    - num_filters: The number of blur filters to create.
    - max_std: The maximum standard deviation for the Gaussian blur.
    - filter_size: The size of each filter.

    Returns:
    - A tensor containing the filters.
    """
    std_intervals = np.linspace(0.1, max_std, num_filters)
    filters = []
    for std in std_intervals:
        kernel = gaussian_kernel(filter_size, std)
        kernel = np.expand_dims(kernel, axis=-1)
        filters.append(kernel)

    filters = np.stack(filters, axis=-1)
    return tf.constant(filters, dtype=tf.float32)


def apply_blur_filters_to_batch(images, filters):
    """
    Apply a set of Gaussian blur filters to a batch of images.

    Args:
    - images: Batch of images with shape (B, W, H, 1).
    - filters: Filters to apply, with shape (filter_size, filter_size, 1, N).

    Returns:
    - Batch of blurred images with shape (B, W, H, N).
    """
    # Apply the filters
    blurred_images = tf.nn.depthwise_conv2d(images, filters, strides=[1, 1, 1, 1], padding='SAME')
    return blurred_images

# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
#
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.random.normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Sampling(layers.Layer):
    def call(self, z_mean):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + 0.0 * epsilon

def correlation_coefficient_loss(y_true, y_pred):
    # Step 1: Flatten the images
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    # Step 2: Calculate the mean of each image
    mean_true = tf.reduce_mean(y_true_flat, axis=1, keepdims=True)
    mean_pred = tf.reduce_mean(y_pred_flat, axis=1, keepdims=True)

    # Step 3: Compute the covariance and variance
    y_true_centered = y_true_flat - mean_true
    y_pred_centered = y_pred_flat - mean_pred
    covariance = tf.reduce_sum(y_true_centered * y_pred_centered, axis=1)
    variance_true = tf.reduce_sum(tf.square(y_true_centered), axis=1)
    variance_pred = tf.reduce_sum(tf.square(y_pred_centered), axis=1)

    # Step 4: Calculate the correlation coefficient
    # correlation_coefficient = covariance / tf.sqrt(variance_true * variance_pred + 1e-6)
    correlation_coefficient = tf.math.divide_no_nan(covariance, tf.sqrt(variance_true * variance_pred + 1e-6))

    # Step 5: Define the loss
    loss = 1.0 - correlation_coefficient

    return loss

def safe_acos(x):
    """
    A safe version of tf.acos to avoid NaN values due to numerical issues.
    Clips the input to be within the valid range [-1, 1].
    """
    return tf.acos(tf.clip_by_value(x, -1.0 + 1e-7, 1.0 - 1e-7))

def uniform_distribution_loss(vectors):
    """
    Loss to encourage uniform distribution of pairs of vectors on a sphere.
    `vectors` is assumed to be of shape [batch_size, 3], where each row contains two 3D vectors.
    """
    batch_size = tf.shape(vectors)[0]
    batch_size_f = tf.cast(batch_size, tf.float32)

    # Compute the cosine similarity between each pair of vectors
    cosine_similarity = tf.matmul(vectors, vectors, transpose_b=True)

    # Angular distance (in radians) between vectors
    angular_distances = safe_acos(cosine_similarity)

    # Repulsion term: Use an inverse square law-like term for repulsion
    # Add a small epsilon to the angular distances to avoid division by zero
    epsilon = 1e-4
    repulsion = 1 / (angular_distances + epsilon)

    # Mask the diagonal (self-repulsion) because it's always zero and not meaningful
    mask = 1.0 - tf.eye(batch_size)
    repulsion *= mask

    # Summing up the repulsion terms and normalizing
    loss = tf.reduce_sum(repulsion) / (batch_size_f * (batch_size_f - 1.0))

    return loss


class Encoder(tf.keras.Model):
    def __init__(self, input_dim, architecture="convnn", maxAngleDiff=5., maxShiftDiff=2.):
        super(Encoder, self).__init__()
        filters = create_blur_filters(5, 5, 15)
        # init_rows = RandomUniform(minval=-20.0, maxval=20.0)
        init_shifts = RandomNormal(stddev=0.01)
        images = tf.keras.Input(shape=(input_dim, input_dim, 1))

        if architecture == "convnn":

            # x = tf.keras.layers.Flatten()(images)
            # x = tf.keras.layers.Dense(64 * 64)(x)
            # x = tf.keras.layers.Reshape((64, 64, 1))(x)
            x = resizeImageFourier(images, 64)
            x = apply_blur_filters_to_batch(x, filters)

            x = tf.keras.layers.Conv2D(64, 5, activation="relu", strides=(2, 2), padding="same")(x)
            b1_out = tf.keras.layers.Conv2D(128, 5, activation="relu", strides=(2, 2), padding="same")(x)

            b2_x = tf.keras.layers.Conv2D(128, 1, activation="relu", strides=(1, 1), padding="same")(b1_out)
            b2_x = tf.keras.layers.Conv2D(128, 1, activation="linear", strides=(1, 1), padding="same")(b2_x)
            b2_add = layers.Add()([b1_out, b2_x])
            b2_add = layers.ReLU()(b2_add)

            for _ in range(12):
                b2_x = tf.keras.layers.Conv2D(128, 1, activation="linear", strides=(1, 1), padding="same")(b2_add)
                b2_add = layers.Add()([b2_add, b2_x])
                b2_add = layers.ReLU()(b2_add)

            b2_out = tf.keras.layers.Conv2D(256, 3, activation="relu", strides=(2, 2), padding="same")(b2_add)

            b3_x = tf.keras.layers.Conv2D(256, 1, activation="relu", strides=(1, 1), padding="same")(b2_out)
            b3_x = tf.keras.layers.Conv2D(256, 1, activation="linear", strides=(1, 1), padding="same")(b3_x)
            b3_add = layers.Add()([b2_out, b3_x])
            b3_add = layers.ReLU()(b3_add)

            for _ in range(12):
                b3_x = tf.keras.layers.Conv2D(256, 1, activation="linear", strides=(1, 1), padding="same")(b3_add)
                b3_add = layers.Add()([b3_add, b3_x])
                b3_add = layers.ReLU()(b3_add)

            b3_out = tf.keras.layers.Conv2D(512, 3, activation="relu", strides=(2, 2), padding="same")(b3_add)
            x = tf.keras.layers.Flatten()(b3_out)

            x = layers.Flatten()(x)
            x = layers.Dense(512, activation='relu')(x)
            for _ in range(4):
                aux = layers.Dense(512, activation='relu')(x)
                x = layers.Add()([x, aux])

            # # x = resizeImageFourier(images, 32)
            # x = apply_blur_filters_to_batch(images, filters)
            # base_model = VGG16(input_tensor=x, weights=None, include_top=False,
            #                    input_shape=(input_dim, input_dim, 10))
            # x = base_model.output
            # x = tf.keras.layers.Flatten()(x)

        elif architecture == "mlpnn":
            x = resizeImageFourier(images, 32)
            x = layers.Flatten()(x)
            x = layers.Dense(2048, activation='relu')(x)
            aux = layers.Dense(2048, activation='relu')(x)
            x = layers.Add()([x, aux])
            for _ in range(12):
                aux = layers.Dense(2048, activation='relu')(x)
                x = layers.Add()([x, aux])
            # x = layers.Dropout(.3)(x)
            # x = layers.BatchNormalization()(x)

        rows = layers.Dense(512, activation="relu")(x)
        for _ in range(1):
            rows = layers.Dense(256, activation="relu")(rows)
        # rows = layers.Dropout(.3)(rows)
        # rows = layers.BatchNormalization()(rows)
        # rows = 360. * layers.Dense(3, activation="tanh")(rows)
        # rows = layers.Dense(3, activation=Sine(w0=1.0), kernel_initializer=SIRENInitializer())(rows)
        rows_mean = layers.Dense(6, activation="linear")(rows)
        # rows_log_var = layers.Dense(6, activation="linear")(rows)
        # rows = Sampling()([rows_mean, rows_log_var])
        rows = Sampling()(rows_mean)

        shifts = layers.Dense(512, activation="relu")(x)
        for _ in range(1):
            shifts = layers.Dense(256, activation="relu")(shifts)
        # shifts_mean = 0.25 * input_dim * layers.Dense(2, activation=Sine(w0=0.1), kernel_initializer=SIRENInitializer())(shifts)
        shifts_mean = layers.Dense(2, activation="linear")(shifts)
        # shifts_log_var = layers.Dense(2, activation="linear")(shifts)
        # shifts = Sampling()([shifts_mean, shifts_log_var])
        shifts = Sampling()(shifts_mean)

        # self.encoder = tf.keras.Model(images, [[rows_mean, rows_log_var, rows],
        #                                        [shifts_mean, shifts_log_var, shifts]], name="encoder")
        self.encoder = tf.keras.Model(images, [rows, shifts], name="encoder")

    def call(self, x):
        encoded = self.encoder(x)
        return encoded


class Decoder(tf.keras.Model):
    def __init__(self, generator, CTF="apply"):
        super(Decoder, self).__init__()
        self.generator = generator
        self.CTF = CTF

        alignment = tf.keras.Input(shape=(6,))
        shifts = tf.keras.Input(shape=(2,))

        # Apply alignment
        c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentEuler(inp, 0), trainable=True)(alignment)
        c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentEuler(inp, 1), trainable=True)(alignment)

        # Apply shifts
        c_r_s_x = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 0), trainable=True)([c_r_x, shifts])
        c_r_s_y = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 1), trainable=True)([c_r_y, shifts])

        # Scatter image and bypass gradient
        scatter_images = layers.Lambda(self.generator.scatterImgByPass, trainable=True)([c_r_s_x, c_r_s_y])

        if self.generator.step >= 1 or self.generator.ref_is_struct:
            # Gaussian filter image
            decoded = layers.Lambda(self.generator.gaussianFilterImage)(scatter_images)

            if self.CTF == "apply":
                # CTF filter image
                decoded = layers.Lambda(self.generator.ctfFilterImage)(decoded)
        else:
            if self.CTF == "apply":
                # CTF filter image
                decoded = layers.Lambda(self.generator.ctfFilterImage)(scatter_images)

        self.decoder = tf.keras.Model([alignment, shifts], decoded, name="decoder")

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, architecture="convnn", CTF="apply", n_gradients=1,
                 maxAngleDiff=5., maxShiftDiff=2., multires=True, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.multires = multires
        self.encoder = Encoder(generator.xsize, architecture=architecture,
                               maxAngleDiff=maxAngleDiff, maxShiftDiff=maxShiftDiff)
        self.decoder = Decoder(generator, CTF=CTF)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_loss_tracker = tf.keras.metrics.Mean(name="test_loss")
        self.filters = create_blur_filters(5, 5, 15)

        # self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        # self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        # self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in
        #                               self.trainable_variables]

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.test_loss_tracker,
        ]

    def train_step(self, data):
        # self.n_acum_step.assign_add(1)

        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
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

        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)

        with tf.GradientTape() as tape:
            # encoded_6d, encoded_shifts = self.encoder(images)
            # decoded = self.decoder([encoded_6d[-1], encoded_shifts[-1]])
            encoded_6d, encoded_shifts = self.encoder(images)
            decoded = self.decoder([encoded_6d, encoded_shifts])

            # # Geometric median
            # B = tf.shape(images)[0]
            # x = y = tf.range(self.decoder.generator.xsize)
            # X, Y = tf.meshgrid(x, y, indexing="xy")
            # X, Y = tf.cast(X, tf.int32), tf.cast(Y, tf.int32)
            # X, Y = tf.reshape(X, (-1, 1)), tf.reshape(Y, (-1, 1))
            # coords = tf.concat([X, Y], axis=-1)
            # coords = tf.tile(coords[None, ...], (B, 1, 1))
            # indices_B = tf.reshape(tf.range(B), [B, 1, 1])
            # indices_B = tf.tile(indices_B, [1, tf.shape(coords)[1], 1])
            # indices = tf.concat([indices_B, coords], axis=2)
            # coords = tf.cast(coords, tf.float32)
            # coords = coords - 0.5 * self.decoder.generator.xsize
            # values = tf.gather_nd(images, indices)[..., 0]
            # d = tf.square(tf.reduce_sum(coords - encoded_shifts[:, None, :], axis=-1))
            # gm = tf.reduce_sum(values * d, axis=-1) / (self.decoder.generator.xsize * self.decoder.generator.xsize)

            # Unit norm constrain
            # n1 = tf.reduce_sum(tf.square(encoded_6d[..., :3]), axis=-1)
            # n2 = tf.reduce_sum(tf.square(encoded_6d[..., 3:]), axis=-1)
            # u_norm_loss = 0.5 * (tf.reduce_mean(n1) + tf.reduce_mean(n2))

            # r = gramSchmidt(encoded_6d)
            # uniform_dist_loss = uniform_distribution_loss(r[..., -1])

            # # Multiresolution loss
            # multires_loss = 0.0
            # for mr in self.multires:
            #     images_mr = self.decoder.generator.downSampleImages(images, mr)
            #     decoded_mr = self.decoder.generator.downSampleImages(decoded, mr)
            #     multires_loss += self.decoder.generator.cost_function(images_mr, decoded_mr)
            # multires_loss = multires_loss / len(self.multires)
            #
            # total_loss = self.decoder.generator.cost_function(images, decoded) + multires_loss
            total_loss = correlation_coefficient_loss(images, decoded)  # + 0.0001 * gm

            if self.multires:
                filt_images = apply_blur_filters_to_batch(images, self.filters)
                filt_decoded = apply_blur_filters_to_batch(decoded, self.filters)
                for idx in range(5):
                    total_loss += self.decoder.generator.cost_function(filt_images[..., idx][..., None],
                                                                       filt_decoded[..., idx][..., None])
                total_loss /= 6

            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         tf.keras.losses.binary_crossentropy(images, decoded),
            #         axis=(1, 2),
            #     )
            # )
            # reconstruction_loss = self.decoder.generator.cost_function(images, decoded)
            # kl_loss_6d = -0.5 * (1.0 + encoded_6d[1] - tf.square(encoded_6d[0]) - tf.exp(encoded_6d[1]))
            # kl_loss_6d = tf.reduce_mean(tf.reduce_sum(kl_loss_6d, axis=1))
            # kl_loss_shifts = -0.5 * (1.0 + encoded_shifts[1] - tf.square(encoded_shifts[0]) - tf.exp(encoded_shifts[1]))
            # kl_loss_shifts = tf.reduce_mean(tf.reduce_sum(kl_loss_shifts, axis=1))

            # total_loss = 1. * reconstruction_loss + 1e-6 * (kl_loss_6d + kl_loss_shifts)

        # # Calculate batch gradients
        # gradients = tape.gradient(total_loss, self.trainable_variables)
        #
        # # Accumulate batch gradients
        # for i in range(len(self.gradient_accumulation)):
        #     self.gradient_accumulation[i].assign_add(gradients[i])
        #
        # # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        # tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result(),
        }

    def test_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
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

        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)

        # encoded_6d, encoded_shifts = self.encoder(images)
        # decoded = self.decoder([encoded_6d[-1], encoded_shifts[-1]])
        encoded_6d, encoded_shifts = self.encoder(images)
        decoded = self.decoder([encoded_6d, encoded_shifts])
        decoded = decoded + tf.random.normal(stddev=10.0, shape=tf.shape(decoded))

        # # Multiresolution loss
        # multires_loss = 0.0
        # for mr in self.multires:
        #     images_mr = self.decoder.generator.downSampleImages(images, mr)
        #     decoded_mr = self.decoder.generator.downSampleImages(decoded, mr)
        #     multires_loss += self.decoder.generator.cost_function(images_mr, decoded_mr)
        # multires_loss = multires_loss / len(self.multires)
        #
        # total_loss = self.decoder.generator.cost_function(images, decoded) + multires_loss
        total_loss = self.decoder.generator.cost_function(images, decoded)

        # reconstruction_loss = tf.reduce_mean(
        #     tf.reduce_sum(
        #         tf.keras.losses.binary_crossentropy(images, decoded),
        #         axis=(1, 2, 3),
        #     )
        # )
        # kl_loss_6d = -0.5 * (1 + encoded_6d[1] - tf.square(encoded_6d[0]) - tf.exp(encoded_6d[1]))
        # kl_loss_6d = tf.reduce_mean(tf.reduce_sum(kl_loss_6d, axis=1))
        # kl_loss_shifts = -0.5 * (1 + encoded_shifts[1] - tf.square(encoded_shifts[0]) - tf.exp(encoded_shifts[1]))
        # kl_loss_shifts = tf.reduce_mean(tf.reduce_sum(kl_loss_shifts, axis=1))
        #
        # total_loss = 1. * reconstruction_loss + kl_loss_6d + kl_loss_shifts

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

    def call(self, input_features):
        # encoded_6d, encoded_shifts = self.encoder(input_features)
        # decoded = self.decoder([encoded_6d[-1], encoded_shifts[-1]])
        encoded_6d, encoded_shifts = self.encoder(input_features)
        decoded = self.decoder([encoded_6d, encoded_shifts])
        return decoded

    def predict_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
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

        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)

        # O = euler_matrix_batch(self.decoder.generator.rot_batch,
        #                        self.decoder.generator.tilt_batch,
        #                        self.decoder.generator.psi_batch)
        # O = tf.cast(O, tf.float32)

        # Decode inputs
        # encoded_6d, encoded_shifts = self.encoder(images)
        # encoded = [encoded_6d[-1], encoded_shifts[-1]]

        encoded = self.encoder(images)
        encoded[0] = gramSchmidt(encoded[0])
        # encoded[1] *= 0.0

        return encoded

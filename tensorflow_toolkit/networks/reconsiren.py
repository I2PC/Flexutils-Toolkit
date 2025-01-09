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
import tensorflow_addons as tfa
from keras.initializers import RandomUniform, RandomNormal, Zeros, Constant, Orthogonal
from tensorflow.keras import layers, Input, Model

import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats as st

from tensorflow_toolkit.utils import computeCTF, gramSchmidt, euler_matrix_batch, full_fft_pad, full_ifft_pad, \
    quaternion_to_rotation_matrix
from tensorflow_toolkit.layers.siren import Sine, SIRENFirstLayerInitializer, SIRENInitializer


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


def total_variation_loss(volume, diff1, diff2, diff3):
    """
    Computes the Total Variation Loss.
    Encourages spatial smoothness in the image output.

    Parameters:
    volume (Tensor): The image tensor of shape (batch_size, depth, height, width)
    diff1 (Tensor): Voxel value differences of shape (batch_size, depth - 1, height, width)
    diff2 (Tensor): Voxel value differences of shape (batch_size, depth, height - 1, width)
    diff3 (Tensor): Voxel value differences of shape (batch_size, depth, height, width - 1)

    Returns:
    Tensor: The total variation loss.
    """

    # Sum for both directions.
    sum_axis = [1, 2, 3]
    loss = tf.reduce_sum(tf.abs(diff1), axis=sum_axis) + \
           tf.reduce_sum(tf.abs(diff2), axis=sum_axis) + \
           tf.reduce_sum(tf.abs(diff3), axis=sum_axis)

    # Normalize by the volume size
    num_pixels = tf.cast(tf.reduce_prod(volume.shape[1:]), tf.float32)
    loss /= num_pixels

    return loss


def mse_smoothness_loss(volume, diff1, diff2, diff3):
    """
    Computes an MSE-based smoothness loss.
    This loss penalizes large intensity differences between adjacent pixels.

    Parameters:
    volume (Tensor): The image tensor of shape (batch_size, depth, height, width)
    diff1 (Tensor): Voxel value differences of shape (batch_size, depth - 1, height, width)
    diff2 (Tensor): Voxel value differences of shape (batch_size, depth, height - 1, width)
    diff3 (Tensor): Voxel value differences of shape (batch_size, depth, height, width - 1)

    Returns:
    Tensor: The MSE-based smoothness loss.
    """

    # Square differences
    diff1 = tf.square(diff1)
    diff2 = tf.square(diff2)
    diff3 = tf.square(diff3)

    # Sum the squared differences
    sum_axis = [1, 2, 3]
    loss = tf.reduce_sum(diff1, axis=sum_axis) + tf.reduce_sum(diff2, axis=sum_axis) + tf.reduce_sum(diff3,
                                                                                                     axis=sum_axis)

    # Normalize by the number of pixel pairs
    num_pixel_pairs = tf.cast(2 * tf.reduce_prod(volume.shape[1:3]) - volume.shape[1] - volume.shape[2], tf.float32)
    loss /= num_pixel_pairs

    return loss


def densitySmoothnessVolume(xsize, indices, values):
    grid = tf.zeros((1, xsize, xsize, xsize), dtype=tf.float32)
    indices = tf.cast(indices[None, ...], dtype=tf.int32)

    # Scatter in volumes
    fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
    grid = tf.map_fn(fn, [grid, indices, values], fn_output_signature=tf.float32)

    # Calculate the differences of neighboring pixel-values.
    # The total variation loss is the sum of absolute differences of neighboring pixels
    # in both dimensions.
    pixel_diff1 = grid[:, 1:, :, :] - grid[:, :-1, :, :]
    pixel_diff2 = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    pixel_diff3 = grid[:, :, :, 1:] - grid[:, :, :, :-1]

    # Compute total variation and density MSE losses
    return (total_variation_loss(grid, pixel_diff1, pixel_diff2, pixel_diff3),
            mse_smoothness_loss(grid, pixel_diff1, pixel_diff2, pixel_diff3))


def connected_component_penalty(xsize, indices, values):
    threshold = tf.reduce_max(values)

    grid = tf.zeros((1, xsize, xsize, xsize), dtype=tf.float32)
    indices = tf.cast(indices[None, ...], dtype=tf.int32)

    # Scatter in volumes
    fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
    grid = tf.map_fn(fn, [grid, indices, values], fn_output_signature=tf.float32)[0, ..., None]

    # Step 1: Threshold the prediction to get a binary mask
    binary_mask = tf.cast(grid > threshold, tf.float32)

    # Step 2: Create 3D convolution filters to detect connected components
    kernel = tf.constant(
        [[[[[0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0]],

           [[1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]],

           [[0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0]]]]], dtype=tf.float32)

    # Apply the convolution to count neighbors
    convolved = tf.nn.conv3d(binary_mask, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')

    # Step 3: Create a mask for isolated components (no neighbors)
    isolated_components = tf.cast(convolved < 1.5, tf.float32) * binary_mask

    # Sum to find number of isolated components (loosely connected components)
    num_isolated_components = tf.reduce_sum(isolated_components)

    # Step 4: Penalize more than one component
    penalty = tf.maximum(0.0, num_isolated_components - 1)

    return penalty


def diversity_loss(y_pred, alpha=1.0):
    # Calculate the mean of the predictions
    mean_pred = tf.reduce_mean(y_pred, axis=0)

    # Calculate the variance across the batch
    variance = tf.reduce_mean(tf.square(y_pred - mean_pred), axis=0)

    # Encourage higher variance (diversity)
    diversity_loss = -tf.reduce_mean(variance)
    return alpha * diversity_loss


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


def l1_distance_norm(volumes, coords):
    B = tf.shape(volumes)[0]
    total_mass = tf.reduce_sum(tf.abs(volumes), axis=1)
    r = tf.tile(tf.reduce_sum(coords * coords, axis=1)[None, :], (B, 1))
    l1_dist = tf.reduce_sum(tf.abs(r * volumes), axis=1)
    return 0.01 * l1_dist / (tf.cast(tf.shape(coords)[1], tf.float32) * total_mass)


class QuaternionLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # The network predicts a 4D vector for the quaternion
        q = inputs

        # Normalize the quaternion to ensure it represents a valid rotation
        q_normalized = tf.linalg.l2_normalize(q, axis=-1)

        return q_normalized


class CommonEncoder(Model):
    def __init__(self, input_dim, architecture="convnn"):
        super(CommonEncoder, self).__init__()
        filters = create_blur_filters(10, 10, 30)

        images = Input(shape=(input_dim, input_dim, 1))

        if architecture == "convnn":

            x = tf.keras.layers.Reshape((input_dim, input_dim, 1))(images)

            x = layers.Lambda(lambda y: resizeImageFourier(y, 64))(x)

            x = layers.Lambda(lambda y: apply_blur_filters_to_batch(y, filters))(x)

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
            x = layers.Dense(1024, activation='relu')(x)
            for _ in range(3):
                aux = layers.Dense(1024, activation='relu')(x)
                x = layers.Add()([x, aux])

        elif architecture == "mlpnn":
            x = layers.Lambda(lambda y: resizeImageFourier(y, 64))(images)
            x = layers.Flatten()(x)
            x = layers.Dense(1024, activation='relu')(x)
            aux = layers.Dense(1024, activation='relu')(x)
            x = layers.Add()([x, aux])
            for _ in range(12):
                aux = layers.Dense(1024, activation='relu')(x)
                x = layers.Add()([x, aux])

        self.encoder = tf.keras.Model(images, x)

    def call(self, x):
        encoded = self.encoder(x)
        return encoded

class HeadEncoder(Model):
    def __init__(self, refinement=False, suffix="1", useQuaternions=False):
        super(HeadEncoder, self).__init__()
        if useQuaternions:
            bias_values = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            bias_values = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        x = Input(shape=(1024,))

        rows = layers.Dense(1024, activation="relu")(x)
        for _ in range(3):
            rows = layers.Dense(1024, activation="relu")(rows)
        if refinement:
            if useQuaternions:
                rows = layers.Dense(4, activation="linear", kernel_initializer=Zeros(),
                                    bias_initializer=Constant(bias_values))(rows)
                rows = QuaternionLayer()(rows)
            else:
                rows = layers.Dense(6, activation="linear", kernel_initializer=Zeros(),
                                    bias_initializer=Constant(bias_values))(rows)
        else:
            if useQuaternions:
                rows = layers.Dense(4, activation="linear")(rows)
                rows = QuaternionLayer()(rows)
            else:
                rows = layers.Dense(6, activation="linear")(rows)

        shifts = layers.Dense(1024, activation="relu")(x)
        for _ in range(3):
            shifts = layers.Dense(1024, activation="relu")(shifts)
        shifts = layers.Dense(2, activation="linear", kernel_initializer=RandomNormal(stddev=0.0001))(shifts)

        self.encoder = tf.keras.Model(x, [rows, shifts])

    def call(self, x):
        encoded = self.encoder(x)
        return encoded


class HetEncoder(Model):
    def __init__(self, input_dim, latDim=8):
        super(HetEncoder, self).__init__()
        images = Input(shape=(input_dim, input_dim, 1))

        x = layers.Flatten()(images)
        x = layers.Dense(64 * 64)(x)
        x = layers.Reshape((64, 64, 1))(x)

        x = layers.Conv2D(4, 5, activation="relu", strides=(2, 2), padding="same")(x)
        b1_out = layers.Conv2D(8, 5, activation="relu", strides=(2, 2), padding="same")(x)

        b2_x = layers.Conv2D(8, 1, activation="relu", strides=(1, 1), padding="same")(b1_out)
        b2_x = layers.Conv2D(8, 1, activation="linear", strides=(1, 1), padding="same")(b2_x)
        b2_add = layers.Add()([b1_out, b2_x])
        b2_add = layers.ReLU()(b2_add)

        for _ in range(1):
            b2_x = layers.Conv2D(8, 1, activation="linear", strides=(1, 1), padding="same")(b2_add)
            b2_add = layers.Add()([b2_add, b2_x])
            b2_add = layers.ReLU()(b2_add)

        b2_out = layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(b2_add)

        b3_x = layers.Conv2D(16, 1, activation="relu", strides=(1, 1), padding="same")(b2_out)
        b3_x = layers.Conv2D(16, 1, activation="linear", strides=(1, 1), padding="same")(b3_x)
        b3_add = layers.Add()([b2_out, b3_x])
        b3_add = layers.ReLU()(b3_add)

        for _ in range(1):
            b3_x = layers.Conv2D(16, 1, activation="linear", strides=(1, 1), padding="same")(b3_add)
            b3_add = layers.Add()([b3_add, b3_x])
            b3_add = layers.ReLU()(b3_add)

        b3_out = layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(b3_add)
        x = layers.Flatten()(b3_out)

        x = layers.Flatten()(x)
        for _ in range(4):
            x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(latDim, activation="linear")(x)

        latent = layers.Dense(256, activation="relu")(x)
        for _ in range(2):
            latent = layers.Dense(256, activation="relu")(latent)
        latent = layers.Dense(latDim, activation="linear")(latent)

        self.encoder = tf.keras.Model(images, latent)

    def call(self, x):
        encoded = self.encoder(x)
        return encoded


class Decoder(Model):
    def __init__(self, total_voxels, CTF="apply", only_pos=False):
        super(Decoder, self).__init__()
        self.CTF = CTF

        coords = Input(shape=(total_voxels, 3,))

        # Volume decoder
        delta_vol = layers.Flatten()(coords)
        delta_vol = layers.Dense(10, activation=Sine(w0=1.0),
                                 kernel_initializer=SIRENFirstLayerInitializer(scale=1.0))(delta_vol)
        for _ in range(3):
            delta_vol = layers.Dense(10, activation=Sine(w0=1.0),
                                     kernel_initializer=SIRENInitializer(c=1.0))(delta_vol)
        if not only_pos:
            delta_vol = layers.Dense(total_voxels, activation='linear')(delta_vol)  # If input volume, give near zero init?
        else:
            delta_vol = layers.Dense(total_voxels, activation='relu')(delta_vol)  # For classes works fine

        self.decoder = tf.keras.Model(coords, delta_vol)

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class HetDecoder(Model):
    def __init__(self, generator, latDim=8, num_neurons=8):
        super(HetDecoder, self).__init__()

        latent = Input(shape=(latDim, ))

        delta_het = layers.Dense(num_neurons, activation=Sine(30.0),
                                 kernel_initializer=SIRENFirstLayerInitializer(scale=1.0))(latent)
        for _ in range(3):
            aux = layers.Dense(num_neurons, activation=Sine(1.0),
                               kernel_initializer=SIRENInitializer())(latent)
            delta_het = layers.Add()([delta_het, aux])
        delta_het = layers.Dense(generator.total_voxels, activation="linear",
                                 kernel_initializer=RandomUniform(-1e-5, 1e-5))(delta_het)

        self.delta_decoder = tf.keras.Model(latent, delta_het)

    def call(self, inputs):
        delta = self.delta_decoder(inputs)
        return delta


class AutoEncoder(Model):
    def __init__(self, generator, architecture="convnn", CTF="wiener",
                 l1_lambda=0.1, multires=None, tv_lambda=0.5, mse_lambda=0.5,
                 ud_lambda=0.000001, un_lambda=0.0001, useQuaternions=False,
                 only_pos=True, only_pose=False, n_candidates=6, useHet=False, latDim=8, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.applyCTF = bool(generator.applyCTF)
        self.multires = multires
        self.useHet = useHet
        self.useQuaternions = useQuaternions
        self.common_encoder = CommonEncoder(generator.xsize, architecture=architecture)
        self.head_encoder = [HeadEncoder(generator.refinement, suffix=str(idx + 1), useQuaternions=useQuaternions) for idx in range(n_candidates)]
        if self.useHet:
            self.het_encoder = HetEncoder(generator.xsize, latDim=latDim)
            self.het_decoder = HetDecoder(generator, latDim=latDim)
        self.decoder_delta = Decoder(generator.total_voxels, CTF=CTF, only_pos=only_pos)
        self.n_candidates = n_candidates

        self.e_optimizer = None
        self.d_optimizer = None
        self.het_optimizer = None

        if multires is None:
            self.filters = None
        else:
            self.filters = create_blur_filters(multires, 10, 30)

        self.generator = generator
        self.xsize = generator.xsize
        self.l1_lambda = l1_lambda
        self.tv_lambda = tv_lambda
        self.mse_lambda = mse_lambda
        self.ud_lambda = ud_lambda
        self.un_lambda = un_lambda
        self.only_pos = only_pos
        self.only_pose = only_pose
        if only_pose:
            self.cost = correlation_coefficient_loss
        else:
            self.cost = self.generator.mse
        self.rec_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.het_rec_loss_tracker = tf.keras.metrics.Mean(name="het_rec_loss")

    @property
    def metrics(self):
        return [
            self.rec_loss_tracker,
            self.het_rec_loss_tracker,
        ]

    def prepare_batch(self, indexes):
        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(indexes)[0]

        if self.generator.refinement:
            # Precompute batch alignments
            self.generator.rot_batch = tf.gather(self.generator.angle_rot, indexes, axis=0)
            self.generator.tilt_batch = tf.gather(self.generator.angle_tilt, indexes, axis=0)
            self.generator.psi_batch = tf.gather(self.generator.angle_psi, indexes, axis=0)

            # Precompute shifts
            shifts_x = tf.gather(self.generator.shifts[0], indexes, axis=0)
            shifts_y = tf.gather(self.generator.shifts[1], indexes, axis=0)
            self.generator.shifts_batch = tf.stack([shifts_x, shifts_y], axis=1)

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.generator.defocusU, indexes, axis=0)
        defocusV_batch = tf.gather(self.generator.defocusV, indexes, axis=0)
        defocusAngle_batch = tf.gather(self.generator.defocusAngle, indexes, axis=0)
        cs_batch = tf.gather(self.generator.cs, indexes, axis=0)
        kv_batch = self.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.generator.sr, self.generator.pad_factor,
                         [self.generator.xsize, int(0.5 * self.generator.xsize + 1)],
                         batch_size_scope, self.generator.applyCTF)
        self.generator.ctf = ctf

    def compile(self, e_optimizer, d_optimizer, het_optimizer, jit_compile=False):
        super().compile(jit_compile=jit_compile)
        self.e_optimizer = e_optimizer
        self.d_optimizer = d_optimizer
        self.het_optimizer = het_optimizer

    def decode_images_with_loss(self, images, images_corrected):
        B = tf.shape(images)[0]

        # Original coordinates
        o = tf.constant(self.generator.coords, dtype=tf.float32)[None, ...]
        prev_loss_rec = 10000. * tf.ones(B)
        u_norm_loss = 0.0
        uniform_dist_loss = 0.0

        keep_r = tf.zeros((B, 3, 3), dtype=tf.float32)
        keep_shifts = tf.zeros((B, 2), dtype=tf.float32)

        encoded = self.common_encoder(images_corrected)

        # Consensus volume decoder
        if not self.only_pose:
            delta = self.decoder_delta(o)
        else:
            delta = 0.0

        # Coordinates with batch dimension
        o = self.generator.scale_factor * tf.tile(o, (B, 1, 1))

        for idr in range(self.n_candidates):
            rows, shifts = self.head_encoder[idr](encoded)

            # Compute rotation matrix
            if self.useQuaternions:
                r_no_sym = quaternion_to_rotation_matrix(rows)
            else:
                r_no_sym = gramSchmidt(rows)

            if self.generator.refinement:
                shifts = shifts + self.generator.shifts_batch

            # Symmetry loop
            loss_rec = 0.0
            for iSym in range(self.generator.noSym):
                # Prepare symmetry matrix
                R = tf.tile(self.generator.sym_matrices[iSym][None, ...], (B, 1, 1))

                # Apply symmetrix matrix
                r = tf.matmul(r_no_sym, tf.transpose(R, perm=[0, 2, 1]))

                if self.generator.refinement:
                    r_o = euler_matrix_batch(self.generator.rot_batch, self.generator.tilt_batch, self.generator.psi_batch)
                    r_o = tf.stack(r_o, axis=1)
                    r_o = tf.matmul(r_o, tf.transpose(R, perm=[0, 2, 1]))
                    r = tf.matmul(r, r_o)

                # Get rotated coords
                ro = tf.matmul(o, tf.transpose(r, perm=[0, 2, 1]))

                # Get XY coords
                ro = ro[..., :-1]

                # Apply shifts
                ro = ro - (shifts[:, None, :]) + self.generator.xmipp_origin[0]

                # Permute coords
                ro = tf.stack([ro[..., 1], ro[..., 0]], axis=-1)

                # Initialize images
                imgs = tf.zeros((B, self.generator.xsize, self.generator.xsize), dtype=tf.float32)

                # Image values
                original_values = tf.tile(self.generator.values[None, :], (B, 1))
                values = tf.cast(original_values, tf.float32) + delta

                # Backprop through coords
                bpos_round = tf.round(ro)
                bpos_flow = tf.cast(bpos_round, tf.int32)
                num = tf.reduce_sum(((bpos_round - ro) ** 2.), axis=-1)
                weight = tf.exp(-num / (2. * 1. ** 2.))
                values = values * weight

                # Scatter images
                fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
                imgs = tf.map_fn(fn, [imgs, bpos_flow, values], fn_output_signature=tf.float32)

                # Reshape images
                imgs = tf.reshape(imgs, [-1, self.xsize, self.xsize, 1])

                # Gaussian filtering
                imgs = tfa.image.gaussian_filter2d(imgs, 3, 1)

                # CTF corruption
                if self.applyCTF:
                    imgs = self.generator.ctfFilterImage(imgs)

                # Image loss
                loss_rec += self.cost(images, imgs)

                if self.multires is not None:
                    filt_images = apply_blur_filters_to_batch(images_corrected, self.filters)
                    filt_decoded = apply_blur_filters_to_batch(imgs, self.filters)
                    for idx in range(self.multires):
                        loss_rec += 0.001 * self.cost(filt_images[..., idx], filt_decoded[..., idx])

            loss_rec /= self.generator.noSym

            # Preparing indexing for "winner's takes it all"
            mask = tf.less_equal(loss_rec, prev_loss_rec)  # Shape: (B,)
            mask_shape_r = tf.concat([tf.shape(mask), tf.ones(2, dtype=tf.int32)], axis=0)
            mask_shape_shifts = tf.concat([tf.shape(mask), tf.ones(1, dtype=tf.int32)], axis=0)
            mask_r = tf.reshape(mask, mask_shape_r)
            mask_shifts = tf.reshape(mask, mask_shape_shifts)

            # Minimum indexing
            keep_r = tf.where(mask_r, r_no_sym, keep_r)
            keep_shifts = tf.where(mask_shifts, shifts, keep_shifts)

            loss_rec = tf.reduce_min(tf.stack([loss_rec, prev_loss_rec], axis=-1), axis=-1)

            # Unit norm constrain
            if self.useQuaternions:
                x = tf.abs(tf.reduce_sum(tf.square(rows), axis=-1) - 1.0)
            else:
                n1 = tf.abs(tf.reduce_sum(tf.square(rows[..., :3]), axis=-1) - 1.0)
                n2 = tf.abs(tf.reduce_sum(tf.square(rows[..., 3:]), axis=-1) - 1.0)
                x = (0.5 * (tf.reduce_mean(n1) + tf.reduce_mean(n2)))
            u_norm_loss += x

            # Uniform distribution loss
            z_vec = tf.tile(tf.constant([0, 0, 1], tf.float32)[None, None, :], (B, 1, 1))
            r_z_vec = tf.squeeze(tf.matmul(z_vec, tf.transpose(r_no_sym, perm=[0, 2, 1])))
            x = uniform_distribution_loss(r_z_vec)
            uniform_dist_loss += x

            # Keep best loss
            prev_loss_rec = loss_rec

        u_norm_loss = u_norm_loss / self.n_candidates
        uniform_dist_loss = uniform_dist_loss / self.n_candidates

        # L1 penalization delta_het
        values = delta + self.generator.values[None, :]
        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(values), axis=1))
        l1_loss = self.l1_lambda * l1_loss / self.generator.total_voxels
        l1_dist_loss = l1_distance_norm(values, tf.constant(self.generator.coords, tf.float32))
        l1_loss += self.l1_lambda * l1_dist_loss

        # Total variation and MSE losses
        tv_loss, d_mse_loss = densitySmoothnessVolume(self.generator.xsize,
                                                      self.generator.indices, values)
        tv_loss *= self.tv_lambda
        d_mse_loss *= self.mse_lambda

        # Negative loss
        if not self.only_pos:
            mask = tf.less(values, 0.0)
            delta_neg = tf.boolean_mask(values, mask)
            delta_neg = tf.reduce_mean(tf.abs(tf.cast(delta_neg, tf.float32)), keepdims=True)
            neg_loss = tf.cast(self.only_pos, tf.float32) * self.l1_lambda * delta_neg

        else:
            neg_loss = 0.0

        # Reconstruction loss
        loss_rec += (l1_loss + self.ud_lambda * uniform_dist_loss + self.un_lambda * u_norm_loss
                     + tv_loss + d_mse_loss + 0.1 * neg_loss)

        return loss_rec, keep_r, keep_shifts, delta

    def decode_images_het_with_loss(self, images, images_corrected, r_no_sym, shifts, delta):
        B = tf.shape(images)[0]

        # Original coordinates
        o = tf.constant(self.generator.coords, dtype=tf.float32)[None, ...]

        # Heterogeneous volume decoder
        het = self.het_encoder(images_corrected)
        delta_het = self.het_decoder(het)

        # Coordinates with batch dimension
        o = self.generator.scale_factor * tf.tile(o, (B, 1, 1))

        if self.generator.refinement:
            shifts = shifts + self.generator.shifts_batch

        if self.generator.refinement:
            r_o = euler_matrix_batch(self.generator.rot_batch, self.generator.tilt_batch, self.generator.psi_batch)
            r_o = tf.stack(r_o, axis=1)
            r_no_sym = tf.matmul(r_no_sym, r_o)

        # if useHet:
        #     if self.useQuaternions:
        #         r_het = quaternion_to_rotation_matrix(rows_het)
        #     else:
        #         r_het = gramSchmidt(rows_het)
        #     # r_het = tf.matmul(r_het, tf.transpose(R, perm=[0, 2, 1]))
        #     r = tf.matmul(r_het, r)

        # Get rotated coords
        ro = tf.matmul(o, tf.transpose(r_no_sym, perm=[0, 2, 1]))

        # Get XY coords
        ro = ro[..., :-1]

        # Apply shifts
        ro = ro - (shifts[:, None, :]) + self.generator.xmipp_origin[0]

        # Permute coords
        ro = tf.stack([ro[..., 1], ro[..., 0]], axis=-1)

        # Initialize images
        imgs_with_het = tf.zeros((B, self.generator.xsize, self.generator.xsize), dtype=tf.float32)

        # Image values
        original_values = tf.tile(self.generator.values[None, :], (B, 1))
        values_with_het = tf.cast(original_values, tf.float32) + delta_het

        # Backprop through coords
        bpos_round = tf.round(ro)
        bpos_flow = tf.cast(bpos_round, tf.int32)
        num = tf.reduce_sum(((bpos_round - ro) ** 2.), axis=-1)
        weight = tf.exp(-num / (2. * 1. ** 2.))
        values_with_het = values_with_het * weight

        # Scatter images
        fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
        imgs_with_het = tf.map_fn(fn, [imgs_with_het, bpos_flow, values_with_het], fn_output_signature=tf.float32)

        # Reshape images
        imgs_with_het = tf.reshape(imgs_with_het, [-1, self.xsize, self.xsize, 1])

        # Gaussian filtering
        imgs_with_het = tfa.image.gaussian_filter2d(imgs_with_het, 3, 1)

        # CTF corruption
        if self.applyCTF:
            imgs_with_het = self.generator.ctfFilterImage(imgs_with_het)

        loss_rec_with_het = self.cost(images, imgs_with_het)

        if self.multires is not None:
            filt_images = apply_blur_filters_to_batch(images_corrected, self.filters)
            filt_decoded = apply_blur_filters_to_batch(imgs_with_het, self.filters)
            for idx in range(self.multires):
                loss_rec_with_het += 0.001 * self.cost(filt_images[..., idx], filt_decoded[..., idx])

        # L1 penalization delta_het
        values_het = delta + self.generator.values[None, :] + delta_het
        l1_loss_het = tf.reduce_mean(tf.reduce_sum(tf.abs(values_het), axis=1))
        l1_loss_het = self.l1_lambda * l1_loss_het / self.generator.total_voxels
        l1_dist_loss = l1_distance_norm(values_het, tf.constant(self.generator.coords, tf.float32))
        l1_loss_het += self.l1_lambda * l1_dist_loss

        # Total variation and MSE losses
        tv_loss_het, d_mse_loss_het = densitySmoothnessVolume(self.generator.xsize,
                                                              self.generator.indices, values_het)
        tv_loss_het *= self.tv_lambda
        d_mse_loss_het *= self.mse_lambda

        # Negative loss
        mask = tf.less(values_with_het, 0.0)
        delta_neg_het = tf.boolean_mask(values_with_het, mask)
        delta_neg_het = tf.reduce_mean(tf.abs(tf.cast(delta_neg_het, tf.float32)), keepdims=True)
        neg_loss_het = tf.cast(self.only_pos, tf.float32) * self.l1_lambda * delta_neg_het

        # Reconstruction loss
        loss_rec_with_het += tv_loss_het + d_mse_loss_het + l1_loss_het + 0.1 * neg_loss_het

        return loss_rec_with_het

    def train_step(self, data):
        images = data[0]

        # Prepare batch
        self.prepare_batch(data[1])

        # Wiener filter
        if self.applyCTF:
            images_corrected = self.generator.wiener2DFilter(images)
        else:
            images_corrected = images

        # Encoder tape
        with tf.GradientTape() as tape_e:
            loss_rec_e, r_no_sym, shifts, delta = self.decode_images_with_loss(images, images_corrected)

        # Get weights encoder + pose + shifts + het
        encoder_weights = self.common_encoder.trainable_weights
        for model in self.head_encoder:
            encoder_weights += model.trainable_weights

        # Gradients
        if self.only_pose:
            grads_e = tape_e.gradient(loss_rec_e, encoder_weights)
        else:
            grads_e, grads_d = tape_e.gradient(loss_rec_e, [encoder_weights, self.decoder_delta.trainable_weights])

        # Apply encoder gradients
        self.e_optimizer[0].apply_gradients(zip(grads_e, encoder_weights))
        # tf.cond(tf.less_equal(self.epoch_id, 25), lambda: self.apply_opt_ab_initio(grads_e, encoder_weights),
        #         lambda: self.apply_opt_refinement(grads_e, encoder_weights))

        # Apply decoder gradients
        if not self.only_pose:
            self.d_optimizer.apply_gradients(zip(grads_d, self.decoder_delta.trainable_weights))

        if self.useHet:
            with tf.GradientTape() as tape_het:
                loss_rec_with_het = self.decode_images_het_with_loss(images, images_corrected, tf.stop_gradient(r_no_sym),  tf.stop_gradient(shifts),  tf.stop_gradient(delta))

        if self.useHet:
            grads_het_e, grads_het_d = tape_het.gradient(loss_rec_with_het, [self.het_encoder.trainable_weights,
                                                                             self.het_decoder.trainable_weights])

        # Apply het encoder gradients
        if self.useHet:
            self.het_optimizer[0].apply_gradients(zip(grads_het_e, self.het_encoder.trainable_weights))
            self.het_optimizer[1].apply_gradients(zip(grads_het_d, self.het_decoder.trainable_weights))

        self.rec_loss_tracker.update_state(loss_rec_e)
        return_dict = {"rec_loss": self.rec_loss_tracker.result(),}
        if self.useHet:
            self.het_rec_loss_tracker.update_state(loss_rec_with_het)
            return_dict["het_rec_loss"] = self.het_rec_loss_tracker.result()
        return return_dict

    def apply_opt_ab_initio(self, grads, weights):
        self.e_optimizer[0].apply_gradients(zip(grads, weights))

    def apply_opt_refinement(self, grads, weights):
        self.e_optimizer[1].apply_gradients(zip(grads, weights))

    def test_step(self, data):
        images = data[0]

        # Prepare batch
        self.prepare_batch(data[1])

        # Wiener filter
        if self.applyCTF:
            images_corrected = self.generator.wiener2DFilter(images)
        else:
            images_corrected = images

        # Encoder
        loss_rec_e, r_no_sym, shifts, delta = self.decode_images_with_loss(images, images_corrected)

        if self.useHet:
            loss_rec_with_het = self.decode_images_het_with_loss(images, images_corrected,
                                                                     tf.stop_gradient(r_no_sym),
                                                                     tf.stop_gradient(shifts), tf.stop_gradient(delta))

        self.rec_loss_tracker.update_state(loss_rec_e)
        return_dict = {"rec_loss": self.rec_loss_tracker.result(), }
        if self.useHet:
            self.het_rec_loss_tracker.update_state(loss_rec_with_het)
            return_dict["het_rec_loss"] = self.het_rec_loss_tracker.result()
        return return_dict

    def eval_volume(self, filter=True, het=None):
        coords = tf.constant(self.generator.coords, dtype=tf.float32)[None, ...]
        o = self.generator.indices[None, ...]

        # Delta volume
        if not self.only_pose and het is None:
            delta = self.decoder_delta(coords)
        else:
            delta = 0.0

        # Heterogeneous delta volume
        if self.useHet and het is not None:
            delta_het = self.het_decoder(het)
            num_vols = het.shape[0]
            o = tf.tile(o, (num_vols, 1, 1))
        else:
            delta_het = 0.0
            num_vols = 1

        # Decode map
        values = self.generator.values[None, ...] + delta + delta_het

        # Create volume grid
        volumes = tf.zeros((num_vols, self.generator.xsize, self.generator.xsize, self.generator.xsize),
                           dtype=tf.float32)

        # Scatter in volumes
        fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
        volumes = tf.map_fn(fn, [volumes, o, values], fn_output_signature=tf.float32).numpy()

        # Filter volumes
        if filter:
            for idx in range(num_vols):
                volumes[idx] = gaussian_filter(volumes[idx], sigma=1)

        return volumes

    def predict_step(self, data):
        self.generator.indexes = data[1]
        self.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        if self.generator.refinement:
            # Precompute batch alignments
            rot_batch = tf.gather(self.generator.angle_rot, data[1], axis=0)
            tilt_batch = tf.gather(self.generator.angle_tilt, data[1], axis=0)
            psi_batch = tf.gather(self.generator.angle_psi, data[1], axis=0)

            # Precompute shifts
            shifts_x = tf.gather(self.generator.shifts[0], data[1], axis=0)
            shifts_y = tf.gather(self.generator.shifts[1], data[1], axis=0)
            shifts_batch = tf.stack([shifts_x, shifts_y], axis=1)

        # Precompute batch CTFs
        defocusU_batch = tf.gather(self.generator.defocusU, data[1], axis=0)
        defocusV_batch = tf.gather(self.generator.defocusV, data[1], axis=0)
        defocusAngle_batch = tf.gather(self.generator.defocusAngle, data[1], axis=0)
        cs_batch = tf.gather(self.generator.cs, data[1], axis=0)
        kv_batch = self.generator.kv
        ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                         self.generator.sr, self.generator.pad_factor,
                         [self.generator.xsize, int(0.5 * self.generator.xsize + 1)],
                         batch_size_scope, self.generator.applyCTF)
        self.generator.ctf = ctf

        # Wiener filter
        if self.applyCTF:
            images_corrected = self.generator.wiener2DFilter(images)
        else:
            images_corrected = images

        # Original coordinates
        o = tf.constant(self.generator.coords, dtype=tf.float32)[None, ...]

        # Common encoder and volume decoder
        encoded = self.common_encoder(images_corrected)

        # Consensus volume decoder
        if not self.only_pose:
            delta = self.decoder_delta(o)
        else:
            delta = 0.0

        # Heterogeneous volume decoder
        if self.useHet:
            # het, rows_het, shifts_het = self.het_encoder(images_corrected)
            het = self.het_encoder(images_corrected)
            delta_het = self.het_decoder(het)
        else:
            het = 0.0
            delta_het = 0.0

        # Coordinates with batch dimension
        o = self.generator.scale_factor * tf.tile(o, (batch_size_scope, 1, 1))

        # Prepare outputs
        prev_loss_rec = 10000. * tf.ones(batch_size_scope, dtype=tf.float32)
        prev_loss_rec_cons = 10000. * tf.ones(batch_size_scope, dtype=tf.float32)
        keep_r = tf.zeros((batch_size_scope, 3, 3), dtype=tf.float32)
        keep_shifts = tf.zeros((batch_size_scope, 2), dtype=tf.float32)
        keep_imgs = tf.zeros((batch_size_scope, self.generator.xsize, self.generator.xsize), dtype=tf.float32)
        keep_imgs_cons = tf.zeros((batch_size_scope, self.generator.xsize, self.generator.xsize), dtype=tf.float32)

        # Multi-head encoders
        for idr in range(self.n_candidates):
            rows, shifts = self.head_encoder[idr](encoded)

            if self.generator.refinement:
                shifts = shifts + shifts_batch

            # if self.useHet:
            #     shifts = shifts + shifts_het

            # Get rotation matrices
            if self.useQuaternions:
                r = quaternion_to_rotation_matrix(rows)
            else:
                r = gramSchmidt(rows)

            if self.generator.refinement:
                r_o = euler_matrix_batch(rot_batch, tilt_batch, psi_batch)
                r_o = tf.stack(r_o, axis=1)
                r = tf.matmul(r, r_o)

            # if self.useHet:
            #     if self.useQuaternions:
            #         r_het = quaternion_to_rotation_matrix(rows_het)
            #     else:
            #         r_het = gramSchmidt(rows_het)
            #     r = tf.matmul(r_het, r)

            # Get rotated coords
            ro = tf.matmul(o, tf.transpose(r, perm=[0, 2, 1]))

            # Get XY coords
            ro = ro[..., :-1]

            # Apply shifts
            ro = ro - (shifts[:, None, :]) + self.generator.xmipp_origin[0]

            # Permute coords
            ro = tf.stack([ro[..., 1], ro[..., 0]], axis=-1)

            # Initialize images
            imgs_cons = tf.zeros((batch_size_scope, self.generator.xsize, self.generator.xsize), dtype=tf.float32)
            imgs = tf.zeros((batch_size_scope, self.generator.xsize, self.generator.xsize), dtype=tf.float32)

            # Image values
            original_values = tf.tile(self.generator.values[None, :], (batch_size_scope, 1))
            values_cons = original_values + delta
            values = original_values + delta_het

            # Backprop through coords
            bpos_round = tf.round(ro)
            bpos_flow = tf.cast(bpos_round, tf.int32)
            num = tf.reduce_sum(((bpos_round - ro) ** 2.), axis=-1)
            weight = tf.exp(-num / (2. * 1. ** 2.))
            values_cons = values_cons * weight
            values = values * weight

            # Scatter images
            fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
            imgs_cons = tf.map_fn(fn, [imgs_cons, bpos_flow, values_cons], fn_output_signature=tf.float32)
            imgs = tf.map_fn(fn, [imgs, bpos_flow, values], fn_output_signature=tf.float32)

            # Reshape images
            imgs_cons = tf.reshape(imgs_cons, [-1, self.xsize, self.xsize, 1])
            imgs = tf.reshape(imgs, [-1, self.xsize, self.xsize, 1])

            # Gaussian filtering
            imgs_cons = tfa.image.gaussian_filter2d(imgs_cons, 3, 1)
            imgs = tfa.image.gaussian_filter2d(imgs, 3, 1)

            # CTF corruption
            if self.applyCTF:
                imgs_cons = self.generator.ctfFilterImage(imgs_cons)
                imgs = self.generator.ctfFilterImage(imgs)

            # Image loss
            loss_rec_cons = self.cost(images, imgs_cons)
            loss_rec = self.cost(images, imgs)

            # Preparing indexing for "winner's takes it all"
            # mask = tf.less_equal(loss_rec, prev_loss_rec)  # Shape: (B,)
            mask = tf.less_equal(loss_rec_cons, prev_loss_rec_cons)  # Shape: (B,)
            mask_shape_r = tf.concat([tf.shape(mask), tf.ones(2, dtype=tf.int32)], axis=0)
            mask_shape_shifts = tf.concat([tf.shape(mask), tf.ones(1, dtype=tf.int32)], axis=0)
            mask_shape_imgs_cons = tf.concat([tf.shape(mask), tf.ones(2, dtype=tf.int32)], axis=0)
            mask_shape_imgs = tf.concat([tf.shape(mask), tf.ones(2, dtype=tf.int32)], axis=0)
            mask_r = tf.reshape(mask, mask_shape_r)
            mask_shifts = tf.reshape(mask, mask_shape_shifts)
            mask_imgs_cons = tf.reshape(mask, mask_shape_imgs_cons)
            mask_imgs = tf.reshape(mask, mask_shape_imgs)

            # Minimum indexing
            prev_loss_rec = tf.where(mask, loss_rec, prev_loss_rec)
            prev_loss_rec_cons = tf.where(mask, loss_rec_cons, prev_loss_rec_cons)
            keep_r = tf.where(mask_r, r, keep_r)
            keep_shifts = tf.where(mask_shifts, shifts, keep_shifts)
            keep_imgs_cons = tf.where(mask_imgs_cons, imgs_cons[..., 0], keep_imgs_cons)
            keep_imgs = tf.where(mask_imgs, imgs[..., 0], keep_imgs)

        return keep_r, keep_shifts, keep_imgs, het, prev_loss_rec, prev_loss_rec_cons

    def call(self, input_features):
        # Original coordinates
        o = tf.constant(self.generator.coords, dtype=tf.float32)[None, ...]
        delta = self.decoder_delta(o)
        return delta

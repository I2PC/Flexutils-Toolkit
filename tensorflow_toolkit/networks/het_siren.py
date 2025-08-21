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

from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import signal
from xmipp_metadata.image_handler import ImageHandler
from xmipp_metadata.metadata import XmippMetaData

from tensorflow_toolkit.utils import computeCTF, full_fft_pad, full_ifft_pad, create_blur_filters, \
    apply_blur_filters_to_batch
from tensorflow_toolkit.layers.siren import SIRENFirstLayerInitializer, SIRENInitializer, MetaDenseWrapper, Sine


##### Extra functions for HetSIREN network #####
def richardsonLucyDeconvolver(volume, iter=5):
    original_volume = volume.copy()
    volume = tf.constant(volume, dtype=tf.float32)
    original_volume = tf.constant(original_volume, dtype=tf.float32)

    std = np.pi * np.sqrt(volume.shape[1])
    gauss_1d = signal.windows.gaussian(volume.shape[1], std)
    kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
    kernel = tf.constant(kernel, dtype=tf.float32)

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


def richardsonLucyBlindDeconvolver(volume, global_iter=5, iter=20):
    original_volume = volume.copy()
    volume = tf.constant(volume, dtype=tf.float32)
    original_volume = tf.constant(original_volume, dtype=tf.float32)

    # Create a gaussian kernel that will be used to blur the original acquisition
    std = 1.0
    gauss_1d = signal.windows.gaussian(volume.shape[1], std)
    kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
    kernel = tf.constant(kernel, dtype=tf.float32)

    def applyKernelFourier(x, y):
        x = tf.cast(x, dtype=tf.complex64)
        y = tf.cast(y, dtype=tf.complex64)
        ft_x = tf.signal.fftshift(tf.signal.fft3d(x))
        ft_y = tf.abs(tf.signal.fftshift(tf.signal.fft3d(y)))
        ft_x_real = tf.math.real(ft_x) * ft_y
        ft_x_imag = tf.math.imag(ft_x) * ft_y
        ft_x = tf.complex(ft_x_real, ft_x_imag)
        return tf.math.real(tf.signal.ifft3d(tf.signal.fftshift(ft_x)))

    for _ in range(global_iter):
        for _ in range(iter):
            # Deconvolve image (update)
            conv_1 = applyKernelFourier(volume, kernel)
            conv_1_2 = conv_1 * conv_1
            epsilon = 1e-6 * tf.reduce_mean(conv_1_2)
            update = original_volume * conv_1 / (conv_1_2 + epsilon)
            update = applyKernelFourier(update, tf.reverse(kernel, axis=[0, 1, 2]))
            volume = volume * update

            # volume = volume.numpy()
            # thr = 1e-6
            # volume = volume - (volume > thr) * thr + (volume < -thr) * thr - (volume == thr) * volume
            # volume = tf.constant(volume, dtype=tf.float32)

        for _ in range(iter):
            # Update kernel
            conv_1 = applyKernelFourier(kernel, volume)
            conv_1_2 = conv_1 * conv_1
            epsilon = 1e-6 * tf.reduce_mean(conv_1_2)
            update = original_volume * conv_1 / (conv_1_2 + epsilon)
            update = applyKernelFourier(update, tf.reverse(volume, axis=[0, 1, 2]))
            kernel = kernel * update

            # kernel = kernel.numpy()
            # thr = 1e-6
            # kernel = kernel - (kernel > thr) * thr + (kernel < -thr) * thr - (kernel == thr) * kernel
            # kernel = tf.constant(kernel, dtype=tf.float32)

    return volume


def deconvolveTV(volume, iterations, regularization_weight, lr=0.01):
    original = tf.Variable(volume, dtype=tf.float32)

    # Create a gaussian kernel that will be used to blur the original acquisition
    std = 1.0
    gauss_1d = signal.windows.gaussian(volume.shape[1], std)
    psf = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
    psf = tf.constant(psf, dtype=tf.float32)

    def applyKernelFourier(x, y):
        x = tf.cast(x, dtype=tf.complex64)
        y = tf.cast(y, dtype=tf.complex64)
        ft_x = tf.signal.fftshift(tf.signal.fft3d(x))
        ft_y = tf.abs(tf.signal.fftshift(tf.signal.fft3d(y)))
        ft_x_real = tf.math.real(ft_x) * ft_y
        ft_x_imag = tf.math.imag(ft_x) * ft_y
        ft_x = tf.complex(ft_x_real, ft_x_imag)
        return tf.math.real(tf.signal.ifft3d(tf.signal.fftshift(ft_x)))

    for i in range(iterations):
        with tf.GradientTape() as tape:
            # Convolve with PSF
            # convolved = tf.nn.conv2d(tf.expand_dims(original, axis=0), tf.expand_dims(psf, axis=0), strides=[1, 1, 1, 1], padding='SAME')
            # convolved = tf.squeeze(convolved)
            convolved = applyKernelFourier(volume, psf)

            # Calculate the loss (data fidelity term + TV regularization)
            loss = tf.reduce_mean(tf.square(convolved - volume)) + regularization_weight * tf.reduce_sum(
                tf.image.total_variation(original))

        # Perform a gradient descent step
        grads = tape.gradient(loss, [original])
        # grads = tf.gradients(loss, [original])
        original.assign_sub(lr * grads[0])

    return original.numpy()


def tv_deconvolution_bregman(volume, iterations, regularization_weight, lr=0.01):
    deconvolved = tf.Variable(volume, dtype=tf.float32)
    bregman = tf.Variable(tf.zeros_like(volume), dtype=tf.float32)

    # Create a gaussian kernel that will be used to blur the original acquisition
    std = 1.0
    gauss_1d = signal.windows.gaussian(volume.shape[1], std)
    psf = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
    psf_tf = tf.constant(psf, dtype=tf.float32)

    # psf_mirror = tf.reverse(tf.reverse(psf_tf, axis=[0]), axis=[1])

    def applyKernelFourier(x, y):
        x = tf.cast(x, dtype=tf.complex64)
        y = tf.cast(y, dtype=tf.complex64)
        ft_x = tf.signal.fftshift(tf.signal.fft3d(x))
        ft_y = tf.abs(tf.signal.fftshift(tf.signal.fft3d(y)))
        ft_x_real = tf.math.real(ft_x) * ft_y
        ft_x_imag = tf.math.imag(ft_x) * ft_y
        ft_x = tf.complex(ft_x_real, ft_x_imag)
        return tf.math.real(tf.signal.ifft3d(tf.signal.fftshift(ft_x)))

    for i in range(iterations):
        with tf.GradientTape() as tape:
            # Convolve with PSF
            # convolved = tf.nn.conv2d(tf.expand_dims(original, axis=0), tf.expand_dims(psf, axis=0), strides=[1, 1, 1, 1], padding='SAME')
            # convolved = tf.squeeze(convolved)
            convolved = applyKernelFourier(volume, psf)

            # Calculate the loss (data fidelity term + TV regularization)
            loss = tf.reduce_mean(tf.square(convolved - volume)) + regularization_weight * tf.reduce_sum(
                tf.image.total_variation(deconvolved - bregman))

        # Perform a gradient descent step
        grads = tape.gradient(loss, [deconvolved])
        # grads = tf.gradients(loss, [deconvolved])
        deconvolved.assign_sub(lr * grads[0])

        # Bregman Update
        bregman.assign(bregman + deconvolved - tv_minimization_step(deconvolved, lr))

    return deconvolved.numpy()


def tv_minimization_step(image, lr):
    # Implement the TV minimization step
    # This is a placeholder function, in practice, you'll need a proper implementation
    return image - lr * tf.image.total_variation(image)


### Image smoothness with TV ###
def total_variation_loss(volume, diff1, diff2, diff3, precision, precision_scaled=tf.float32):
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
    # sum_axis = 1
    loss = tf.reduce_sum(tf.abs(tf.cast(diff1, precision_scaled)), axis=sum_axis) + \
           tf.reduce_sum(tf.abs(tf.cast(diff2, precision_scaled)), axis=sum_axis) + \
           tf.reduce_sum(tf.abs(tf.cast(diff3, precision_scaled)), axis=sum_axis)

    # Normalize by the volume size
    num_pixels = tf.cast(tf.reduce_prod(volume.shape[1:]), precision_scaled)
    loss = tf.cast(loss / num_pixels, precision)

    return loss


def mse_smoothness_loss(volume, diff1, diff2, diff3, precision, precision_scaled=tf.float32):
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
    diff1 = tf.square(tf.cast(diff1, precision_scaled))
    diff2 = tf.square(tf.cast(diff2, precision_scaled))
    diff3 = tf.square(tf.cast(diff3, precision_scaled))

    # Sum the squared differences
    sum_axis = [1, 2, 3]
    # sum_axis = 1
    loss = tf.reduce_sum(diff1, axis=sum_axis) + tf.reduce_sum(diff2, axis=sum_axis) + tf.reduce_sum(diff3,
                                                                                                     axis=sum_axis)

    # Normalize by the number of pixel pairs
    num_pixel_pairs = tf.cast(2 * tf.reduce_prod(volume.shape[1:3]) - volume.shape[1] - volume.shape[2], precision_scaled)
    loss = tf.cast(tf.cast(loss, precision_scaled) / num_pixel_pairs, precision)

    return loss


def densitySmoothnessVolume(xsize, indices, values, precision, precision_scaled=tf.float32):
    B = tf.shape(values)[0]

    grid = tf.zeros((B, xsize, xsize, xsize), dtype=precision)
    indices = tf.tile(tf.cast(indices[None, ...], dtype=tf.int32), (B, 1, 1))

    # Scatter in volumes
    fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
    grid = tf.map_fn(fn, [grid, indices, values], fn_output_signature=precision)

    # Calculate the differences of neighboring pixel-values.
    # The total variation loss is the sum of absolute differences of neighboring pixels
    # in both dimensions.
    pixel_diff1 = grid[:, 1:, :, :] - grid[:, :-1, :, :]
    pixel_diff2 = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    pixel_diff3 = grid[:, :, :, 1:] - grid[:, :, :, :-1]

    # Compute total variation and density MSE losses
    return (total_variation_loss(grid, pixel_diff1, pixel_diff2, pixel_diff3, precision, precision_scaled),
            mse_smoothness_loss(grid, pixel_diff1, pixel_diff2, pixel_diff3, precision, precision_scaled))

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

    # Create a gaussian kernel that will be used to blur the original acquisition
    # std = 2.0
    # gauss_1d = signal.windows.gaussian(volume.shape[1], std)
    # kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
    # kernel = tf.constant(kernel, dtype=tf.complex64)
    # ft_kernel = tf.abs(tf.signal.fftshift(tf.signal.fft3d(kernel)))

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


def normalize_to_other_volumes(batch1, batch2):
    """
    Normalize volumes in batch2 to have the same mean and std as the corresponding volumes in batch1.

    Parameters:
    batch1, batch2: numpy arrays of shape (B, W, H, D) representing batches of volumes.

    Returns:
    normalized_batch2: numpy array of normalized images.
    """
    # Calculate mean and std for each image in batch1
    means1 = batch1.mean(axis=(1, 2, 3), keepdims=True)
    stds1 = batch1.std(axis=(1, 2, 3), keepdims=True)

    # Calculate mean and std for each image in batch2
    means2 = batch2.mean(axis=(1, 2, 3), keepdims=True)
    stds2 = batch2.std(axis=(1, 2, 3), keepdims=True)

    # Normalize batch2 to have the same mean and std as batch1
    normalized_batch2 = ((batch2 - means2) / stds2) * stds1 + means1

    return normalized_batch2


def match_histograms(source, reference):
    """
    Adjust the pixel values of a N-D source volume to match the histogram of a reference volume.

    Parameters:
    - source: ndarray
      Input volume. Can be of shape (B, W, H, D).
    - reference: ndarray
      Reference volume. Must have the same shape as the source.

    Returns:
    - matched: ndarray
      The source volume after histogram matching.
    """
    matched = np.zeros_like(source)

    for b in range(source.shape[0]):
        # Flatten the volumes
        s_values = source[b].ravel()
        r_values = reference[b].ravel()

        # Get unique values and their corresponding indices for both source and reference
        s_values_unique, s_inverse = np.unique(s_values, return_inverse=True)
        r_values_unique, r_counts = np.unique(r_values, return_counts=True)

        # Calculate the CDF for the source and reference
        s_quantiles = np.cumsum(np.bincount(s_inverse, minlength=s_values_unique.size))
        s_quantiles = s_quantiles / s_quantiles[-1]
        r_quantiles = np.cumsum(r_counts)
        r_quantiles = r_quantiles / r_quantiles[-1]

        # Interpolate
        interp_r_values = np.interp(s_quantiles, r_quantiles, r_values_unique)

        # Map the source pixels to the reference pixels
        matched[b] = interp_r_values[s_inverse].reshape(source[b].shape)

    return matched

def compute_histogram(tensor, bins=10, minval=None, maxval=None):
    """Computes histograms for each row in a batched tensor.

    Args:
        tensor: A Tensor of shape (B, M) representing the data.
        bins: The number of histogram bins to use.
        minval: Optional minimum value for histogram range.
        maxval: Optional maximum value for histogram range.

    Returns:
        A Tensor of shape (B, bins) containing the histograms for each row.
    """
    B = tf.shape(tensor)[0]
    M = tf.shape(tensor)[1]  # Get the number of elements per row

    if minval is None:
        minval = tf.reduce_min(tensor)
    if maxval is None:
        maxval = tf.reduce_max(tensor)

    bin_width = (maxval - minval) / bins
    bin_indices = tf.cast(
        tf.math.floor((tensor - minval) / bin_width), dtype=tf.int32
    )
    bin_indices = tf.clip_by_value(bin_indices, 0, bins - 1)

    # Pre-allocate a tensor for bin counts
    bin_counts = tf.zeros((B, bins), dtype=tf.int32)

    # Create row indices for scatter update
    row_indices = tf.repeat(tf.range(B), M)

    # Reshape bin_indices for scatter update
    scatter_indices = tf.stack([row_indices, tf.reshape(bin_indices, [-1])], axis=1)

    # Update bin counts using scatter_add
    bin_counts = tf.tensor_scatter_nd_add(
        bin_counts, scatter_indices, tf.ones(tf.size(bin_indices), dtype=tf.int32)
    )
    bin_counts = tf.cast(bin_counts, tf.float32)

    return bin_counts / tf.reduce_sum(bin_counts, axis=1, keepdims=True)


class Encoder(Model):
    def __init__(self, latent_dim, input_dim, architecture="convnn", refPose=True,
                 mode="spa", downsample=False):
        super(Encoder, self).__init__()
        self.mode = mode
        filters = create_blur_filters(5, 5, 15)

        images = Input(shape=(input_dim, input_dim, 1))
        subtomo_pe = Input(shape=(100,))

        if architecture == "convnn":
            if downsample:
                x = resizeImageFourier(images, 64)
                x = tf.keras.layers.Reshape((64, 64, 1))(x)
                x = layers.ReLU()(x)
                # x = apply_blur_filters_to_batch(x, filters)
            else:
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

        elif architecture == "deepconv":
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

        elif architecture == "mlpnn":
            x = layers.Flatten()(images)
            x = layers.Dense(1024, activation='relu')(x)
            for _ in range(2):
                x = layers.Dense(1024, activation='relu')(x)

        latent = layers.Dense(256, activation="relu")(x)
        for _ in range(2):
            latent = layers.Dense(256, activation="relu")(latent)
        if mode == "tomo":
            latent_label = layers.Dense(1024, activation="relu")(subtomo_pe)
            for _ in range(2):  # TODO: Is it better to use 12 hidden layers as in Zernike3Deep?
                latent_label = layers.Dense(1024, activation="relu")(latent_label)
            for _ in range(2):
                latent_label = layers.Dense(256, activation="relu")(latent_label)
        # latent = layers.Dense(latent_dim, activation="linear")(latent)  # Tanh [-1,1] as needed by SIREN?

        rows = layers.Dense(256, activation="relu", trainable=refPose)(x)
        for _ in range(2):
            rows = layers.Dense(256, activation="relu", trainable=refPose)(rows)
        # rows = layers.Dense(3, activation="linear", trainable=refPose)(rows)

        shifts = layers.Dense(256, activation="relu", trainable=refPose)(x)
        for _ in range(2):
            shifts = layers.Dense(256, activation="relu", trainable=refPose)(shifts)
        # shifts = layers.Dense(2, activation="linear", trainable=refPose)(shifts)

        if mode == "spa":
            self.encoder = Model(images, [rows, shifts, latent], name="encoder")
        elif mode == "tomo":
            self.encoder = Model([images, subtomo_pe], [rows, shifts, latent, latent_label], name="encoder")
            # self.encoder_latent = Model(subtomo_pe, latent_label, name="encode_latent")

    def call(self, x):
        encoded = self.encoder(x)
        if self.mode == "spa":
            encoded.append(None)
        return encoded


class Decoder(Model):
    def __init__(self, latent_dim, generator, CTF="apply", use_hyper_network=True):
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
        if use_hyper_network:
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
                                     name=f"het_{count}", kernel_initializer=self.generator.weight_initializer)(delta_het)
        else:
            delta_het = layers.Dense(latent_dim, activation=Sine(w0_first),
                                     kernel_initializer=SIRENFirstLayerInitializer(scale=1.0),
                                     name=f"het_{count}")(latent)  # activation=Sine(w0=1.0)
            for _ in range(3):
                count += 1
                aux = layers.Dense(latent_dim, activation=Sine(1.0),
                                   kernel_initializer=SIRENInitializer(),
                                   name=f"het_{count}")(latent)  # activation=Sine(w0=1.0)
                delta_het = layers.Add()([delta_het, aux])
            count += 1
            delta_het = layers.Dense(self.generator.total_voxels, activation='linear',
                                     name=f"het_{count}", kernel_initializer=self.generator.weight_initializer)(delta_het)

        # Scatter image and bypass gradient
        decoded_het = layers.Lambda(self.generator.scatterImgByPass)([coords, shifts, delta_het])

        # Gaussian filter image
        decoded_het = layers.Lambda(self.generator.gaussianFilterImage)(decoded_het)

        # Soft threshold image
        decoded_het = layers.Lambda(self.generator.softThresholdImage)(decoded_het)

        if self.CTF == "apply":
            # CTF filter image
            decoded_het_ctf = layers.Lambda(self.generator.ctfFilterImage)(decoded_het)
        else:
            decoded_het_ctf = decoded_het

        self.decode_het = Model(latent, delta_het, name="decoder_het")
        self.decoder = Model([rows, shifts, latent], [decoded_het, decoded_het_ctf, delta_het], name="decoder")

    def eval_volume_het(self, x_het, filter=True, only_pos=False):
        batch_size = x_het.shape[0]

        delta_het = self.decode_het(x_het)

        # Update values within mask
        if self.generator.isFocused:
            flat_indices = tf.constant(self.generator.flat_indices, dtype=tf.int32)[:, None]
            fn = lambda inp: tf.scatter_nd(flat_indices, inp, [self.generator.cube])
            updates = tf.map_fn(fn, delta_het, fn_output_signature=tf.float32)
            updates = tf.gather(updates, self.generator.mask, axis=1)
        else:
            updates = delta_het

        values = tf.tile(self.generator.values[None, :], [batch_size, 1]) + updates

        # Coords indices
        o_z, o_y, o_x = (self.generator.full_indices[:, 0].astype(int), self.generator.full_indices[:, 1].astype(int),
                         self.generator.full_indices[:, 2].astype(int))

        # Get numpy volumes
        values = values.numpy()
        volume_grids = np.zeros((batch_size, self.generator.xsize, self.generator.xsize, self.generator.xsize),
                                dtype=np.float32)
        for idx in range(batch_size):
            volume_grids[idx, o_z, o_y, o_x] = values[idx]
            if filter:
                volume_grids[idx] = filterVol(volume_grids[idx])

            # Only for deconvolvers
            if not only_pos:
                neg_part = volume_grids[idx] * (volume_grids[idx] < 0.0)
            volume_grids[idx] = volume_grids[idx] * (volume_grids[idx] >= 0.0)

            # Deconvolvers
            # volume_grids[idx] = richardsonLucyDeconvolver(volume_grids[idx])
            # volume_grids[idx] = richardsonLucyBlindDeconvolver(volume_grids[idx], global_iter=5, iter=5)
            # volume_grids[idx] = deconvolveTV(volume_grids[idx], iterations=50, regularization_weight=0.001, lr=0.01)
            # volume_grids[idx] = tv_deconvolution_bregman(volume_grids[idx], iterations=50,
            #                                              regularization_weight=0.1, lr=0.01)

            if not only_pos:
                volume_grids[idx] += neg_part

        return volume_grids.astype(np.float32)

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class AutoEncoder(Model):
    def __init__(self, generator, het_dim=10, architecture="convnn", CTF="wiener", refPose=True,
                 l1_lambda=0.5, tv_lambda=0.5, mse_lambda=0.5, mode=None, train_size=None, only_pos=True,
                 multires_levels=None, poseReg=0.0, ctfReg=0.0, precision=tf.float32, precision_scaled=tf.float32,
                 use_hyper_network=True, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.precision = precision
        self.precision_scaled = precision_scaled
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.mode = generator.mode if mode is None else mode
        metadata = XmippMetaData(str(generator.filename))
        self.xsize = metadata.getMetaDataImage(0).shape[1] if metadata.binaries else generator.xsize
        self.encoder_exp = Encoder(het_dim, self.xsize, architecture=architecture,
                                   refPose=refPose, mode=self.mode, downsample=False)
        if poseReg > 0.0:
            self.encoder_clean = Encoder(het_dim, self.xsize, architecture=architecture,
                                         refPose=refPose, mode=self.mode, downsample=True)
        if ctfReg > 0.0:
            self.encoder_ctf = Encoder(het_dim, self.xsize, architecture=architecture,
                                       refPose=refPose, mode=self.mode, downsample=True)
        self.latent = layers.Dense(het_dim, activation="linear")
        self.rows = layers.Dense(3, activation="linear", trainable=refPose)
        self.shifts = layers.Dense(2, activation="linear", trainable=refPose)
        self.decoder = Decoder(het_dim, generator, CTF=CTF, use_hyper_network=use_hyper_network)
        self.refPose = 1.0 if refPose else 0.0
        self.l1_lambda = l1_lambda
        self.tv_lambda = tv_lambda
        self.mse_lambda = mse_lambda
        self.pose_lambda = poseReg
        self.ctf_lambda = ctfReg
        if self.mode == "tomo":
            max_lamba = max(poseReg, ctfReg)
            self.pose_lambda = max_lamba
            self.ctf_lambda = max_lamba
        self.het_dim = het_dim
        self.only_pos = only_pos
        self.train_size = train_size if train_size is not None else self.xsize
        self.multires_levels = multires_levels
        if multires_levels is None:
            self.filters = None
        else:
            self.filters = tf.cast(create_blur_filters(multires_levels, 10, 30), self.precision)
        self.disantangle_pose = poseReg > 0.0
        self.disantangle_ctf = ctfReg > 0.0
        self.isFocused = generator.isFocused
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_loss_tracker = tf.keras.metrics.Mean(name="test_loss")
        self.loss_het_tracker = tf.keras.metrics.Mean(name="rec_het")
        self.loss_disantangle_tracker = tf.keras.metrics.Mean(name="loss_disentangled")
        self.loss_hist_tracker = tf.keras.metrics.Mean(name="loss_hist")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.test_loss_tracker,
            self.loss_het_tracker,
            self.loss_disantangle_tracker,
            self.loss_hist_tracker,
        ]

    def train_step(self, data):
        inputs = data[0]

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

        images = tf.cast(images, self.precision)

        self.decoder.generator.indexes = indexes
        self.decoder.generator.current_images = images

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(images)[0]

        # Precompute batch aligments
        self.decoder.generator.rot_batch = tf.cast(tf.gather(self.decoder.generator.angle_rot, indexes, axis=0), self.precision)
        self.decoder.generator.tilt_batch = tf.cast(tf.gather(self.decoder.generator.angle_tilt, indexes, axis=0), self.precision)
        self.decoder.generator.psi_batch = tf.cast(tf.gather(self.decoder.generator.angle_psi, indexes, axis=0), self.precision)

        # Precompute batch shifts
        self.decoder.generator.shifts_batch = [tf.cast(tf.gather(self.decoder.generator.shift_x, indexes, axis=0), self.precision),
                                               tf.cast(tf.gather(self.decoder.generator.shift_y, indexes, axis=0), self.precision)]

        # Random permutations of angles and shifts
        euler_batch = tf.stack([self.decoder.generator.rot_batch,
                                self.decoder.generator.tilt_batch,
                                self.decoder.generator.psi_batch], axis=1)
        shifts_batch = tf.stack(self.decoder.generator.shifts_batch, axis=1)
        euler_batch_perm = tf.random.shuffle(euler_batch)
        shifts_batch_perm = tf.random.shuffle(shifts_batch)

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

        # Random permutations of CTF
        ctf_perm = tf.random.shuffle(ctf)

        # Wiener filter
        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)
            if self.mode == "spa":
                inputs = images
            elif self.mode == "tomo":
                inputs[0] = images

        with tf.GradientTape() as tape:
            # Forward pass (first encoder and decoder)
            l_rows, l_shifts, l_het, l_het_label = self.encoder_exp(inputs)
            het, rows, shifts = self.latent(l_het), self.rows(l_rows), self.shifts(l_shifts)

            if self.mode == "spa":
                decoded_het, decoded_het_ctf, delta_het = self.decoder(
                    [self.refPose * rows, self.refPose * shifts, het])

                if self.disantangle_pose:
                    # Forward pass (second encoder - no permutation)
                    _, _, l_het_clean, _ = self.encoder_clean(decoded_het)
                    het_clean = self.latent(l_het_clean)

                    # Forward pass (third encoder - permuted CTF)
                    if self.disantangle_ctf and self.CTF is not None:
                        _, _, l_het_ctf, _ = self.encoder_ctf(decoded_het_ctf)
                        het_ctf = self.latent(l_het_ctf)
                        self.decoder.generator.ctf = ctf_perm
                        decoded_het_ctf_perm = self.decoder.generator.ctfFilterImage(decoded_het)
                        _, _, l_het_ctf_perm, _ = self.encoder_ctf(decoded_het_ctf_perm)
                        het_ctf_perm = self.latent(l_het_ctf_perm)
                    else:
                        het_ctf_perm = het

                    # Forward pass (second encoder - permutation)
                    self.decoder.generator.rot_batch = euler_batch_perm[..., 0]
                    self.decoder.generator.tilt_batch = euler_batch_perm[..., 1]
                    self.decoder.generator.psi_batch = euler_batch_perm[..., 2]
                    self.decoder.generator.shifts_batch = [shifts_batch_perm[..., 0], shifts_batch_perm[..., 1]]
                    decoded_het, _, _ = self.decoder([self.refPose * rows, self.refPose * shifts, het])
                    _, _, l_het_clean_perm, _ = self.encoder_clean(decoded_het)
                    het_clean_perm = self.latent(l_het_clean_perm)

            elif self.mode == "tomo":
                het_label = self.latent(l_het_label)
                decoded_het, decoded_het_ctf, delta_het = self.decoder([self.refPose * rows, self.refPose * shifts, het_label])

            # delta_het = self.decoder.decode_het(het)

            # Update values within mask
            if self.isFocused:
                flat_indices = tf.constant(self.decoder.generator.flat_indices, dtype=tf.int32)[:, None]
                fn = lambda inp: tf.scatter_nd(flat_indices, inp, [self.decoder.generator.cube])
                updates = tf.map_fn(fn, delta_het, fn_output_signature=self.precision)
                updates = tf.gather(updates, self.decoder.generator.mask, axis=1)
            else:
                updates = delta_het

            # L1 penalization delta_het
            delta_het = tf.tile(self.decoder.generator.values[None, :], [batch_size_scope, 1]) + updates
            l1_loss_het = tf.cast(self.l1_lambda * tf.reduce_mean(tf.abs(tf.cast(delta_het, self.precision_scaled))), self.precision)

            # Volume range loss
            if self.decoder.generator.null_ref or not self.isFocused:
                hist_loss = 0.0
            else:
                orig_values = tf.tile(self.decoder.generator.values_no_masked[None, :], [batch_size_scope, 1])
                values_in_het = tf.cast(tf.gather(delta_het, self.decoder.generator.values_in_mask, axis=1), self.precision_scaled)
                values_in_mask = tf.cast(tf.gather(orig_values, self.decoder.generator.values_in_mask, axis=1), self.precision_scaled)
                # val_range = [tf.reduce_min(values_in_mask), tf.reduce_max(values_in_mask)]
                # hist_loss = tf.keras.losses.MSE(
                #     compute_histogram(values_in_het, bins=50, minval=val_range[0], maxval=val_range[1]),
                #     compute_histogram(values_in_mask, bins=50, minval=val_range[0], maxval=val_range[1])
                # )
                hist_loss = tf.cast(tf.keras.losses.MSE(tf.reduce_max(values_in_het, axis=1), tf.reduce_max(values_in_mask))
                                    + tf.keras.losses.MSE(tf.reduce_min(values_in_het, axis=1), tf.reduce_min(values_in_mask))
                                    + tf.keras.losses.MSE(tf.reduce_mean(values_in_het, axis=1), tf.reduce_mean(values_in_mask))
                                    + tf.keras.losses.MSE(tf.math.reduce_std(values_in_het, axis=1), tf.math.reduce_std(values_in_mask)), self.precision)

            # Total variation and MSE losses
            tv_loss, d_mse_loss = densitySmoothnessVolume(self.decoder.generator.xsize,
                                                          self.decoder.generator.full_indices, delta_het, self.precision)
            tv_loss *= self.tv_lambda
            d_mse_loss *= self.mse_lambda

            # Negative loss
            mask = tf.less(delta_het, 0.0)
            mask_has_values = tf.cast(tf.reduce_any(mask, axis=-1), self.precision)
            delta_neg = tf.boolean_mask(delta_het, mask)
            delta_neg = tf.reduce_mean(tf.abs(tf.cast(delta_neg, self.precision_scaled)))
            neg_loss_het = mask_has_values * tf.cast(self.l1_lambda * delta_neg, self.precision)

            # # Positive loss
            # mask = tf.greater(delta_het, 0.0)
            # delta_pos = tf.boolean_mask(delta_het, mask)
            # delta_pos_size = tf.cast(tf.shape(delta_pos)[-1], dtype=self.precision_scaled)
            # delta_pos = tf.reduce_mean(tf.abs(delta_pos))
            # pos_loss_het = self.l1_lambda * delta_pos / delta_pos_size

            # Reconstruction mask for projections (Decoder size)
            mask_imgs = self.decoder.generator.resizeImageFourier(self.decoder.generator.mask_imgs,
                                                                  self.decoder.generator.xsize)
            mask_imgs = tf.abs(mask_imgs)
            mask_imgs = tf.math.divide_no_nan(mask_imgs, mask_imgs)

            # Reconstruction loss for original size images
            images_masked = mask_imgs * self.decoder.generator.resizeImageFourier(images, self.decoder.generator.xsize)
            loss_het_ori = tf.cast(self.decoder.generator.cost_function(tf.cast(images_masked, self.precision_scaled),
                                                                        tf.cast(decoded_het_ctf, self.precision_scaled)), self.precision)

            # Reconstruction mask for projections (Train size)
            mask_imgs = self.decoder.generator.resizeImageFourier(self.decoder.generator.mask_imgs, self.train_size)
            mask_imgs = tf.abs(mask_imgs)
            mask_imgs = tf.math.divide_no_nan(mask_imgs, mask_imgs)

            # Reconstruction loss for downscaled images
            images_masked = mask_imgs * self.decoder.generator.resizeImageFourier(images, self.train_size)
            decoded_het_scl = self.decoder.generator.resizeImageFourier(decoded_het_ctf, self.train_size)
            loss_het_scl = tf.cast(self.decoder.generator.cost_function(tf.cast(images_masked, self.precision_scaled),
                                                                        tf.cast(decoded_het_scl, self.precision_scaled)), self.precision)

            # MR loss
            if self.filters is not None:
                filt_images = apply_blur_filters_to_batch(images, self.filters)
                filt_decoded = apply_blur_filters_to_batch(decoded_het_ctf, self.filters)
                for idx in range(self.multires_levels):
                    loss_het_ori += tf.cast(self.decoder.generator.cost_function(tf.cast(filt_images, self.precision_scaled)[..., idx][..., None],
                                                                                 tf.cast(filt_decoded, self.precision_scaled)[..., idx][..., None]), self.precision)
                loss_het_ori = tf.cast(tf.cast(loss_het_ori, self.precision_scaled) / (float(self.multires_levels) + 1), self.precision)

            # Loss disantagled (pose)
            if self.disantangle_pose and self.mode == "spa":
                loss_disantagled_pose = tf.cast(tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_clean, self.precision_scaled))
                                         + tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_clean_perm, self.precision_scaled)), self.precision)
            elif self.mode == "tomo":
                loss_disantagled_pose = 0.5 * tf.cast(
                    tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_label, self.precision_scaled)),
                    self.precision)
            else:
                loss_disantagled_pose = 0.0

            # Loss disantagled (CTF)
            if self.disantangle_ctf and self.mode == "spa" and self.CTF is not None:
                loss_disantagled_ctf = tf.cast(tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_ctf, self.precision_scaled))
                                        + tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_ctf_perm, self.precision_scaled)), self.precision)
            elif self.mode == "tomo":
                loss_disantagled_ctf = 0.5 * tf.cast(
                    tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_label, self.precision_scaled)),
                    self.precision)
            else:
                loss_disantagled_ctf = 0.0

            # Final losses
            rec_loss = loss_het_ori + loss_het_scl
            reg_loss = l1_loss_het + neg_loss_het + tv_loss + d_mse_loss

            total_loss = (rec_loss + reg_loss + self.pose_lambda * loss_disantagled_pose
                          + self.ctf_lambda * loss_disantagled_ctf + 100.0 * hist_loss)
            # scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        # scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        # gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.loss_het_tracker.update_state(rec_loss)
        self.loss_disantangle_tracker.update_state(loss_disantagled_pose + loss_disantagled_ctf)
        self.loss_hist_tracker.update_state(hist_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.loss_het_tracker.result(),
            "loss_disentangled": self.loss_disantangle_tracker.result(),
            "loss_hist": self.loss_hist_tracker.result(),
        }

    def test_step(self, data):
        inputs = data[0]

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

        images = tf.cast(images, self.precision)

        self.decoder.generator.indexes = indexes
        self.decoder.generator.current_images = images

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(images)[0]

        # Precompute batch aligments
        self.decoder.generator.rot_batch = tf.cast(tf.gather(self.decoder.generator.angle_rot, indexes, axis=0),
                                                   self.precision)
        self.decoder.generator.tilt_batch = tf.cast(tf.gather(self.decoder.generator.angle_tilt, indexes, axis=0),
                                                    self.precision)
        self.decoder.generator.psi_batch = tf.cast(tf.gather(self.decoder.generator.angle_psi, indexes, axis=0),
                                                   self.precision)

        # Precompute batch shifts
        self.decoder.generator.shifts_batch = [
            tf.cast(tf.gather(self.decoder.generator.shift_x, indexes, axis=0), self.precision),
            tf.cast(tf.gather(self.decoder.generator.shift_y, indexes, axis=0), self.precision)]

        # Random permutations of angles and shifts
        euler_batch = tf.stack([self.decoder.generator.rot_batch,
                                self.decoder.generator.tilt_batch,
                                self.decoder.generator.psi_batch], axis=1)
        shifts_batch = tf.stack(self.decoder.generator.shifts_batch, axis=1)
        euler_batch_perm = tf.random.shuffle(euler_batch)
        shifts_batch_perm = tf.random.shuffle(shifts_batch)

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

        # Random permutations of CTF
        ctf_perm = tf.random.shuffle(ctf)

        # Wiener filter
        if self.CTF == "wiener":
            images = self.decoder.generator.wiener2DFilter(images)
            if self.mode == "spa":
                inputs = images
            elif self.mode == "tomo":
                inputs[0] = images

            # Forward pass (first encoder and decoder)
        l_rows, l_shifts, l_het, l_het_label = self.encoder_exp(inputs)
        het, rows, shifts = self.latent(l_het), self.rows(l_rows), self.shifts(l_shifts)

        if self.mode == "spa":
            decoded_het, decoded_het_ctf, delta_het = self.decoder(
                [self.refPose * rows, self.refPose * shifts, het])

            if self.disantangle_pose:
                # Forward pass (second encoder - no permutation)
                _, _, l_het_clean, _ = self.encoder_clean(decoded_het)
                het_clean = self.latent(l_het_clean)

                # Forward pass (third encoder - permuted CTF)
                if self.disantangle_ctf and self.CTF is not None:
                    _, _, l_het_ctf, _ = self.encoder_ctf(decoded_het_ctf)
                    het_ctf = self.latent(l_het_ctf)
                    self.decoder.generator.ctf = ctf_perm
                    decoded_het_ctf_perm = self.decoder.generator.ctfFilterImage(decoded_het)
                    _, _, l_het_ctf_perm, _ = self.encoder_ctf(decoded_het_ctf_perm)
                    het_ctf_perm = self.latent(l_het_ctf_perm)
                else:
                    het_ctf_perm = het

                # Forward pass (second encoder - permutation)
                self.decoder.generator.rot_batch = euler_batch_perm[..., 0]
                self.decoder.generator.tilt_batch = euler_batch_perm[..., 1]
                self.decoder.generator.psi_batch = euler_batch_perm[..., 2]
                self.decoder.generator.shifts_batch = [shifts_batch_perm[..., 0], shifts_batch_perm[..., 1]]
                decoded_het, _, _ = self.decoder([self.refPose * rows, self.refPose * shifts, het])
                _, _, l_het_clean_perm, _ = self.encoder_clean(decoded_het)
                het_clean_perm = self.latent(l_het_clean_perm)

        elif self.mode == "tomo":
            het_label = self.latent(l_het_label)
            decoded_het, decoded_het_ctf, delta_het = self.decoder(
                [self.refPose * rows, self.refPose * shifts, het_label])

        # delta_het = self.decoder.decode_het(het)

        # Update values within mask
        if self.isFocused:
            flat_indices = tf.constant(self.decoder.generator.flat_indices, dtype=tf.int32)[:, None]
            fn = lambda inp: tf.scatter_nd(flat_indices, inp, [self.decoder.generator.cube])
            updates = tf.map_fn(fn, delta_het, fn_output_signature=self.precision)
            updates = tf.gather(updates, self.decoder.generator.mask, axis=1)
        else:
            updates = delta_het

        # L1 penalization delta_het
        delta_het = tf.tile(self.decoder.generator.values[None, :], [batch_size_scope, 1]) + updates
        l1_loss_het = tf.cast(self.l1_lambda * tf.reduce_mean(tf.abs(tf.cast(delta_het, self.precision_scaled))),
                              self.precision)

        # Volume range loss
        if self.decoder.generator.null_ref or not self.isFocused:
            hist_loss = 0.0
        else:
            orig_values = tf.tile(self.decoder.generator.values_no_masked[None, :], [batch_size_scope, 1])
            values_in_het = tf.cast(tf.gather(delta_het, self.decoder.generator.values_in_mask, axis=1), self.precision_scaled)
            values_in_mask = tf.cast(tf.gather(orig_values, self.decoder.generator.values_in_mask, axis=1),
                                     self.precision_scaled)
            # val_range = [tf.reduce_min(values_in_mask), tf.reduce_max(values_in_mask)]
            # hist_loss = tf.keras.losses.MSE(
            #     compute_histogram(values_in_het, bins=50, minval=val_range[0], maxval=val_range[1]),
            #     compute_histogram(values_in_mask, bins=50, minval=val_range[0], maxval=val_range[1])
            # )
            hist_loss = tf.cast(
                tf.keras.losses.MSE(tf.reduce_max(values_in_het, axis=1), tf.reduce_max(values_in_mask))
                + tf.keras.losses.MSE(tf.reduce_min(values_in_het, axis=1), tf.reduce_min(values_in_mask))
                + tf.keras.losses.MSE(tf.reduce_mean(values_in_het, axis=1), tf.reduce_mean(values_in_mask))
                + tf.keras.losses.MSE(tf.math.reduce_std(values_in_het, axis=1),
                                      tf.math.reduce_std(values_in_mask)), self.precision)

        # Total variation and MSE losses
        tv_loss, d_mse_loss = densitySmoothnessVolume(self.decoder.generator.xsize,
                                                      self.decoder.generator.full_indices, delta_het,
                                                      self.precision)
        tv_loss *= self.tv_lambda
        d_mse_loss *= self.mse_lambda

        # Negative loss
        mask = tf.less(delta_het, 0.0)
        mask_has_values = tf.cast(tf.reduce_any(mask, axis=-1), self.precision)
        delta_neg = tf.boolean_mask(delta_het, mask)
        delta_neg = tf.reduce_mean(tf.abs(tf.cast(delta_neg, self.precision_scaled)))
        neg_loss_het = mask_has_values * tf.cast(self.l1_lambda * delta_neg, self.precision)

        # # Positive loss
        # mask = tf.greater(delta_het, 0.0)
        # delta_pos = tf.boolean_mask(delta_het, mask)
        # delta_pos_size = tf.cast(tf.shape(delta_pos)[-1], dtype=self.precision_scaled)
        # delta_pos = tf.reduce_mean(tf.abs(delta_pos))
        # pos_loss_het = self.l1_lambda * delta_pos / delta_pos_size

        # Reconstruction mask for projections (Decoder size)
        mask_imgs = self.decoder.generator.resizeImageFourier(self.decoder.generator.mask_imgs,
                                                              self.decoder.generator.xsize)
        mask_imgs = tf.abs(mask_imgs)
        mask_imgs = tf.math.divide_no_nan(mask_imgs, mask_imgs)

        # Reconstruction loss for original size images
        images_masked = mask_imgs * self.decoder.generator.resizeImageFourier(images, self.decoder.generator.xsize)
        loss_het_ori = tf.cast(self.decoder.generator.cost_function(tf.cast(images_masked, self.precision_scaled),
                                                                    tf.cast(decoded_het_ctf, self.precision_scaled)),
                               self.precision)

        # Reconstruction mask for projections (Train size)
        mask_imgs = self.decoder.generator.resizeImageFourier(self.decoder.generator.mask_imgs, self.train_size)
        mask_imgs = tf.abs(mask_imgs)
        mask_imgs = tf.math.divide_no_nan(mask_imgs, mask_imgs)

        # Reconstruction loss for downscaled images
        images_masked = mask_imgs * self.decoder.generator.resizeImageFourier(images, self.train_size)
        decoded_het_scl = self.decoder.generator.resizeImageFourier(decoded_het_ctf, self.train_size)
        loss_het_scl = tf.cast(self.decoder.generator.cost_function(tf.cast(images_masked, self.precision_scaled),
                                                                    tf.cast(decoded_het_scl, self.precision_scaled)),
                               self.precision)

        # MR loss
        if self.filters is not None:
            filt_images = apply_blur_filters_to_batch(images, self.filters)
            filt_decoded = apply_blur_filters_to_batch(decoded_het_ctf, self.filters)
            for idx in range(self.multires_levels):
                loss_het_ori += tf.cast(
                    self.decoder.generator.cost_function(tf.cast(filt_images, self.precision_scaled)[..., idx][..., None],
                                                         tf.cast(filt_decoded, self.precision_scaled)[..., idx][..., None]),
                    self.precision)
            loss_het_ori = tf.cast(tf.cast(loss_het_ori, self.precision_scaled) / (float(self.multires_levels) + 1),
                                   self.precision)

        # Loss disantagled (pose)
        if self.disantangle_pose and self.mode == "spa":
            loss_disantagled_pose = tf.cast(
                tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_clean, self.precision_scaled))
                + tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_clean_perm, self.precision_scaled)),
                self.precision)
        elif self.mode == "tomo":
            loss_disantagled_pose = 0.5 * tf.cast(
                tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_label, self.precision_scaled)),
                self.precision)
        else:
            loss_disantagled_pose = 0.0

        # Loss disantagled (CTF)
        if self.disantangle_ctf and self.mode == "spa" and self.CTF is not None:
            loss_disantagled_ctf = tf.cast(
                tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_ctf, self.precision_scaled))
                + tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_ctf_perm, self.precision_scaled)), self.precision)
        elif self.mode == "tomo":
            loss_disantagled_ctf = 0.5 * tf.cast(
                tf.keras.losses.MSE(tf.cast(het, self.precision_scaled), tf.cast(het_label, self.precision_scaled)),
                self.precision)
        else:
            loss_disantagled_ctf = 0.0

        # Final losses
        rec_loss = loss_het_ori + loss_het_scl
        reg_loss = l1_loss_het + neg_loss_het + tv_loss + d_mse_loss

        total_loss = (rec_loss + reg_loss + self.pose_lambda * loss_disantagled_pose
                      + self.ctf_lambda * loss_disantagled_ctf + 100.0 * hist_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.loss_het_tracker.update_state(rec_loss)
        self.loss_disantangle_tracker.update_state(loss_disantagled_pose + loss_disantagled_ctf)
        self.loss_hist_tracker.update_state(hist_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.loss_het_tracker.result(),
            "loss_disentangled": self.loss_disantangle_tracker.result(),
            "loss_hist": self.loss_hist_tracker.result(),
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

        l_rot, l_shifts, l_het = self.encoder_exp(x[0])
        het, rot, shifts = self.latent(l_het), self.rows(l_rot), self.shifts(l_shifts)

        return self.refPose * rot.numpy(), self.refPose * shifts.numpy(), het.numpy()

    def eval_volume_het(self, x_het, allCoords=False, filter=True, only_pos=False, add_to_original=False):
        batch_size = x_het.shape[0]

        if allCoords and self.decoder.generator.step > 1:
            new_coords, prev_coords = self.decoder.generator.getAllCoordsMask(), \
                self.decoder.generator.coords
        else:
            new_coords = [self.decoder.generator.coords]

        # Read original volume (if needed)
        volume_path = Path(self.decoder.generator.filename.parent, 'volume.mrc')
        if add_to_original and volume_path.exists():
            original_volume = ImageHandler(str(volume_path)).getData()
            original_volume = np.tile(original_volume[None, ...], (x_het.shape[0], 1, 1, 1))
        else:
            original_volume = None

        # Volume
        volume = np.zeros((batch_size, self.decoder.generator.xsize,
                           self.decoder.generator.xsize,
                           self.decoder.generator.xsize), dtype=np.float32)
        for coords in new_coords:
            self.decoder.generator.coords = coords
            volume += self.decoder.eval_volume_het(x_het, filter=filter, only_pos=only_pos)

        # if original_volume is not None:
        #     original_norm = match_histograms(original_volume, volume)
        #     # original_norm = normalize_to_other_volumes(volume, original_volume)
        #     volume = original_norm + volume

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
            l_rot, l_shifts, l_het, l_l_het = self.encoder_exp(inputs)
            if self.mode == "spa":
                het, rot, shifts = self.latent(l_het), self.rows(l_rot), self.shifts(l_shifts)
                return rot, shifts, het
            elif self.mode == "tomo":
                het, het_l, rot, shifts = self.latent(l_het), self.latent(l_l_het), self.rows(l_rot), self.shifts(l_shifts)
                return rot, shifts, het, het_l
        elif "particles" in self.predict_mode:
            l_rot, l_shifts, l_het, _ = self.encoder_exp(inputs)
            het, rot, shifts = self.latent(l_het), self.rows(l_rot), self.shifts(l_shifts)
            if "ctf" in self.predict_mode:
                return self.decoder([rot, shifts, het])[1]
            else:
                return self.decoder([rot, shifts, het])[0]
        else:
            raise ValueError("Prediction mode not understood!")

    def call(self, input_features):
        l_rot, l_shifts, l_het, _ = self.encoder_exp(input_features)
        # _ = self.encoder_clean(input_features)
        # _ = self.encoder_ctf(input_features)
        het, rot, shifts = self.latent(l_het), self.rows(l_rot), self.shifts(l_shifts)
        decoded = self.decoder([rot, shifts, het])
        return decoded

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
from tensorflow.python.ops.linalg.linalg_impl import diag_part
from tensorflow.python.ops.array_ops import shape_internal


class Generator(tf.keras.utils.Sequence):
    def __init__(self, dataset, latent_dim, batch_size=8, shuffle=True, splitTrain=0.8):
        # Attributes
        self.dataset = dataset
        self.space_dims = [space.shape[1] for space in dataset]
        self.lat_dim = latent_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_idx = np.arange(self.dataset[0].shape[0])

        # Get train dataset
        if splitTrain < 1.0:
            self.getTrainDataset(splitTrain)


    #----- Initialization methods -----#

    def getTrainDataset(self, splitTrain):
        indexes = np.arange(self.list_idx.size)
        np.random.shuffle(indexes)
        self.list_idx = self.list_idx[indexes[:int(splitTrain * indexes.size)]]

    # ----- -------- -----#


    # ----- Data generation methods -----#

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.list_idx)

    def __data_generation(self):
        inputs = []
        for dataset in self.dataset:
            inputs.append(dataset[self.indexes])
        return inputs, self.indexes

    def __getitem__(self, index):
        # Generate indexes of the batch
        self.indexes = self.list_idx[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation()
        return X, y

    def __len__(self):
        return int(np.ceil(self.list_idx.size / self.batch_size))

    # ----- -------- -----#


    # ----- Utils -----#

    @tf.function()
    def pairwise_distances(self, embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = diag_part(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances

    def rmse(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=1) / x.shape[1])

    def hist_match(self, source, template):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    # ----- -------- -----#


    # ----- Losses -----#

    @tf.function()
    def compute_encoder_loss(self, inputs):
        encoder_loss = tf.constant(0.0, dtype=tf.float32)
        for space_1 in inputs:
            for space_2 in inputs:
                loss_1 = tf.losses.mean_squared_error(space_1, space_2)

                dmat1 = self.pairwise_distances(space_1)
                dmat2 = self.pairwise_distances(space_2)
                dmat1 = self.normalize_distributions(dmat1)
                dmat2 = self.normalize_distributions(dmat2)
                # loss_2 = 1.0 * -self.spearman_rank_correlation(dmat1, dmat2)

                hist1 = self.compute_histogram(tf.reshape(dmat1, [-1]), 20)
                hist2 = self.compute_histogram(tf.reshape(dmat2, [-1]), 20)
                loss_3 = 1.0 * self.wasserstein_distance(hist1, hist2)

                encoder_loss += loss_1 + loss_3
        # encoder_loss *= 0.5
        return encoder_loss

    @tf.function()
    def compute_shannon_loss(self, inputs, predictions):
        encoder_loss = tf.constant(0.0, dtype=tf.float32)
        for input_data in inputs:
            dist_input = self.pairwise_distances(input_data)
            dist_space = self.pairwise_distances(predictions)
            mask_epsilon_distance = tf.fill(shape_internal(dist_space), tf.cast(1e-5, dtype=tf.float32))
            div_dist_mat_true = tf.multiply(dist_input,
                                            tf.cast(tf.greater(dist_input, mask_epsilon_distance),
                                                    dtype=tf.float32))
            upper_mat_pred = tf.linalg.band_part(dist_space, 0, -1) - tf.linalg.band_part(dist_space, 0, 0)
            div_upper_mat_true = tf.linalg.band_part(div_dist_mat_true, 0, -1) - tf.linalg.band_part(
                div_dist_mat_true, 0, 0)
            aux_1 = tf.reduce_sum(
                tf.math.divide_no_nan(tf.square(div_upper_mat_true - upper_mat_pred), div_upper_mat_true))
            aux_2 = 1. / tf.reduce_sum(div_upper_mat_true)
            encoder_loss += aux_2 * aux_1
        return encoder_loss

    def normalize_distributions(self, dist):
        dist = dist - tf.reduce_mean(dist)
        return dist / tf.math.reduce_std(dist)

    @tf.function
    def spearman_rank_correlation(self, dmat1, dmat2):
        """Compute the Spearman rank correlation between two distance matrices."""

        def rank_tensor(tensor):
            """Return the ranks of the values in the tensor."""
            flat_tensor = tf.reshape(tensor, [-1])
            sorted_indices = tf.argsort(flat_tensor)
            rank_indices = tf.argsort(sorted_indices)
            return tf.cast(tf.reshape(rank_indices, tf.shape(tensor)), tf.float32)

        rank1 = rank_tensor(dmat1)
        rank2 = rank_tensor(dmat2)

        covariance = tf.reduce_mean((rank1 - tf.reduce_mean(rank1)) * (rank2 - tf.reduce_mean(rank2)))
        stddev1 = tf.math.reduce_std(rank1)
        stddev2 = tf.math.reduce_std(rank2)
        correlation = covariance / (stddev1 * stddev2)
        # return tf.clip_by_value(correlation, -1.0, 1.0)
        return correlation

    def spearman_correlation_loss(self, tensor1, tensor2):
        encoder_loss = tf.constant(0.0, dtype=tf.float32)
        for input_data, space in zip(tensor1, tensor2):
            # for space in tensor2:
            """Custom loss function to minimize the dissimilarity between distance distributions."""
            dmat1 = self.pairwise_distances(input_data)
            dmat2 = self.pairwise_distances(space)
            dmat1 = self.normalize_distributions(dmat1)
            dmat2 = self.normalize_distributions(dmat2)
            encoder_loss += -self.spearman_rank_correlation(dmat1, dmat2)
        return encoder_loss

    def compute_histogram(self, distances, bins):
        """Compute the histogram of the distances."""
        histogram = tf.histogram_fixed_width(distances, [0.0, tf.reduce_max(distances)], nbins=bins)
        histogram = tf.math.divide_no_nan(tf.cast(histogram, tf.float32), tf.cast(tf.reduce_sum(histogram), tf.float32))
        return histogram

    def wasserstein_distance(self, hist1, hist2):
        """Compute the Wasserstein distance between two histograms."""
        return tf.reduce_sum(tf.abs(tf.cumsum(hist1) - tf.cumsum(hist2)))

    def wasserstein_distance_loss(self, inputs, encoded, bins=20):
        encoder_loss = tf.constant(0.0, dtype=tf.float32)
        dmat1 = self.pairwise_distances(encoded)
        dmat1 = self.normalize_distributions(dmat1)
        for input_data in inputs:
            # for space in tensor2:
            """Custom loss function to minimize the dissimilarity between distance distributions."""
            dmat2 = self.pairwise_distances(input_data)
            dmat2 = self.normalize_distributions(dmat2)

            hist1 = self.compute_histogram(tf.reshape(dmat1, [-1]), bins)
            hist2 = self.compute_histogram(tf.reshape(dmat2, [-1]), bins)

            encoder_loss += self.wasserstein_distance(hist1, hist2)

        return encoder_loss

    def compute_centering_loss(self, tensor):
        loss = tf.reduce_sum(tf.square(tf.reduce_mean(tensor, axis=0)))
        return loss

    @tf.function()
    def compute_decoder_loss(self, inputs, predictions):
        loss = tf.constant(0.0, dtype=tf.float32)
        for data, space in zip(inputs, predictions):
            loss += tf.losses.mean_squared_error(data, space)
            # loss += tf.losses.mean_absolute_error(inputs, space)
        loss /= len(predictions)
        return loss

    def logistic_transform_std_shift(self, errors, mu=None, sigma=None):
        """
        Transforms a 1D array of errors to a 0-1 scale using a logistic function,
        such that a value of 0.5 corresponds to errors that are +1 standard deviation away from the mean.

        The transformation is defined as:
            f(x) = 1 / (1 + exp(-k*(x - (mu+sigma))))
        with k chosen so that:
            f(mu)   ~ 0.25  and  f(mu+2*sigma) ~ 0.75.

        Parameters:
            errors (np.array): 1D array of error values.

        Returns:
            np.array: Transformed error values in the interval (0, 1).
        """
        # Compute the mean and standard deviation
        mu = np.mean(errors) if mu is None else mu
        sigma = np.std(errors) if sigma is None else sigma

        # Prevent division by zero in case sigma is zero
        if sigma == 0:
            return np.full_like(errors, 0.5)

        # Choose k such that one std below and above the center give 0.25 and 0.75 respectively.
        k = np.log(3) / sigma

        # Shift the logistic function center to mu + sigma so that f(mu+sigma) = 0.5
        transformed = 1 / (1 + np.exp(-k * (errors - (mu + sigma))))
        return transformed

    # ----- -------- -----#

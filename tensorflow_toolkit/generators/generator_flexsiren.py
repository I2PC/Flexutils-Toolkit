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


from packaging import version
import numpy as np
import mrcfile
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

import tensorflow as tf

try:
    import open3d.ml.tf as ml3d
    allow_open3d = True
except ImportError:
    allow_open3d = False
    print("Open3D has not been installed. The program will continue without this package")

from tensorflow_toolkit.generators.generator_template import DataGeneratorBase
from tensorflow_toolkit.utils import basisDegreeVectors, computeBasis, euler_matrix_batch, fft_pad, ifft_pad


@tf.function
def compute_energy(conv, points, queries, extent):
    fn = lambda x: conv(x[0], x[1], x[2], extent)
    energy = tf.map_fn(fn, [points, points, queries], fn_output_signature=tf.float32)
    return energy


class Generator(DataGeneratorBase):
    def __init__(self, batch_coords=0, refinePose=True, cap_def=False, precision=tf.float32, **kwargs):
        super().__init__(**kwargs)

        self.batch_coords = batch_coords
        self.refinePose = refinePose
        self.cap_def = cap_def
        self.shuf_order = tf.range(self.coords.shape[0], dtype=tf.int32)
        self.unshuf_order = tf.range(self.coords.shape[0], dtype=tf.int32)
        self.precision = precision

        # Save mask map and indices
        mask_path = Path(self.filename.parent, 'mask.mrc')
        with mrcfile.open(mask_path) as mrc:
            self.mask_map = mrc.data
            coords = np.asarray(np.where(mrc.data == 1))
            self.indices = tf.constant(coords.T, dtype=tf.int32)
        self.total_voxels = self.indices.shape[0]

        # Get coords group
        mask_file = Path(Path(kwargs.get("md_file")).parent, 'mask.mrc')
        if self.ref_is_struct:
            groups, centers = None, None
        else:
            groups, centers = self.getCoordsGroup(mask_file)

        # Precompute Zernike3D basis
        self.half_xsize = 0.5 * self.xsize
        # self.scaled_coords = tf.cast(self.coords_to_9D(self.coords / self.half_xsize), self.precision)
        # self.scaled_coords = tf.constant(self.positional_encode_coords(self.coords / self.half_xsize), self.precision)
        self.scaled_coords = tf.constant(self.coords / self.half_xsize, dtype=self.precision)

        if self.ref_is_struct:
            # self.scaled_atom_coords = tf.cast(self.coords_to_9D(self.atom_coords, self.half_xsize), self.precision)
            self.scaled_atom_coords = tf.constant(self.atom_coords / self.half_xsize, dtype=self.precision)

        # Initialize weightd
        # self.weight_initializer = "glorot_uniform"
        self.weight_initializer = tf.keras.initializers.RandomUniform(minval=-0.001, maxval=0.001,
                                                                      seed=None)

        # Initialize pose information
        if refinePose:
            self.rot_batch = np.zeros(self.batch_size)
            self.tilt_batch = np.zeros(self.batch_size)
            self.psi_batch = np.zeros(self.batch_size)

        # Initial bonds and angles
        if self.ref_is_struct:
            coords = [self.atom_coords[:, 0][..., None], self.atom_coords[:, 1][..., None],
                      self.atom_coords[:, 2][..., None]]
            self.angle0 = self.calcAngle(coords)
            self.bond0 = self.calcBond(coords)
        else:
            self.angle0 = 0.0
            self.bond0 = 0.0

        # Train coords
        # self.select_train_coords()


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

    def coords_to_9D(self, coords):
        coords_corner_1 = coords + 1.0
        coords_corner_2 = coords - 1.0
        spherical = self.cartesian_to_spherical(coords)
        return tf.concat([coords_corner_1, coords_corner_2, spherical], axis=-1)

    def positional_encode_coords(self, coords):
        self.get_sinusoid_encoding_table(np.amax(self.indices) + 1, 100)
        pe_x = self.sinusoid_table[self.indices[:, 2]]
        pe_y = self.sinusoid_table[self.indices[:, 1]]
        pe_z = self.sinusoid_table[self.indices[:, 0]]
        return np.concatenate([coords, pe_x, pe_y, pe_z], axis=-1)

    def cartesian_to_spherical(self, coords):
        coords_2 = coords ** 2.0

        xy = tf.sqrt(coords_2[..., 0] + coords_2[..., 1])

        r = tf.sqrt(coords_2[..., 0] + coords_2[..., 1] + coords_2[..., 2])
        phi = tf.atan2(coords_2[..., 1], coords_2[..., 0])
        theta = tf.atan2(xy, coords_2[..., 2])

        return tf.stack([r, phi, theta], axis=-1)

    def circular_shuffle_indices(self):
        if self.coords.shape[0] > self.batch_coords:
            shuf_order = np.arange(self.coords.shape[0])
            np.random.shuffle(shuf_order)

            unshuf_order = np.zeros_like(shuf_order)
            unshuf_order[shuf_order] = np.arange(self.coords.shape[0])

            self.shuf_order = tf.constant(shuf_order, dtype=tf.int32)
            self.unshuf_order = tf.constant(unshuf_order, dtype=tf.int32)
        else:
            self.shuf_order = tf.range(self.coords.shape[0], dtype=tf.int32)
            self.unshuf_order = tf.range(self.coords.shape[0], dtype=tf.int32)

    def select_train_coords(self):
        # Apply KMeans to find marker positions
        kmeans = KMeans(n_clusters=self.batch_coords).fit(self.scaled_coords)
        marker_coords = kmeans.cluster_centers_

        # Find the indices within coords where marker_coords are located
        _, marker_indices = KDTree(self.scaled_coords).query(marker_coords, k=1)
        marker_indices = np.squeeze(marker_indices)

        # Create a new array of coords with marker_coords first, followed by the remaining coordinates
        shuf_order = np.hstack((marker_indices, np.delete(np.arange(len(self.scaled_coords)).astype(int), marker_indices, axis=0)))
        self.shuf_order = tf.constant(shuf_order.flatten(), dtype=tf.int32)

        # Create an indices array to reorder new_coords back to the original order of coords
        unshuf_order = np.argsort(np.hstack((marker_indices, np.setdiff1d(np.arange(len(self.scaled_coords)), marker_indices))))
        self.unshuf_order = tf.constant(unshuf_order.flatten(), dtype=tf.int32)

    # ----- -------- -----#

    # ----- Utils -----#

    def ctfFilterImage(self, images, ctf):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(images)[0]

        # Sizes
        pad_size = tf.constant(int(self.pad_factor * self.xsize), dtype=tf.int32)
        size = tf.constant(int(self.xsize), dtype=tf.int32)

        # ft_images = tf.signal.fftshift(tf.signal.rfft2d(images[:, :, :, 0]))
        ft_images = fft_pad(images, pad_size, pad_size)
        ft_ctf_images_real = tf.multiply(tf.math.real(ft_images), ctf)
        ft_ctf_images_imag = tf.multiply(tf.math.imag(ft_images), ctf)
        ft_ctf_images = tf.complex(ft_ctf_images_real, ft_ctf_images_imag)
        # ctf_images = tf.signal.irfft2d(tf.signal.ifftshift(ft_ctf_images))
        ctf_images = ifft_pad(ft_ctf_images, size, size)
        return tf.reshape(ctf_images, [batch_size_scope, self.xsize, self.xsize, 1])

    def wiener2DFilter(self, images, ctf):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(images)[0]

        # Sizes
        pad_size = tf.constant(int(self.pad_factor * self.xsize), dtype=tf.int32)
        size = tf.constant(int(self.xsize), dtype=tf.int32)

        ctf_2 = ctf * ctf
        # epsilon = 1e-5
        epsilon = 0.1 * tf.reduce_mean(ctf_2)

        ft_images = fft_pad(images, pad_size, pad_size)
        ft_w_images_real = tf.math.real(ft_images) * ctf / (ctf_2 + epsilon)
        ft_w_images_imag = tf.math.imag(ft_images) * ctf / (ctf_2 + epsilon)
        ft_w_images = tf.complex(ft_w_images_real, ft_w_images_imag)
        w_images = ifft_pad(ft_w_images, size, size)
        return tf.reshape(w_images, [batch_size_scope, self.xsize, self.xsize, 1])

    def computeDeformationFieldVol(self, z):
        Z = tf.constant(self.Z, dtype=tf.float32)
        d = tf.matmul(Z, tf.transpose(z))
        return d

    def computeDeformationFieldAtoms(self, z):
        Z = tf.constant(self.Z_atoms, dtype=tf.float32)
        d = tf.matmul(Z, tf.transpose(z))
        return d

    def applyDeformationFieldVol(self, d, axis):
        coords = tf.constant(self.coords, dtype=self.precision)
        coords_axis = tf.transpose(tf.gather(coords, axis, axis=1))
        return tf.add(coords_axis[:, None], d)

    def applyDeformationFieldAtoms(self, d, axis):
        coords = tf.constant(self.atom_coords, dtype=self.precision)
        coords_axis = tf.transpose(tf.gather(coords, axis, axis=1))
        return tf.add(coords_axis[:, None], d)

    def applyAlignmentMatrix(self, c, axis):
        c_r_1 = tf.multiply(c[0], tf.cast(tf.gather(self.r[axis], 0, axis=1), dtype=self.precision))
        c_r_2 = tf.multiply(c[1], tf.cast(tf.gather(self.r[axis], 1, axis=1), dtype=self.precision))
        c_r_3 = tf.multiply(c[2], tf.cast(tf.gather(self.r[axis], 2, axis=1), dtype=self.precision))
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyAlignmentDeltaEuler(self, inputs, alignments, axis):

        r = euler_matrix_batch(alignments[0] + inputs[3][:, 0],
                               alignments[1] + inputs[3][:, 1],
                               alignments[2] + inputs[3][:, 2])

        c_r_1 = tf.multiply(inputs[0], tf.cast(tf.gather(r[axis], 0, axis=1), dtype=self.precision))
        c_r_2 = tf.multiply(inputs[1], tf.cast(tf.gather(r[axis], 1, axis=1), dtype=self.precision))
        c_r_3 = tf.multiply(inputs[2], tf.cast(tf.gather(r[axis], 2, axis=1), dtype=self.precision))
        return tf.add(tf.add(c_r_1, c_r_2), c_r_3)

    def applyShifts(self, c, shifts_batch, axis):
        return tf.add(tf.subtract(c, shifts_batch[axis][None, :]), self.xmipp_origin[axis])

    def applyDeltaShifts(self, c, shifts_batch, axis):
        return tf.add(tf.subtract(c[0], shifts_batch[axis][None, :] + c[1][..., axis]),
                      self.xmipp_origin[axis])

    def batch_scatter_nd_add(self, ref, indices, updates):
        # Get batch size
        batch_size = tf.shape(ref)[0]

        # Create a range tensor for batch indices
        batch_indices = tf.range(batch_size)
        batch_indices = tf.reshape(batch_indices, [-1, 1, 1])  # Shape: [B, 1, 1]
        batch_indices = tf.tile(batch_indices, [1, tf.shape(indices)[1], 1])

        # Expand indices to include batch dimension
        expanded_indices = tf.concat([batch_indices, indices], axis=-1)  # Shape: [B, M, 3]

        # Flatten the first two dimensions of expanded_indices and updates
        flat_indices = tf.reshape(expanded_indices, [-1, 3])  # Shape: [B*M, 3]
        flat_updates = tf.reshape(updates, [-1])  # Shape: [B*M]

        # Perform scatter_nd_add on each item in the batch
        scattered = tf.tensor_scatter_nd_add(ref, flat_indices, flat_updates)

        return scattered

    def scatterImgByPass(self, c):
        # Get current batch size (function scope)
        batch_size_scope = tf.shape(c[0])[1]

        c_x = tf.reshape(tf.transpose(c[0]), [batch_size_scope, -1, 1])
        c_y = tf.reshape(tf.transpose(c[1]), [batch_size_scope, -1, 1])
        c_sampling = tf.concat([c_y, c_x], axis=2)

        imgs = tf.zeros((batch_size_scope, self.xsize, self.xsize), dtype=self.precision)

        bamp = tf.constant(self.values, dtype=self.precision)[None, ...] + c[2]

        bposf = tf.floor(c_sampling)
        bposi = tf.cast(bposf, tf.int32)
        bposf = c_sampling - bposf

        # Bilinear interpolation to provide forward mapping gradients
        bamp0 = bamp * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp1 = bamp * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp2 = bamp * (bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp3 = bamp * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])
        bampall = tf.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposall = tf.concat([bposi, bposi + (1, 0), bposi + (1, 1), bposi + (0, 1)], 1)
        # images = tf.stack([tf.tensor_scatter_nd_add(imgs[i], bposall[i], bampall[i]) for i in range(imgs.shape[0])])

        # bposf = tf.round(c_sampling)
        # bposall = tf.cast(bposf, tf.int32)
        #
        # num = tf.reduce_sum(((bposf - c_sampling) ** 2.), axis=-1)
        # sigma = 1.
        # bampall = bamp[None, :] * tf.exp(-num / (2. * sigma ** 2.))

        # fn = lambda inp: tf.tensor_scatter_nd_add(inp[0], inp[1], inp[2])
        # images = tf.map_fn(fn, [imgs, bposall, bampall], fn_output_signature=tf.float32)
        images = self.batch_scatter_nd_add(imgs, bposall, bampall)
        # images = tf.vectorized_map(fn, [imgs, bposall, bampall])

        images = tf.reshape(images, [-1, self.xsize, self.xsize, 1])

        return images

    def centerMassShift(self):
        coords_o = tf.constant(self.coords, dtype=self.precision)
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

    def averageDeformation(self, d_f):
        d_x, d_y, d_z = d_f[0], d_f[1], d_f[2]

        rmsdef = tf.reduce_mean(tf.sqrt(tf.reduce_mean(d_x * d_x + d_y * d_y + d_z * d_z, axis=0)))

        return rmsdef

    def calcBond(self, coords):
        coords = [tf.transpose(coords[0]), tf.transpose(coords[1]), tf.transpose(coords[2])]
        coords = tf.stack(coords, axis=2)

        bonds_coords = tf.gather(coords, self.bonds)
        dst = tf.sqrt(tf.nn.relu(tf.reduce_sum((bonds_coords[:, :, 1, :] - bonds_coords[:, :, 0, :]) ** 2, axis=2)))

        return dst

    # def calcAngle(self, coords):
    #     coords = [tf.transpose(coords[0]), tf.transpose(coords[1]), tf.transpose(coords[2])]
    #     coords = tf.stack(coords, axis=2)
    #     p0 = tf.gather(coords, self.connectivity[:, 0], axis=1)
    #     p1 = tf.gather(coords, self.connectivity[:, 1], axis=1)
    #     p2 = tf.gather(coords, self.connectivity[:, 2], axis=1)
    #     b0 = p0 - p1
    #     b1 = p2 - p1
    #     b0 = b0 / tf.sqrt(tf.reduce_sum(b0 ** 2.0, axis=2, keepdims=True))
    #     b1 = b1 / tf.reduce_sum(b1 ** 2.0, axis=2, keepdims=True)
    #     ang = tf.reduce_sum(b0 * b1, axis=2)
    #     n0 = tf.linalg.norm(b0, axis=2) * tf.linalg.norm(b1, axis=2)
    #     ang = tf.math.divide_no_nan(ang, n0)
    #     epsilon = 1.0 - 1e-6
    #     ang = tf.minimum(tf.maximum(ang, -epsilon), epsilon)
    #     # ang = np.min(np.max(ang, axis=1), axis=0)
    #     ang = tf.acos(ang)
    #     ang *= 180 / np.pi
    #
    #     return ang

    def calcAngle(self, coords):
        bsz = tf.shape(coords)[-1]
        coords = [tf.transpose(coords[0]), tf.transpose(coords[1]), tf.transpose(coords[2])]
        coords = tf.stack(coords, axis=2)

        dihedrals_coords = tf.gather(coords, self.dihedrals)
        dihedrals_coords = tf.reshape(dihedrals_coords, (bsz, -1, 4, 3, 3))

        ab = dihedrals_coords[:, 0] - dihedrals_coords[:, 1]
        cb = dihedrals_coords[:, 2] - dihedrals_coords[:, 1]
        db = dihedrals_coords[:, 3] - dihedrals_coords[:, 2]

        # Compute normals
        u = tf.linalg.cross(ab, cb)
        v = tf.linalg.cross(db, cb)
        w = tf.linalg.cross(u, v)

        # Normalize normals
        u = tf.nn.l2_normalize(u)
        v = tf.nn.l2_normalize(v)
        w = tf.nn.l2_normalize(w)
        cb = tf.nn.l2_normalize(cb)

        # Compute angle
        angle = tf.acos(tf.clip_by_value(tf.reduce_sum(u * v, axis=3), -1., 1.))
        angle_check = tf.acos(tf.clip_by_value(tf.reduce_sum(cb * w, axis=3), -1., 1.))

        # Adjust sign
        angle = tf.where(angle_check > 0.001, -angle, angle)

        # Convert to degrees
        angle *= 180.0 / np.pi

        return angle

    def search_radius(self, points, queries, radius):
        nsearch = ml3d.layers.FixedRadiusSearch(return_distances=True, ignore_query_point=True)
        ans = nsearch(points, queries, radius)
        return (tf.cast(ans.neighbors_index, tf.int32), tf.cast(ans.neighbors_row_splits, tf.int32),
                tf.cast(ans.neighbors_distance, self.precision))

    def search_knn(self, points, queries, k):
        nsearch = ml3d.layers.KNNSearch(return_distances=True, ignore_query_point=True)
        ans = nsearch(points, queries, k)
        return tf.cast(ans.neighbors_index, tf.int32), tf.cast(ans.neighbors_distance, self.precision)

    # def calcClashes(self, coords):
    #     coords = [tf.transpose(coords[0]), tf.transpose(coords[1]), tf.transpose(coords[2])]
    #     coords = tf.stack(coords, axis=2)
    #     coords = tf.transpose(coords, perm=(1, 0, 2))
    #
    #     B = tf.shape(coords)[0]
    #     extent = 8.  # Twice the radius (4A)
    #
    #     # To simulate gradient computation
    #     spread = tf.cast(tf.linspace(0, 1000, B), tf.float32)[..., None, None]
    #     coords = tf.reshape(coords + spread, (-1, 3))
    #
    #     # Compute neighbour distances
    #     energy = self.conv(tf.ones((tf.shape(coords)[0], 1), tf.float32), coords, coords, extent)
    #     energy = tf.reduce_mean(tf.reshape(energy, (B, -1)), axis=-1)
    #
    #     return energy

    def calcCoords(self, coords):
        B = tf.shape(coords[0])[1]

        # Clashes
        coords = [tf.transpose(coords[0]), tf.transpose(coords[1]), tf.transpose(coords[2])]
        coords = tf.stack(coords, axis=2)
        coords *= self.sr

        # Correct CA indices
        ca_indices = tf.tile(self.ca_indices[None, ..., None], (B, 1, 1))
        indices_B = tf.reshape(tf.range(B), [B, 1, 1])
        indices_B = tf.tile(indices_B, [1, tf.shape(ca_indices)[1], 1])
        ca_indices = tf.concat([indices_B, ca_indices], axis=2)

        # Extract CA (only for clashes)
        coords = tf.gather_nd(coords, ca_indices)

        # B = tf.shape(coords)[0]
        # N = tf.cast(tf.shape(coords)[1], tf.float32)
        # extent = 8.  # Twice the radius (4A)

        # To simulate gradient computation
        # spread = tf.cast(tf.linspace(0, 1000, B), tf.float32)[..., None, None]
        # coords = tf.reshape(coords + spread, (-1, 3))
        coords = tf.reshape(coords, (-1, 3))

        return coords

    def compute_histogram(self, tensor, nbins=20):
        """
        Compute histograms for each batch in the tensor.

        Args:
        - tensor: A 3D Tensor of shape (B, M, N) where B is the batch size and M, N are the dimensions of each batch.
        - value_range: A tuple (min, max) specifying the range of values to be covered by the histogram bins.
        - nbins: An integer specifying the number of bins in the histogram.

        Returns:
        - A 2D Tensor of shape (B, nbins) containing the histograms for each batch.
        """
        min_val, max_val = tf.reduce_min(tensor), tf.reduce_max(tensor)

        # Normalize the input tensor to the range [0, nbins)
        normalized_tensor = (tensor - min_val) / (max_val - min_val) * nbins
        normalized_tensor = tf.clip_by_value(normalized_tensor, 0, nbins - 1)
        indices = tf.cast(normalized_tensor, tf.int32)

        # One-hot encode the indices
        one_hot_encoded = tf.one_hot(indices, depth=nbins, axis=-1)

        # Sum the one-hot encoded tensor along the last two axes (M, N)
        histograms = tf.reduce_sum(one_hot_encoded, axis=[1, 2])

        return histograms / (tf.reduce_sum(histograms, axis=1)[..., None] + 1e-6)

    def wasserstein_distance(self, hist1, hist2):
        """Compute the Wasserstein distance between two histograms."""
        return tf.reduce_sum(tf.abs(tf.cumsum(hist1, axis=1) - tf.cumsum(hist2, axis=1)))

    def wasserstein_distance_loss(self, y_true, y_pred, bins=20):
        hist1 = self.compute_histogram(tf.squeeze(y_true), bins)
        hist2 = self.compute_histogram(tf.squeeze(y_pred), bins)
        return self.wasserstein_distance(hist1, hist2)

    # ----- -------- -----#

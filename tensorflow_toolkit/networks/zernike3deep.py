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

import tensorflow as tf
from tensorflow.keras import layers

try:
    import open3d.ml.tf as ml3d
    allow_open3d = True
except ImportError:
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    allow_open3d = False
    print(YELLOW + "Open3D has not been installed. The program will continue without this package" + RESET)

from tensorflow_toolkit.utils import computeCTF, euler_matrix_batch, full_fft_pad, full_ifft_pad, \
    apply_blur_filters_to_batch, create_blur_filters
from tensorflow_toolkit.layers.residue_conv2d import ResidueConv2D


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


def lennard_jones(r2, radius):
    # r2 = r * radius * radius
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    s6 = 0.1176
    s12 = 0.0138
    return (s12 / r12) - (s6 / r6)


def simple_clash(r2, splits, radius):
    lengths = tf.math.subtract(splits[1:], splits[:-1])
    expanded_lengths = tf.cast(tf.repeat(lengths, lengths), tf.float32)
    # r = tf.sqrt(r2)
    radius2 = radius * radius
    # r2 = r * radius * radius
    return tf.abs(r2 - radius2) / expanded_lengths


def tanh(x):
    return 10. * tf.math.tanh(x)


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, architecture="convnn", mode="spa", jit_compile=True):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        l2 = tf.keras.regularizers.l2(1e-3)

        # XLA compilation of methods
        if jit_compile:
            self.call = tf.function(jit_compile=jit_compile)(self.call)

        encoder_inputs = tf.keras.Input(shape=(input_dim, input_dim, 1))
        subtomo_pe = tf.keras.Input(shape=(100,))

        if architecture == "mlpnn":
            x = tf.keras.layers.Flatten()(encoder_inputs)
            for _ in range(12):
                x = layers.Dense(1024, activation='relu', kernel_regularizer=l2)(x)
            x = layers.Dropout(0.3)(x)
            x = layers.BatchNormalization()(x)

        elif architecture == "convnn":
            x = tf.keras.layers.Flatten()(encoder_inputs)
            for _ in range(3):
                x = layers.Dense(64 * 64, activation='relu', kernel_regularizer=l2)(x)

            x = tf.keras.layers.Dense(64 * 64, kernel_regularizer=l2)(x)
            x = tf.keras.layers.Reshape((64, 64, 1))(x)

            x = tf.keras.layers.Conv2D(64, 5, activation="relu", strides=(2, 2), padding="same")(x)
            for _ in range(1):
                x = ResidueConv2D(64, 5, activation="relu", padding="same")(x)
            x = tf.keras.layers.Conv2D(32, 5, activation="relu", strides=(2, 2), padding="same")(x)
            for _ in range(1):
                x = ResidueConv2D(32, 5, activation="relu", padding="same")(x)
            x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(x)
            x = ResidueConv2D(16, 3, activation="relu", padding="same")(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dropout(.1)(x)
            x = tf.keras.layers.BatchNormalization()(x)

            for _ in range(3):
                x = layers.Dense(3 * 16 * 16, activation='relu', kernel_regularizer=l2)(x)
            x = layers.Dropout(.1)(x)
            x = layers.BatchNormalization()(x)

        if mode == "tomo":
            latent = layers.Dense(1024, activation="relu")(subtomo_pe)
            for _ in range(2):  # TODO: Is it better to use 12 hidden layers as in Zernike3Deep?
                latent = layers.Dense(1024, activation="relu")(latent)

        if mode == "spa":
            self.encoder = tf.keras.Model(encoder_inputs, x, name="encoder")
        elif mode == "tomo":
            self.encoder = tf.keras.Model([encoder_inputs, subtomo_pe], [x, latent], name="encoder")

    # @tf.function(jit_compile=True)
    def call(self, x):
        return self.encoder(x)


class Decoder:
    # @tf.function(jit_compile=True)
    def __init__(self, generator, CTF="apply", jit_compile=True):
        super(Decoder, self).__init__()
        self.generator = generator
        self.CTF = CTF

        # XLA compilation of methods
        if jit_compile:
            self.prepare_batch = tf.function(jit_compile=jit_compile)(self.prepare_batch)
            self.compute_field_volume = tf.function(jit_compile=jit_compile)(self.compute_field_volume)
            self.compute_field_atoms = tf.function(jit_compile=jit_compile)(self.compute_field_atoms)
            self.compute_atom_cost_params = tf.function(jit_compile=jit_compile)(self.compute_atom_cost_params)
            self.apply_alignment_and_shifts = tf.function(jit_compile=jit_compile)(self.apply_alignment_and_shifts)
            self.compute_theo_proj = tf.function(jit_compile=jit_compile)(self.compute_theo_proj)
            self.__call__ = tf.function(jit_compile=jit_compile)(self.__call__)

    # @tf.function(jit_compile=True)
    def prepare_batch(self, indexes, permute_view=False):
        # images, indexes = x

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(indexes)[0]

        # Precompute batch alignments
        rot_batch = tf.gather(self.generator.angle_rot, indexes, axis=0)
        tilt_batch = tf.gather(self.generator.angle_tilt, indexes, axis=0)
        psi_batch = tf.gather(self.generator.angle_psi, indexes, axis=0)

        shifts_x = tf.gather(self.generator.shifts[0], indexes, axis=0)
        shifts_y = tf.gather(self.generator.shifts[1], indexes, axis=0)

        if permute_view:
            # Random permutations of angles and shifts
            euler_batch = tf.stack([rot_batch, tilt_batch, psi_batch], axis=1)
            shifts_batch = tf.stack([shifts_x, shifts_y], axis=1)
            euler_batch_perm = tf.random.shuffle(euler_batch)
            shifts_batch_perm = tf.random.shuffle(shifts_batch)
            rot_batch, tilt_batch, psi_batch = euler_batch_perm[:, 0], euler_batch_perm[:, 1], euler_batch_perm[:, 2]
            shifts_x, shifts_y = shifts_batch_perm[:, 0], shifts_batch_perm[:, 1]

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

        return [rot_batch, tilt_batch, psi_batch], [shifts_x, shifts_y], ctf

    # @tf.function(jit_compile=True)
    def compute_field_volume(self, decoder_inputs_x, decoder_inputs_y, decoder_inputs_z):
        # Compute deformation field
        d_x = self.generator.computeDeformationFieldVol(decoder_inputs_x)
        d_y = self.generator.computeDeformationFieldVol(decoder_inputs_y)
        d_z = self.generator.computeDeformationFieldVol(decoder_inputs_z)

        # Apply deformation field
        c_x = self.generator.applyDeformationFieldVol(d_x, 0)
        c_y = self.generator.applyDeformationFieldVol(d_y, 1)
        c_z = self.generator.applyDeformationFieldVol(d_z, 2)

        return c_x, c_y, c_z

    # @tf.function(jit_compile=True)
    def compute_field_atoms(self, decoder_inputs_x, decoder_inputs_y, decoder_inputs_z):
        # Compute atoms deformation field
        da_x = self.generator.computeDeformationFieldAtoms(decoder_inputs_x)
        da_y = self.generator.computeDeformationFieldAtoms(decoder_inputs_y)
        da_z = self.generator.computeDeformationFieldAtoms(decoder_inputs_z)

        # Apply deformation field
        a_x = self.generator.applyDeformationFieldAtoms(da_x, 0)
        a_y = self.generator.applyDeformationFieldAtoms(da_y, 1)
        a_z = self.generator.applyDeformationFieldAtoms(da_z, 2)

        return a_x, a_y, a_z

    # @tf.function(jit_compile=True)
    def compute_atom_cost_params(self, a_x, a_y, a_z):
        bondk = self.generator.calcBond([a_x, a_y, a_z])
        anglek = self.generator.calcAngle([a_x, a_y, a_z])
        coords = self.generator.calcCoords([a_x, a_y, a_z])

        return bondk, anglek, coords

    # @tf.function(jit_compile=True)
    def apply_alignment_and_shifts(self, c_x, c_y, c_z, alignments, shifts, delta_euler, delta_shifts):
        # Apply alignment
        c_r_x = self.generator.applyAlignmentDeltaEuler([c_x, c_y, c_z, delta_euler], alignments, 0)
        c_r_y = self.generator.applyAlignmentDeltaEuler([c_x, c_y, c_z, delta_euler], alignments, 1)

        # Apply shifts
        c_r_s_x = self.generator.applyDeltaShifts([c_r_x, delta_shifts], shifts, 0)
        c_r_s_y = self.generator.applyDeltaShifts([c_r_y, delta_shifts], shifts, 1)

        return c_r_s_x, c_r_s_y

    # @tf.function(jit_compile=True)
    def compute_theo_proj(self, c_x, c_y, ctf):
        # Scatter image and bypass gradient
        decoded = self.generator.scatterImgByPass([c_x, c_y])

        if self.generator.step > 1:
            # Gaussian filter image
            decoded = self.generator.gaussianFilterImage(decoded)

            if self.CTF == "apply":
                # CTF filter image
                decoded_ctf = self.generator.ctfFilterImage(decoded, ctf)
            else:
                decoded_ctf = decoded
        else:
            if self.CTF == "apply":
                # CTF filter image
                decoded_ctf = self.generator.ctfFilterImage(decoded, ctf)
            else:
                decoded_ctf = decoded

        return decoded, decoded_ctf

    # @tf.function(jit_compile=False)
    def __call__(self, x, permute_view=False):
        # encoded, images, indexes = x
        encoded, indexes = x
        alignments, shifts, ctf = self.prepare_batch(indexes, permute_view)

        decoder_inputs_x, decoder_inputs_y, decoder_inputs_z, delta_euler, delta_shifts = encoded

        # Compute deformation field
        c_x, c_y, c_z = self.compute_field_volume(decoder_inputs_x, decoder_inputs_y, decoder_inputs_z)

        # Bond and angle
        if self.generator.ref_is_struct:
            # Compute atoms deformation field
            a_x, a_y, a_z = self.compute_field_atoms(decoder_inputs_x, decoder_inputs_y, decoder_inputs_z)

            bondk, anglek, coords = self.compute_atom_cost_params(a_x, a_y, a_z)

        else:
            bondk = tf.constant(0.0, tf.float32)
            anglek = tf.constant(0.0, tf.float32)
            coords = tf.constant(0.0, tf.float32)

        # Apply alignment and shifts
        c_r_s_x, c_r_s_y = self.apply_alignment_and_shifts(c_x, c_y, c_z, alignments, shifts, delta_euler, delta_shifts)

        # Theoretical projections
        decoded, decoded_ctf = self.compute_theo_proj(c_r_s_x, c_r_s_y, ctf)

        return [decoded, decoded_ctf], bondk, anglek, coords, ctf


class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, architecture="convnn", CTF="apply", mode=None, l_bond=0.01, l_angle=0.01,
                 l_clashes=None, l_norm=1e-4, jit_compile=True, poseReg=0.0, ctfReg=0.0, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.refPose = 1.0 if generator.refinePose else 0.0
        self.mode = generator.mode if mode is None else mode
        self.l_bond = l_bond
        self.l_angle = l_angle
        self.l_clashes = l_clashes if l_clashes is not None else 0.0
        self.l_norm = l_norm
        self.pose_lambda = poseReg
        self.ctf_lambda = ctfReg
        self.disantangle_pose = poseReg > 0.0
        self.disantangle_ctf = ctfReg > 0.0
        self.filters = create_blur_filters(5, 10, 30)
        # self.architecture = architecture
        self.encoder_exp = Encoder(generator.zernike_size.shape[0], generator.xsize,architecture=architecture,
                                   mode=self.mode, jit_compile=jit_compile)
        if poseReg > 0.0:
            self.encoder_clean = Encoder(generator.zernike_size.shape[0], generator.xsize, architecture=architecture,
                                         mode=self.mode, jit_compile=jit_compile)
        if ctfReg > 0.0:
            self.encoder_ctf = Encoder(generator.zernike_size.shape[0], generator.xsize, architecture=architecture,
                                       mode=self.mode, jit_compile=jit_compile)
        self.decoder = Decoder(generator, CTF=self.CTF, jit_compile=jit_compile)
        self.z_space_x = layers.Dense(generator.zernike_size.shape[0], activation="linear", name="z_space_x")
        self.z_space_y = layers.Dense(generator.zernike_size.shape[0], activation="linear", name="z_space_y")
        self.z_space_z = layers.Dense(generator.zernike_size.shape[0], activation="linear", name="z_space_z")
        self.delta_euler = layers.Dense(3, activation="linear", name="delta_euler", trainable=generator.refinePose)
        self.delta_shifts = layers.Dense(2, activation="linear", name="delta_shifts", trainable=generator.refinePose)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.img_loss_tracker = tf.keras.metrics.Mean(name="img_loss")
        self.bond_loss_tracker = tf.keras.metrics.Mean(name="bond_loss")
        self.angle_loss_tracker = tf.keras.metrics.Mean(name="angle_loss")
        self.clash_loss_tracker = tf.keras.metrics.Mean(name="clash_loss")
        self.norm_loss_tracker = tf.keras.metrics.Mean(name="norm_loss")
        self.loss_disantangle_tracker = tf.keras.metrics.Mean(name="loss_disentangled")

        # XLA compilation of cost function
        if jit_compile:
            self.cost_function = tf.function(jit_compile=jit_compile)(self.generator.cost_function)
        else:
            self.cost_function = self.generator.cost_function

        if allow_open3d and self.generator.ref_is_struct:
            # Continuous convolution
            # k_clash = 0.6  # Repulsion value (for bb)
            # extent = 1.2  # 2 * radius, typical class distance between 0.4A-0.6A (for bb)
            self.k_clash = 4.  # Repulsion value
            self.extent = 8.  # 2 * radius, typical class distance between 0.4A-0.6A
            self.fn = lambda x, y: simple_clash(x, y, self.k_clash)
            self.conv = ml3d.layers.ContinuousConv(1, kernel_size=[3, 3, 3],
                                                   activation=None, use_bias=False,
                                                   trainable=False, kernel_initializer=tf.keras.initializers.Ones(),
                                                   kernel_regularizer=None,
                                                   normalize=False,
                                                   radius_search_metric="L2",
                                                   coordinate_mapping="identity",
                                                   interpolation="nearest_neighbor",
                                                   window_function=None, radius_search_ignore_query_points=True)
            self.nsearch = ml3d.layers.FixedRadiusSearch(return_distances=True, ignore_query_point=True)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.img_loss_tracker,
            self.bond_loss_tracker,
            self.angle_loss_tracker,
            self.clash_loss_tracker,
            self.norm_loss_tracker,
            self.loss_disantangle_tracker,
        ]

    def train_step(self, data):
        inputs = data[0]

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

        # Precompute batch zernike coefficients
        z_x_batch = tf.gather(self.generator.z_x_space, indexes, axis=0)
        z_y_batch = tf.gather(self.generator.z_y_space, indexes, axis=0)
        z_z_batch = tf.gather(self.generator.z_z_space, indexes, axis=0)

        if allow_open3d and self.generator.ref_is_struct:
            # Row splits
            B = tf.shape(images)[0]
            # num_points = self.generator.atom_coords.shape[0]
            num_points = tf.cast(tf.shape(self.generator.ca_indices)[0], tf.int64)
            points_row_splits = tf.range(B + 1, dtype=tf.int64) * num_points
            queries_row_splits = tf.range(B + 1, dtype=tf.int64) * num_points

        if self.CTF == "wiener":
            # Precompute batch CTFs
            batch_size_scope = tf.shape(indexes)[0]
            defocusU_batch = tf.gather(self.generator.defocusU, indexes, axis=0)
            defocusV_batch = tf.gather(self.generator.defocusV, indexes, axis=0)
            defocusAngle_batch = tf.gather(self.generator.defocusAngle, indexes, axis=0)
            cs_batch = tf.gather(self.generator.cs, indexes, axis=0)
            kv_batch = self.generator.kv
            ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                             self.generator.sr, self.generator.pad_factor,
                             [self.generator.xsize, int(0.5 * self.generator.xsize + 1)],
                             batch_size_scope, self.generator.applyCTF)
            images = self.generator.wiener2DFilter(images, ctf)
            if self.mode == "spa":
                inputs = images
            elif self.mode == "tomo":
                inputs[0] = images

        with tf.GradientTape() as tape:
            # Forward pass (first encoder and decoder)
            if self.mode == "spa":
                x = self.encoder_exp(inputs)
                encoded = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x), self.delta_euler(x),
                           self.delta_shifts(x)]
            elif self.mode == "tomo":
                x, latent = self.encoder_exp(inputs)
                encoded = [self.z_space_x(latent), self.z_space_y(latent), self.z_space_z(latent), self.delta_euler(x),
                           self.delta_shifts(x)]
            encoded[0] = encoded[0] + z_x_batch
            encoded[1] = encoded[1] + z_y_batch
            encoded[2] = encoded[2] + z_z_batch
            het = tf.concat([encoded[0], encoded[1], encoded[2]], axis=1)
            encoded[3] *= self.refPose
            encoded[4] *= self.refPose
            decoded_vec, bondk, anglek, coords, ctf = self.decoder([encoded, indexes], permute_view=False)
            decoded, decoded_ctf = decoded_vec[0], decoded_vec[1]

            if self.disantangle_pose and self.mode == "spa":
                # Forward pass (second decoder - no permutation)
                x = self.encoder_clean(decoded)
                encoded_clean = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x)]
                encoded_clean[0] = encoded_clean[0] + z_x_batch
                encoded_clean[1] = encoded_clean[1] + z_y_batch
                encoded_clean[2] = encoded_clean[2] + z_z_batch
                het_clean = tf.concat([encoded_clean[0], encoded_clean[1], encoded_clean[2]], axis=1)

                # Forward pass (third encoder - permuted CTF)
                if self.disantangle_ctf and self.CTF is not None:
                    x = self.encoder_ctf(decoded_ctf)
                    encoded_ctf = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x)]
                    encoded_ctf[0] = encoded_ctf[0] + z_x_batch
                    encoded_ctf[1] = encoded_ctf[1] + z_y_batch
                    encoded_ctf[2] = encoded_ctf[2] + z_z_batch
                    het_ctf = tf.concat([encoded_ctf[0], encoded_ctf[1], encoded_ctf[2]], axis=1)
                    ctf_perm = tf.random.shuffle(ctf)
                    decoded_het_ctf_perm = self.decoder.generator.ctfFilterImage(decoded, ctf_perm)
                    x = self.encoder_ctf(decoded_het_ctf_perm)
                    encoded_ctf_perm = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x)]
                    encoded_ctf_perm[0] = encoded_ctf_perm[0] + z_x_batch
                    encoded_ctf_perm[1] = encoded_ctf_perm[1] + z_y_batch
                    encoded_ctf_perm[2] = encoded_ctf_perm[2] + z_z_batch
                    het_ctf_perm = tf.concat([encoded_ctf_perm[0], encoded_ctf_perm[1], encoded_ctf_perm[2]], axis=1)
                else:
                    het_ctf_perm = het

                # Forward pass (second decoder - permutation)
                decoded_vec, _, _, _, _ = self.decoder([encoded, indexes], permute_view=True)
                x = self.encoder_clean(decoded_vec[0])
                encoded_clean_perm = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x)]
                encoded_clean_perm[0] = encoded_clean_perm[0] + z_x_batch
                encoded_clean_perm[1] = encoded_clean_perm[1] + z_y_batch
                encoded_clean_perm[2] = encoded_clean_perm[2] + z_z_batch
                het_clean_perm = tf.concat([encoded_clean_perm[0], encoded_clean_perm[1], encoded_clean_perm[2]], axis=1)

            if allow_open3d and self.generator.ref_is_struct and self.l_clashes > 0.0:
                # Fixed radius search
                result = self.nsearch(coords, coords, 0.5 * self.extent, points_row_splits, queries_row_splits)

                # Compute neighbour distances
                clashes = self.conv(tf.ones((tf.shape(coords)[0], 1), tf.float32), coords, coords, self.extent,
                                    user_neighbors_row_splits=result.neighbors_row_splits,
                                    user_neighbors_index=result.neighbors_index,
                                    user_neighbors_importance=self.fn(result.neighbors_distance,
                                                                      result.neighbors_row_splits))
                clashes = tf.reduce_max(tf.reshape(clashes, (B, -1)), axis=-1)
            else:
                clashes = tf.constant(0.0, tf.float32)

            img_loss = self.cost_function(images, decoded_ctf)

            # Bond and angle losses
            if self.generator.ref_is_struct:
                bond_loss = tf.sqrt(tf.reduce_max(tf.keras.losses.MSE(self.generator.bond0, bondk)))
                angle_loss = tf.sqrt(tf.reduce_max(tf.keras.losses.MSE(self.generator.angle0, anglek)))
            else:
                bond_loss, angle_loss = tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)

            # Loss disantagled (pose)
            if self.disantangle_pose and self.mode == "spa":
                loss_disantagled_pose = (tf.keras.losses.MSE(het, het_clean)
                                         + tf.keras.losses.MSE(het, het_clean_perm))
            else:
                loss_disantagled_pose = 0.0

            # Loss disantagled (CTF)
            if self.disantangle_ctf and self.mode == "spa":
                loss_disantagled_ctf = (tf.keras.losses.MSE(het, het_ctf)
                                        + tf.keras.losses.MSE(het, het_ctf_perm))
            else:
                loss_disantagled_ctf = 0.0

            # Coefficent norm loss
            norm_x = tf.reduce_mean(tf.square(encoded[0]), axis=-1)
            norm_y = tf.reduce_mean(tf.square(encoded[1]), axis=-1)
            norm_z = tf.reduce_mean(tf.square(encoded[2]), axis=-1)
            norm_loss = (norm_x + norm_y + norm_z) / 3.

            total_loss = (img_loss + self.l_bond * bond_loss
                          + self.l_angle * angle_loss + self.l_clashes * clashes +
                          self.l_norm * norm_loss + self.pose_lambda * loss_disantagled_pose +
                          self.ctf_lambda * loss_disantagled_ctf)  # 0.001 works on HetSIREN

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.img_loss_tracker.update_state(img_loss)
        self.angle_loss_tracker.update_state(angle_loss)
        self.bond_loss_tracker.update_state(bond_loss)
        self.clash_loss_tracker.update_state(clashes)
        self.norm_loss_tracker.update_state(norm_loss)
        self.loss_disantangle_tracker.update_state(loss_disantagled_pose + loss_disantagled_ctf)
        return {
            "loss": self.total_loss_tracker.result(),
            "img_loss": self.img_loss_tracker.result(),
            "bond": self.bond_loss_tracker.result(),
            "angle": self.angle_loss_tracker.result(),
            "clashes": self.clash_loss_tracker.result(),
            "norm": self.norm_loss_tracker.result(),
            "loss_disentangled": self.loss_disantangle_tracker.result(),
        }

    def test_step(self, data):
        inputs = data[0]

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

        # Precompute batch zernike coefficients
        z_x_batch = tf.gather(self.generator.z_x_space, indexes, axis=0)
        z_y_batch = tf.gather(self.generator.z_y_space, indexes, axis=0)
        z_z_batch = tf.gather(self.generator.z_z_space, indexes, axis=0)

        # Row splits
        if allow_open3d and self.generator.ref_is_struct:
            B = tf.shape(images)[0]
            # num_points = self.generator.atom_coords.shape[0]
            num_points = tf.cast(tf.shape(self.generator.ca_indices)[0], tf.int64)
            points_row_splits = tf.range(B + 1, dtype=tf.int64) * num_points
            queries_row_splits = tf.range(B + 1, dtype=tf.int64) * num_points

        if self.CTF == "wiener":
            # Precompute batch CTFs
            batch_size_scope = tf.shape(indexes)[0]
            defocusU_batch = tf.gather(self.generator.defocusU, indexes, axis=0)
            defocusV_batch = tf.gather(self.generator.defocusV, indexes, axis=0)
            defocusAngle_batch = tf.gather(self.generator.defocusAngle, indexes, axis=0)
            cs_batch = tf.gather(self.generator.cs, indexes, axis=0)
            kv_batch = self.generator.kv
            ctf = computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                             self.generator.sr, self.generator.pad_factor,
                             [self.generator.xsize, int(0.5 * self.generator.xsize + 1)],
                             batch_size_scope, self.generator.applyCTF)
            images = self.generator.wiener2DFilter(images, ctf)
            if self.mode == "spa":
                inputs = images
            elif self.mode == "tomo":
                inputs[0] = images

        # Forward pass (first encoder and decoder)
        if self.mode == "spa":
            x = self.encoder_exp(inputs)
            encoded = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x), self.delta_euler(x),
                       self.delta_shifts(x)]
        elif self.mode == "tomo":
            x, latent = self.encoder_exp(inputs)
            encoded = [self.z_space_x(latent), self.z_space_y(latent), self.z_space_z(latent),
                       self.delta_euler(x),
                       self.delta_shifts(x)]
        encoded[0] = encoded[0] + z_x_batch
        encoded[1] = encoded[1] + z_y_batch
        encoded[2] = encoded[2] + z_z_batch
        het = tf.concat([encoded[0], encoded[1], encoded[2]], axis=1)
        encoded[3] *= self.refPose
        encoded[4] *= self.refPose
        decoded_vec, bondk, anglek, coords, ctf = self.decoder([encoded, indexes], permute_view=False)
        decoded, decoded_ctf = decoded_vec[0], decoded_vec[1]

        if self.disantangle_pose and self.mode == "spa":
            # Forward pass (second decoder - no permutation)
            x = self.encoder_clean(decoded)
            encoded_clean = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x)]
            encoded_clean[0] = encoded_clean[0] + z_x_batch
            encoded_clean[1] = encoded_clean[1] + z_y_batch
            encoded_clean[2] = encoded_clean[2] + z_z_batch
            het_clean = tf.concat([encoded_clean[0], encoded_clean[1], encoded_clean[2]], axis=1)

            # Forward pass (third encoder - permuted CTF)
            if self.disantangle_ctf and self.CTF is not None:
                x = self.encoder_ctf(decoded_ctf)
                encoded_ctf = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x)]
                encoded_ctf[0] = encoded_ctf[0] + z_x_batch
                encoded_ctf[1] = encoded_ctf[1] + z_y_batch
                encoded_ctf[2] = encoded_ctf[2] + z_z_batch
                het_ctf = tf.concat([encoded_ctf[0], encoded_ctf[1], encoded_ctf[2]], axis=1)
                ctf_perm = tf.random.shuffle(ctf)
                decoded_het_ctf_perm = self.decoder.generator.ctfFilterImage(decoded, ctf_perm)
                x = self.encoder_ctf(decoded_het_ctf_perm)
                encoded_ctf_perm = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x)]
                encoded_ctf_perm[0] = encoded_ctf_perm[0] + z_x_batch
                encoded_ctf_perm[1] = encoded_ctf_perm[1] + z_y_batch
                encoded_ctf_perm[2] = encoded_ctf_perm[2] + z_z_batch
                het_ctf_perm = tf.concat([encoded_ctf_perm[0], encoded_ctf_perm[1], encoded_ctf_perm[2]],
                                         axis=1)
            else:
                het_ctf_perm = het

            # Forward pass (second decoder - permutation)
            decoded_vec, _, _, _, _ = self.decoder([encoded, indexes], permute_view=True)
            x = self.encoder_clean(decoded_vec[0])
            encoded_clean_perm = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x)]
            encoded_clean_perm[0] = encoded_clean_perm[0] + z_x_batch
            encoded_clean_perm[1] = encoded_clean_perm[1] + z_y_batch
            encoded_clean_perm[2] = encoded_clean_perm[2] + z_z_batch
            het_clean_perm = tf.concat([encoded_clean_perm[0], encoded_clean_perm[1], encoded_clean_perm[2]],
                                       axis=1)

        if allow_open3d and self.generator.ref_is_struct and self.l_clashes > 0.0:
            # Fixed radius search
            result = self.nsearch(coords, coords, 0.5 * self.extent, points_row_splits, queries_row_splits)

            # Compute neighbour distances
            clashes = self.conv(tf.ones((tf.shape(coords)[0], 1), tf.float32), coords, coords, self.extent,
                                user_neighbors_row_splits=result.neighbors_row_splits,
                                user_neighbors_index=result.neighbors_index,
                                user_neighbors_importance=self.fn(result.neighbors_distance,
                                                                  result.neighbors_row_splits))
            clashes = tf.reduce_max(tf.reshape(clashes, (B, -1)), axis=-1)
        else:
            clashes = tf.constant(0.0, tf.float32)

        img_loss = self.cost_function(images, decoded_ctf)

        # Bond and angle losses
        if self.generator.ref_is_struct:
            bond_loss = tf.sqrt(tf.reduce_max(tf.keras.losses.MSE(self.generator.bond0, bondk)))
            angle_loss = tf.sqrt(tf.reduce_max(tf.keras.losses.MSE(self.generator.angle0, anglek)))
        else:
            bond_loss, angle_loss = tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)

        # Loss disantagled (pose)
        if self.disantangle_pose and self.mode == "spa":
            loss_disantagled_pose = (tf.keras.losses.MSE(het, het_clean)
                                     + tf.keras.losses.MSE(het, het_clean_perm))
        else:
            loss_disantagled_pose = 0.0

        # Loss disantagled (CTF)
        if self.disantangle_ctf and self.mode == "spa":
            loss_disantagled_ctf = (tf.keras.losses.MSE(het, het_ctf)
                                    + tf.keras.losses.MSE(het, het_ctf_perm))
        else:
            loss_disantagled_ctf = 0.0

        # Coefficent norm loss
        norm_x = tf.reduce_mean(tf.square(encoded[0]), axis=-1)
        norm_y = tf.reduce_mean(tf.square(encoded[1]), axis=-1)
        norm_z = tf.reduce_mean(tf.square(encoded[2]), axis=-1)
        norm_loss = (norm_x + norm_y + norm_z) / 3.

        total_loss = (img_loss + self.l_bond * bond_loss
                      + self.l_angle * angle_loss + self.l_clashes * clashes +
                      self.l_norm * norm_loss + self.pose_lambda * loss_disantagled_pose +
                      self.ctf_lambda * loss_disantagled_ctf)  # 0.001 works on HetSIREN

        self.total_loss_tracker.update_state(total_loss)
        self.img_loss_tracker.update_state(img_loss)
        self.angle_loss_tracker.update_state(angle_loss)
        self.bond_loss_tracker.update_state(bond_loss)
        self.clash_loss_tracker.update_state(clashes)
        self.loss_disantangle_tracker.update_state(loss_disantagled_pose + loss_disantagled_ctf)
        return {
            "loss": self.total_loss_tracker.result(),
            "img_loss": self.img_loss_tracker.result(),
            "bond": self.bond_loss_tracker.result(),
            "angle": self.angle_loss_tracker.result(),
            "clashes": self.clash_loss_tracker.result(),
            "loss_disentangled": self.loss_disantangle_tracker.result(),
        }

    def predict_step(self, data):
        inputs = data[0]

        if self.mode == "spa":
            indexes = data[1]
            # images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            # images = inputs[0]

        # Precompute batch zernike coefficients
        self.decoder.generator.z_x_batch = tf.gather(self.generator.z_x_space, indexes, axis=0)
        self.decoder.generator.z_y_batch = tf.gather(self.generator.z_y_space, indexes, axis=0)
        self.decoder.generator.z_z_batch = tf.gather(self.generator.z_z_space, indexes, axis=0)

        if self.mode == "spa":
            x = self.encoder_exp(inputs)
            encoded = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x), self.delta_euler(x),
                       self.delta_shifts(x)]
        elif self.mode == "tomo":
            x, latent = self.encoder(inputs)
            encoded = [self.z_space_x(latent), self.z_space_y(latent), self.z_space_z(latent),
                       self.delta_euler(x),
                       self.delta_shifts(x)]
        encoded[0] = encoded[0] + self.decoder.generator.z_x_batch
        encoded[1] = encoded[1] + self.decoder.generator.z_y_batch
        encoded[2] = encoded[2] + self.decoder.generator.z_z_batch
        encoded[3] *= self.refPose
        encoded[4] *= self.refPose

        return encoded

    def call(self, input_features):
        if allow_open3d and self.generator.ref_is_struct:
            # To know this weights exist
            coords = tf.zeros((1, 3), tf.float32)
            _ = self.nsearch(coords, coords, 1.0)
            _ = self.conv(tf.ones((tf.shape(coords)[0], 1), tf.float32), coords, coords, self.extent)

        if self.mode == "spa":
            indexes = tf.zeros(tf.shape(input_features)[0], dtype=tf.int32)
            x = self.encoder_exp(input_features)
            encoded = [self.z_space_x(x), self.z_space_y(x), self.z_space_z(x), self.delta_euler(x),
                       self.delta_shifts(x)]
        elif self.mode == "tomo":
            indexes = tf.zeros(tf.shape(input_features[0])[0], dtype=tf.int32)
            x, latent = self.encoder_exp(input_features)
            encoded = [self.z_space_x(latent), self.z_space_y(latent), self.z_space_z(latent),
                       self.delta_euler(x), self.delta_shifts(x)]

        return self.decoder([encoded, indexes])

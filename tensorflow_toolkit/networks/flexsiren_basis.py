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
from scipy.ndimage import gaussian_filter

import tensorflow as tf
from keras.initializers import RandomUniform
from tensorflow.keras import layers

try:
    import open3d.ml.tf as ml3d

    allow_open3d = True
except ImportError:
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    allow_open3d = False
    print(YELLOW + "Open3D has not been installed. The program will continue without this package" + RESET)

from tensorflow_toolkit.utils import computeCTF, full_fft_pad, full_ifft_pad, create_blur_filters
from tensorflow_toolkit.layers.siren import SIRENFirstLayerInitializer, SIRENInitializer, Sine, MetaDenseWrapper


def resizeImageFourier(images, out_size, pad_factor=1, precision=tf.float32):
    # Sizes
    xsize = tf.shape(images)[1]
    pad_size = pad_factor * xsize
    pad_out_size = pad_factor * out_size

    # Fourier transform
    ft_images = full_fft_pad(images, pad_size, pad_size)

    # Normalization constant
    norm = tf.cast(pad_out_size, dtype=precision) / tf.cast(pad_size, dtype=precision)

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


def simple_clash(r2, splits, radius, precision):
    lengths = tf.math.subtract(splits[1:], splits[:-1])
    expanded_lengths = tf.cast(tf.repeat(lengths, lengths), precision)
    # r = tf.sqrt(r2)
    radius2 = radius * radius
    # r2 = r * radius * radius
    return tf.abs(r2 - radius2) / expanded_lengths


def tanh(x):
    return 10. * tf.math.tanh(x)


def compute_determinant_2x2(matrices):
    a, b, c, d = matrices[..., 0, 0], matrices[..., 0, 1], matrices[..., 1, 0], matrices[..., 1, 1]
    return a * d - b * c


def compute_determinant_3x3(matrices):
    a, b, c = matrices[..., 0, 0], matrices[..., 0, 1], matrices[..., 0, 2]
    d, e, f = matrices[..., 1, 0], matrices[..., 1, 1], matrices[..., 1, 2]
    g, h, i = matrices[..., 2, 0], matrices[..., 2, 1], matrices[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, architecture="convnn", mode="spa", refPose=True, jit_compile=True):
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
            for _ in range(3):
                x = layers.Dense(1024, activation='relu')(x)
            # x = layers.Dropout(0.3)(x)
            # x = layers.BatchNormalization()(x)

        elif architecture == "convnn":
            x = tf.keras.layers.Flatten()(encoder_inputs)
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

        latent = layers.Dense(256, activation="relu")(x)
        for _ in range(2):
            latent = layers.Dense(256, activation="relu")(latent)
            # x = layers.Dropout(0.3)(x)
            # x = layers.BatchNormalization()(x)
        if mode == "tomo":
            latent_label = layers.Dense(1024, activation="relu")(subtomo_pe)
            for _ in range(2):  # TODO: Is it better to use 12 hidden layers as in Zernike3Deep?
                latent_label = layers.Dense(1024, activation="relu")(latent_label)
            for _ in range(2):
                latent_label = layers.Dense(256, activation="relu")(latent_label)

        rows = layers.Dense(256, activation="relu", trainable=refPose)(x)
        for _ in range(2):
            rows = layers.Dense(256, activation="relu", trainable=refPose)(rows)

        shifts = layers.Dense(256, activation="relu", trainable=refPose)(x)
        for _ in range(2):
            shifts = layers.Dense(256, activation="relu", trainable=refPose)(shifts)

        if mode == "tomo":
            latent = layers.Dense(1024, activation="relu")(subtomo_pe)
            for _ in range(2):  # TODO: Is it better to use 12 hidden layers as in Zernike3Deep?
                latent = layers.Dense(1024, activation="relu")(latent)

        if mode == "spa":
            self.encoder = tf.keras.Model(encoder_inputs, [latent, shifts, rows], name="encoder")
        elif mode == "tomo":
            self.encoder = tf.keras.Model([encoder_inputs, subtomo_pe], [x, latent], name="encoder")

    # @tf.function(jit_compile=True)
    def call(self, x):
        return self.encoder(x)


class BasisDecoder(tf.keras.Model):
    def __init__(self, generator, latDim=64, jit_compile=True, precision=tf.float32, compute_delta=True):
        super(BasisDecoder, self).__init__()
        self.generator = generator
        self.compute_delta = compute_delta
        self.latDim = latDim
        first_siren = Sine(30.0)  # TODO: Try 30 again
        siren = Sine(1.0)
        # init_first = RandomUniform(-0.01, 0.01)
        # init_next = RandomUniform(-0.01, 0.01)
        # init_first = SIRENFirstLayerInitializer(scale=1.0)
        # init_next = SIRENInitializer(c=1.0, w0=1.0)

        # XLA compilation of methods
        if jit_compile:
            self.call = tf.function(jit_compile=jit_compile)(self.call)

        coords_het = tf.keras.Input(shape=(generator.scaled_coords.shape[-1], ))
        # coords_het = tf.keras.Input(shape=(None, latDim + generator.scaled_coords.shape[-1],))

        layers_comb = layers.Dense(32, activation=first_siren,
                                   kernel_initializer=SIRENFirstLayerInitializer(scale=1.0))(coords_het)  # 64 or 256 units
        # layers_comb = layers.Dense(32, activation="relu")(coords_het)  # 64 or 256 units
        # layers_comb = MetaDenseWrapper(latDim + generator.scaled_coords.shape[-1], 8, 8, w0=30.0,
        #                                meta_kernel_initializer=SIRENFirstLayerInitializer(scale=6.0),
        #                                num_hyper_layers=0)(coords_het)
        for _ in range(3):  # 12 seems OK
            aux_comb = layers.Dense(32, activation=siren,
                                    kernel_initializer=SIRENInitializer(c=1.0))(layers_comb)  # 64 or 256 units
            # aux_comb = layers.Dense(32, activation="relu")(layers_comb)  # 64 or 256 units
            # aux_comb = MetaDenseWrapper(8, 8, 8, w0=1.0,
            #                             meta_kernel_initializer=SIRENInitializer(),
            #                             num_hyper_layers=0)(layers_comb)
            layers_comb = layers.Add()([layers_comb, aux_comb])

        layers_comb_field = layers.Dense(latDim, kernel_initializer=self.generator.weight_initializer)(layers_comb)
        # layers_comb_field = layers.Dense(latDim)(layers_comb)
        layers_comb_field = layers.Activation('linear', dtype=precision)(layers_comb_field)

        self.field_decoder = tf.keras.Model(coords_het, layers_comb_field, name="field_decoder")

    def call(self, inputs):
        basis = self.field_decoder(inputs)
        return basis


class PhysDecoder:
    # @tf.function(jit_compile=True)
    def __init__(self, generator, CTF="apply", jit_compile=True, precision=tf.float32):
        super(PhysDecoder, self).__init__()
        self.generator = generator
        self.CTF = CTF
        self.precision = precision

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
        rot_batch = tf.cast(tf.gather(self.generator.angle_rot, indexes, axis=0), self.precision)
        tilt_batch = tf.cast(tf.gather(self.generator.angle_tilt, indexes, axis=0), self.precision)
        psi_batch = tf.cast(tf.gather(self.generator.angle_psi, indexes, axis=0), self.precision)

        shifts_x = tf.cast(tf.gather(self.generator.shifts[0], indexes, axis=0), self.precision)
        shifts_y = tf.cast(tf.gather(self.generator.shifts[1], indexes, axis=0), self.precision)

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
        ctf = tf.cast(computeCTF(defocusU_batch, defocusV_batch, defocusAngle_batch, cs_batch, kv_batch,
                                 self.generator.sr, self.generator.pad_factor,
                                 [self.generator.xsize, int(0.5 * self.generator.xsize + 1)],
                                 batch_size_scope, self.generator.applyCTF), self.precision)

        return [rot_batch, tilt_batch, psi_batch], [shifts_x, shifts_y], ctf

    # @tf.function(jit_compile=True)
    def compute_field_volume(self, d):
        # Compute deformation field
        d = self.generator.half_xsize * d
        d_x = d[..., 0]
        d_y = d[..., 1]
        d_z = d[..., 2]

        # Apply deformation field
        c_x = self.generator.applyDeformationFieldVol(d_x, 0)
        c_y = self.generator.applyDeformationFieldVol(d_y, 1)
        c_z = self.generator.applyDeformationFieldVol(d_z, 2)

        return c_x, c_y, c_z

    # @tf.function(jit_compile=True)
    def compute_field_atoms(self, d):
        # Compute atoms deformation field
        d = self.generator.half_xsize * d
        d_x = d[..., 0]
        d_y = d[..., 1]
        d_z = d[..., 2]

        # Apply deformation field
        a_x = self.generator.applyDeformationFieldAtoms(d_x, 0)
        a_y = self.generator.applyDeformationFieldAtoms(d_y, 1)
        a_z = self.generator.applyDeformationFieldAtoms(d_z, 2)

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
    def compute_theo_proj(self, c_x, c_y, ctf, delta_volume):
        # Scatter image and bypass gradient
        decoded = self.generator.scatterImgByPass([c_x, c_y, delta_volume])

        if self.generator.step > 1:
            # Gaussian filter image
            decoded = self.generator.gaussianFilterImage(decoded)

            if self.CTF == "apply":
                # CTF filter image
                decoded_ctf = tf.cast(
                    self.generator.ctfFilterImage(tf.cast(decoded, tf.float32), tf.cast(ctf, tf.float32)),
                    self.precision)
            else:
                decoded_ctf = decoded
        else:
            if self.CTF == "apply":
                # CTF filter image
                decoded_ctf = tf.cast(
                    self.generator.ctfFilterImage(tf.cast(decoded, tf.float32), tf.cast(ctf, tf.float32)),
                    self.precision)
            else:
                decoded_ctf = decoded

        return decoded, decoded_ctf

    # @tf.function(jit_compile=False)
    def __call__(self, x, permute_view=False):
        # encoded, images, indexes = x
        encoded, indexes = x
        alignments, shifts, ctf = self.prepare_batch(indexes, permute_view)

        field, delta_volume, delta_euler, delta_shifts = encoded

        # Compute deformation field
        c_x, c_y, c_z = self.compute_field_volume(field)

        # Bond and angle
        if self.generator.ref_is_struct:
            # Compute atoms deformation field
            a_x, a_y, a_z = self.compute_field_atoms(field)

            bondk, anglek, coords = self.compute_atom_cost_params(a_x, a_y, a_z)

        else:
            bondk = tf.constant(0.0, self.precision)
            anglek = tf.constant(0.0, self.precision)
            coords = tf.constant(0.0, self.precision)

        # Apply alignment and shifts
        c_r_s_x, c_r_s_y = self.apply_alignment_and_shifts(c_x, c_y, c_z, alignments, shifts, delta_euler, delta_shifts)

        # Theoretical projections
        decoded, decoded_ctf = self.compute_theo_proj(c_r_s_x, c_r_s_y, ctf, delta_volume)

        return [decoded, decoded_ctf], bondk, anglek, coords, ctf


class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, architecture="convnn", CTF="apply", mode=None, l_bond=0.01, l_angle=0.01,
                 l_clashes=None, jit_compile=True, latDim=8, precision=tf.float32,
                 compute_delta=True, l_dfm=1.0, poseReg=0.0, ctfReg=0.0, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        generator.mode = "spa"
        self.generator = generator
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.refPose = 1.0 if generator.refinePose else 0.0
        self.mode = generator.mode if mode is None else mode
        self.l_bond = l_bond
        self.l_angle = l_angle
        self.l_clashes = l_clashes if l_clashes is not None else 0.0
        self.l_dfm = l_dfm
        self.poseReg = poseReg
        self.ctfReg = ctfReg
        self.filters = create_blur_filters(5, 10, 30)
        self.latDim = latDim
        # self.architecture = architecture
        self.precision = precision
        self.compute_delta = compute_delta
        self.encoder_exp = Encoder(latDim, generator.xsize, architecture=architecture, mode=self.mode,
                                   jit_compile=jit_compile, refPose=generator.refinePose)
        if poseReg > 0.0:
            self.encoder_clean = Encoder(latDim, generator.xsize, architecture=architecture, mode=self.mode,
                                         jit_compile=jit_compile, refPose=generator.refinePose)
        if ctfReg > 0.0:
            self.encoder_ctf = Encoder(latDim, generator.xsize, architecture=architecture, mode=self.mode,
                                       jit_compile=jit_compile, refPose=generator.refinePose)
        self.basis_decoder = BasisDecoder(generator, latDim=latDim, jit_compile=jit_compile, precision=precision,
                                          compute_delta=compute_delta)
        if compute_delta:
            self.basis_volume_decoder = BasisDecoder(generator, latDim=latDim, jit_compile=jit_compile,
                                                     precision=precision, compute_delta=compute_delta)
            # self.basis_volume_decoder = BasisDecoder(generator, latDim=3 * latDim, jit_compile=jit_compile, precision=precision,
            #                                          compute_delta=compute_delta)
        if l_dfm > 0.0:
            self.inverse_basis_decoder = BasisDecoder(generator, latDim=latDim, jit_compile=jit_compile,
                                                            precision=precision,
                                                            compute_delta=compute_delta)
        self.phys_decoder = PhysDecoder(generator, CTF=self.CTF, jit_compile=jit_compile, precision=precision)
        self.z_space_x = layers.Dense(latDim, name="z_space_x", kernel_initializer=RandomUniform(-0.001, 0.001))
        self.z_space_y = layers.Dense(latDim, name="z_space_y", kernel_initializer=RandomUniform(-0.001, 0.001))
        self.z_space_z = layers.Dense(latDim, name="z_space_z", kernel_initializer=RandomUniform(-0.001, 0.001))
        self.delta_euler = layers.Dense(3, name="delta_euler", trainable=generator.refinePose)
        self.delta_shifts = layers.Dense(2, name="delta_shifts", trainable=generator.refinePose)
        self.activation = layers.Activation('linear', dtype=precision)
        self.disantangle_pose = poseReg > 0.0
        self.disantangle_ctf = ctfReg > 0.0
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.img_loss_tracker = tf.keras.metrics.Mean(name="img_loss")
        self.bond_loss_tracker = tf.keras.metrics.Mean(name="bond_loss")
        self.angle_loss_tracker = tf.keras.metrics.Mean(name="angle_loss")
        self.clash_loss_tracker = tf.keras.metrics.Mean(name="clash_loss")
        self.loss_disantangle_tracker = tf.keras.metrics.Mean(name="loss_disentangled")

        # XLA compilation of cost function
        if jit_compile:
            self.cost_function = tf.function(jit_compile=jit_compile)(self.generator.cost_function)
            self.gradient = tf.function(jit_compile=jit_compile)(self.gradient)
            self.compute_jacobian_autograd = tf.function(jit_compile=jit_compile)(self.compute_jacobian_autograd)
            # self.compute_jacobian_finite_diff = tf.function(jit_compile=jit_compile)(self.compute_jacobian_finite_diff)
            self.compute_jacobian_regularization = tf.function(jit_compile=jit_compile)(
                self.compute_jacobian_regularization)
            self.compute_bending_energy_regularization = tf.function(jit_compile=jit_compile)(
                self.compute_bending_energy_regularization)
            self.compute_hyper_elastic_loss = tf.function(jit_compile=jit_compile)(self.compute_hyper_elastic_loss)
            self.compute_smoothness_regularization = tf.function(jit_compile=jit_compile)(
                self.compute_smoothness_regularization)
            self.compute_div_regularization = tf.function(jit_compile=jit_compile)(self.compute_div_regularization)
            self.compute_rot_regularization = tf.function(jit_compile=jit_compile)(self.compute_rot_regularization)
            self.compute_field_loss = tf.function(jit_compile=jit_compile)(self.compute_field_loss)
            self.compute_field_norm = tf.function(jit_compile=jit_compile)(self.compute_field_norm)
        else:
            self.cost_function = self.generator.cost_function

        if allow_open3d and self.generator.ref_is_struct:
            # Continuous convolution
            # k_clash = 0.6  # Repulsion value (for bb)
            # extent = 1.2  # 2 * radius, typical class distance between 0.4A-0.6A (for bb)
            self.k_clash = 4.  # Repulsion value
            self.extent = 8.  # 2 * radius, typical class distance between 0.4A-0.6A
            self.fn = lambda x, y: simple_clash(x, y, self.k_clash, self.precision)
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
            self.loss_disantangle_tracker,
        ]

    def train_step(self, data):
        inputs = tf.cast(data[0], self.precision)

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

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

        coords = self.generator.scaled_coords  # TESTING: Autograd for field derivatives

        with tf.GradientTape(persistent=True) as tape:
            # tape.watch(coords)  # TESTING: Autograd for field derivatives
            B, C = tf.shape(inputs)[0], tf.shape(coords)[0]

            # Forward pass (first encoder and decoder)
            if self.mode == "spa":
                x, euler, shifts = self.encoder_exp(inputs)
                het_x = self.activation(self.z_space_x(x))
                het_y = self.activation(self.z_space_y(x))
                het_z = self.activation(self.z_space_z(x))
                het = tf.stack([het_x, het_y, het_z], axis=-1)

                basis = self.basis_decoder(coords)
                basis = tf.tile(basis[None, ...], (B, 1, 1))
                field = tf.matmul(basis, het)

                if self.compute_delta:
                    volume_basis = self.basis_volume_decoder(coords)
                    volume_basis = tf.tile(volume_basis[None, ...], (B, 1, 1))
                    delta_volume = tf.squeeze(tf.matmul(volume_basis, tf.reduce_mean(het, axis=-1, keepdims=True)))
                    # delta_volume = tf.squeeze(tf.matmul(volume_basis, tf.reshape(het, (B, 3 * self.latDim, 1))))
                else:
                    delta_volume = 0.0

                encoded = [tf.transpose(field, (1, 0, 2)), delta_volume,
                           self.activation(self.delta_euler(euler)), self.activation(self.delta_shifts(shifts))]
            elif self.mode == "tomo":
                x, latent = self.encoder_exp(inputs)
                het = self.activation(self.z_space(latent))

                het_tiled = tf.tile(het[:, None, :], (1, C, 1))
                coords_tiled = tf.tile(coords[None, :, :], (B, 1, 1))
                coords_het = tf.concat([coords_tiled, het_tiled], axis=-1)

                field = self.basis_decoder(coords_het)
                encoded = [tf.transpose(field, (1, 0, 2)),
                           self.activation(self.delta_euler(x)), self.activation(self.delta_shifts(x))]

            # Field losses

            loss = tf.cast(self.compute_field_loss(het, precision=tf.float32), self.precision)

            ##########################

            # Diffeomorphism loss

            if self.l_dfm > 0.0:
                # Better performance and memory saved
                indices = tf.range(C, dtype=tf.int32)
                indices = tf.random.shuffle(indices)
                indices = indices[:1000]
                field_rnd = tf.gather(field, indices, axis=1)
                coords_tiled_rnd = tf.gather(self.generator.scaled_coords, indices, axis=0)


                convected_coords_rnd = field_rnd + coords_tiled_rnd[None, ...]
                convected_coords_rnd = tf.reshape(convected_coords_rnd, (B * 1000, 3))
                inv_basis_rnd = self.inverse_basis_decoder(convected_coords_rnd)
                inv_basis_rnd = tf.reshape(inv_basis_rnd, (B, 1000, self.latDim))
                inv_field_rnd = tf.matmul(inv_basis_rnd, het)

                loss_dfm = tf.cast(tf.reduce_mean(tf.abs(tf.cast(field_rnd, tf.float32) + tf.cast(inv_field_rnd, tf.float32))), self.precision)
            else:
                loss_dfm = 0.0

            ##########################

            encoded[1] *= self.refPose
            encoded[2] *= self.refPose
            decoded_vec, bondk, anglek, coords, ctf = self.phys_decoder([encoded, indexes], permute_view=False)
            decoded, decoded_ctf = decoded_vec[0], decoded_vec[1]

            if self.disantangle_pose and self.mode == "spa":
                # Forward pass (second decoder - no permutation)
                x, _, _ = self.encoder_clean(decoded)
                het_x = self.activation(self.z_space_x(x))
                het_y = self.activation(self.z_space_y(x))
                het_z = self.activation(self.z_space_z(x))
                het_clean = tf.stack([het_x, het_y, het_z], axis=-1)

                # Forward pass (third encoder - permuted CTF)
                if self.disantangle_ctf and self.CTF is not None:
                    x, _, _ = self.encoder_ctf(decoded_ctf)
                    het_x = self.activation(self.z_space_x(x))
                    het_y = self.activation(self.z_space_y(x))
                    het_z = self.activation(self.z_space_z(x))
                    het_ctf = tf.stack([het_x, het_y, het_z], axis=-1)
                    ctf_perm = tf.random.shuffle(ctf)
                    decoded_ctf_perm = tf.cast(self.generator.ctfFilterImage(tf.cast(decoded, tf.float32), tf.cast(ctf_perm, tf.float32)), self.precision)
                    x, _, _ = self.encoder_ctf(decoded_ctf_perm)
                    het_x = self.activation(self.z_space_x(x))
                    het_y = self.activation(self.z_space_y(x))
                    het_z = self.activation(self.z_space_z(x))
                    het_ctf_perm = tf.stack([het_x, het_y, het_z], axis=-1)
                else:
                    het_ctf_perm = het

                # Forward pass (second decoder - permutation)
                decoded_vec, _, _, _, _ = self.phys_decoder([encoded, indexes], permute_view=True)
                x, _, _ = self.encoder_clean(decoded_vec[0])
                het_x = self.activation(self.z_space_x(x))
                het_y = self.activation(self.z_space_y(x))
                het_z = self.activation(self.z_space_z(x))
                het_clean_perm = tf.stack([het_x, het_y, het_z], axis=-1)

            if allow_open3d and self.generator.ref_is_struct and self.l_clashes > 0.0:
                # Fixed radius search
                result = self.nsearch(coords, coords, 0.5 * self.extent, points_row_splits, queries_row_splits)

                # Compute neighbour distances
                clashes = self.conv(tf.ones((tf.shape(coords)[0], 1), self.precision), coords, coords, self.extent,
                                    user_neighbors_row_splits=result.neighbors_row_splits,
                                    user_neighbors_index=result.neighbors_index,
                                    user_neighbors_importance=self.fn(result.neighbors_distance,
                                                                      result.neighbors_row_splits))
                clashes = tf.cast(tf.reduce_max(tf.reshape(tf.cast(clashes, tf.float32), (B, -1)), axis=-1), self.precision)
            else:
                clashes = tf.constant(0.0, self.precision)

            img_loss = tf.cast(self.cost_function(tf.cast(images, tf.float32),
                                                  tf.cast(decoded_ctf, tf.float32)), self.precision)

            # Bond and angle losses
            if self.generator.ref_is_struct:
                bond_loss = tf.cast(tf.sqrt(tf.reduce_max(tf.keras.losses.MSE(tf.cast(self.generator.bond0, tf.float32),
                                                                              tf.cast(bondk, tf.float32)))), self.precision)
                angle_loss = tf.cast(tf.sqrt(tf.reduce_max(tf.keras.losses.MSE(tf.cast(self.generator.angle0, tf.float32),
                                                                               tf.cast(anglek, tf.float32)))), self.precision)
            else:
                bond_loss, angle_loss = tf.constant(0.0, self.precision), tf.constant(0.0, self.precision)

            # Loss disantagled (pose)
            if self.disantangle_pose and self.mode == "spa":
                loss_disantagled_pose = tf.cast(tf.keras.losses.MSE(tf.cast(het, tf.float32), tf.cast(het_clean, tf.float32))
                                                + tf.keras.losses.MSE(tf.cast(het, tf.float32), tf.cast(het_clean_perm, tf.float32)), self.precision)
            else:
                loss_disantagled_pose = 0.0

            # Loss disantagled (CTF)
            if self.disantangle_ctf and self.mode == "spa" and self.CTF is not None:
                loss_disantagled_ctf = tf.cast(
                    tf.keras.losses.MSE(tf.cast(het, tf.float32), tf.cast(het_ctf, tf.float32))
                    + tf.keras.losses.MSE(tf.cast(het, tf.float32), tf.cast(het_ctf_perm, tf.float32)),
                    self.precision)
            else:
                loss_disantagled_ctf = 0.0

            total_loss = (img_loss + self.l_bond * bond_loss
                          + self.l_angle * angle_loss + self.l_clashes * clashes
                          + 1.0 * loss + self.l_dfm * loss_dfm + self.poseReg * loss_disantagled_pose +
                          self.ctfReg * loss_disantagled_ctf)  # 0.001 works on HetSIREN

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        # gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
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

    def test_step(self, data):
        inputs = tf.cast(data[0], self.precision)

        if self.mode == "spa":
            indexes = data[1]
            images = inputs
        elif self.mode == "tomo":
            indexes = data[1][0]
            images = inputs[0]

        # self.decoder.generator.indexes = indexes
        # self.decoder.generator.current_images = images

        # Row splits
        if allow_open3d and self.generator.ref_is_struct:
            B = tf.shape(images)[0]
            # num_points = self.generator.atom_coords.shape[0]
            num_points = tf.cast(tf.shape(self.generator.ca_indices)[0], tf.int64)
            points_row_splits = tf.range(B + 1, dtype=tf.int64) * num_points
            queries_row_splits = tf.range(B + 1, dtype=tf.int64) * num_points

        # Prepare batch
        # images = self.decoder.prepare_batch([images, indexes])

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
            het = self.activation(self.z_space(x))
            encoded = [tf.tranpose(self.field_decoder([het, self.generator.scaled_coords]), (1, 0, 2)),
                       self.activation(self.delta_euler(x)), self.activation(self.delta_shifts(x))]
        elif self.mode == "tomo":
            x, latent = self.encoder_exp(inputs)
            het = self.activation(self.z_space(latent))
            encoded = [tf.tranpose(self.field_decoder([het, self.generator.scaled_coords]), (1, 0, 2)),
                       self.activation(self.delta_euler(x)), self.activation(self.delta_shifts(x))]
        encoded[1] *= self.refPose
        encoded[2] *= self.refPose
        decoded_vec, bondk, anglek, coords, _, loss = self.phys_decoder([encoded, indexes], permute_view=False)
        decoded, decoded_ctf, decoded_only_field_ctf = decoded_vec[0], decoded_vec[1], decoded_vec[2]

        if self.disantangle and self.mode == "spa":
            # Forward pass (second decoder - no permutation)
            x = self.encoder_clean(decoded)
            het_clean = self.activation(self.z_space(x))

            # Forward pass (second decoder - permutation)
            decoded_vec, _, _, _, _, loss = self.phys_decoder([encoded, indexes], permute_view=True)
            x = self.encoder_clean(decoded_vec[0])
            het_clean_perm = self.activation(self.z_space(x))

        if allow_open3d and self.generator.ref_is_struct:
            # Fixed radius search
            result = self.nsearch(coords, coords, 0.5 * self.extent, points_row_splits, queries_row_splits)

            # Compute neighbour distances
            clashes = self.conv(tf.ones((tf.shape(coords)[0], 1), self.precision), coords, coords, self.extent,
                                user_neighbors_row_splits=result.neighbors_row_splits,
                                user_neighbors_index=result.neighbors_index,
                                user_neighbors_importance=self.fn(result.neighbors_distance,
                                                                  result.neighbors_row_splits))
            clashes = tf.reduce_max(tf.reshape(clashes, (B, -1)), axis=-1)
        else:
            clashes = tf.constant(0.0, self.precision)

        img_loss = self.cost_function(images, decoded_ctf)

        # Bond and angle losses
        if self.generator.ref_is_struct:
            bond_loss = tf.sqrt(tf.reduce_max(tf.keras.losses.MSE(self.generator.bond0, bondk)))
            angle_loss = tf.sqrt(tf.reduce_max(tf.keras.losses.MSE(self.generator.angle0, anglek)))
        else:
            bond_loss, angle_loss = tf.constant(0.0, self.precision), tf.constant(0.0, self.precision)

        # Loss disantagled
        if self.disantangle and self.mode == "spa":
            loss_disantagled = tf.keras.losses.MSE(het, het_clean) + tf.keras.losses.MSE(het, het_clean_perm)
        else:
            loss_disantagled = 0.0

        total_loss = (img_loss + self.l_bond * bond_loss
                      + self.l_angle * angle_loss + self.l_clashes * clashes
                      + 0.001 * loss + 0.01 * loss_disantagled)  # 0.001 works on HetSIREN

        self.total_loss_tracker.update_state(total_loss)
        self.img_loss_tracker.update_state(img_loss)
        self.angle_loss_tracker.update_state(angle_loss)
        self.bond_loss_tracker.update_state(bond_loss)
        self.clash_loss_tracker.update_state(clashes)
        self.loss_disantangle_tracker.update_state(loss_disantagled)
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

        # if self.CTF == "wiener":
        #     images = self.decoder.generator.wiener2DFilter(images)
        #     if self.mode == "spa":
        #         inputs = images
        #     elif self.mode == "tomo":
        #         inputs[0] = images

        if self.mode == "spa":
            x, euler, shifts = self.encoder_exp(inputs)
            het_x = self.activation(self.z_space_x(x))
            het_y = self.activation(self.z_space_y(x))
            het_z = self.activation(self.z_space_z(x))
            het = tf.stack([het_x, het_y, het_z], axis=-1)
            het = tf.reshape(het, (-1, self.latDim * 3))
            encoded = [het, self.activation(self.delta_euler(euler)), self.activation(self.delta_shifts(shifts))]
        elif self.mode == "tomo":
            x, latent = self.encoder(inputs)
            encoded = [self.activation(self.z_space(latent)), self.activation(self.delta_euler(x)), self.activation(self.delta_shifts(x))]
        encoded[1] *= self.refPose
        encoded[2] *= self.refPose

        return encoded

    def convect_maps(self, z):
        B, C = tf.shape(z)[0], tf.shape(self.generator.scaled_coords)[0]

        # Decoded motion fields
        z = tf.cast(z, tf.float32)
        z = tf.reshape(z, (B, self.latDim, 3))

        basis = self.basis_decoder(self.generator.scaled_coords)
        basis = tf.tile(basis[None, ...], (B, 1, 1))
        d = tf.matmul(basis, z)
        d = self.generator.half_xsize * d
        d = d.numpy()
        d = np.stack([d[..., 2], d[..., 1], d[..., 0]], axis=-1)

        if self.compute_delta:
            volume_basis = self.basis_volume_decoder(self.generator.scaled_coords)
            volume_basis = tf.tile(volume_basis[None, ...], (B, 1, 1))
            delta_volume = tf.squeeze(tf.matmul(volume_basis, tf.reduce_mean(z, axis=-1, keepdims=True)))
            # delta_volume = tf.squeeze(tf.matmul(volume_basis, tf.reshape(z, (B, 3 * self.latDim, 1))))
        else:
            delta_volume = 0.0

        v = (tf.tile(self.generator.values[None, ...], (B, 1)) + delta_volume).numpy()

        # Convect indices
        convected_indices = np.round(self.generator.indices.numpy()[None, ...] + d).astype(int)

        # Scatter in volume
        convected_vols = np.zeros((d.shape[0], self.generator.xsize, self.generator.xsize, self.generator.xsize))
        for idx in range(convected_indices.shape[0]):
            if np.all(np.abs(convected_indices[idx]) < self.generator.xsize):
                convected_vol = np.zeros((self.generator.xsize, self.generator.xsize, self.generator.xsize))
                convected_vol[
                    convected_indices[idx, :, 0], convected_indices[idx, :, 1], convected_indices[idx, :, 2]] += v[idx]

                # Gaussian filter map
                convected_vol = gaussian_filter(convected_vol, sigma=1.0)

                convected_vols[idx] = convected_vol

        return convected_vols

    # def gradient(self, inputs, targets, tape):
    #     return tape.gradient(targets, inputs)

    # def gradient(self, inputs, axis):
    #     # Axis 0 --> X | Axis 1 --> Y | Axis 2 --> X
    #     with tf.GradientTape() as tape:
    #         tape.watch(inputs)
    #         decoded = self.field_decoder(inputs)[..., axis]
    #     return tape.gradient(decoded, inputs)[0, :, :3]

    def gradient(self, inputs, coords, axis, precision=tf.float32):
        # Axis 0 --> X | Axis 1 --> Y | Axis 2 --> X
        B, C = tf.shape(inputs)[0], tf.shape(coords)[0]
        coords = tf.tile(coords[None, ...], (B, 1, 1))
        with tf.GradientTape() as tape_1:
            tape_1.watch(coords)
            coords_rshp = tf.reshape(coords, (B * C, 3))
            basis = self.basis_decoder(coords_rshp / self.generator.half_xsize)
            basis = tf.reshape(basis, (B, C, self.latDim))
            field = tf.matmul(basis, inputs)[..., axis]
            # convected_coords = (coords + field)[..., axis]
        return tape_1.gradient(field, coords)

    def gradient_gradient(self, inputs, coords, axis, precision=tf.float32):
        # Axis 0 --> X | Axis 1 --> Y | Axis 2 --> X
        B, C = tf.shape(inputs)[0], tf.shape(coords)[0]
        coords = tf.tile(coords[None, ...], (B, 1, 1))
        with tf.GradientTape(persistent=True) as tape_2:
            tape_2.watch(coords)
            with tf.GradientTape() as tape_1:
                tape_1.watch(coords)
                coords_rshp = tf.reshape(coords, (B * C, 3))
                basis = self.basis_decoder(coords_rshp / self.generator.half_xsize)
                basis = tf.reshape(basis, (B, C, self.latDim))
                field = tf.matmul(basis, inputs)[..., axis]
                # convected_coords = (coords + field)[..., axis]
            gradient = tape_1.gradient(field, coords)
            gradient_x, gradient_y, gradient_z = gradient[..., 0], gradient[..., 1], gradient[..., 2]
        gradient_2_x = tape_2.gradient(gradient_x, coords)
        gradient_2_y = tape_2.gradient(gradient_y, coords)
        gradient_2_z = tape_2.gradient(gradient_z, coords)
        gradient_2 = tf.stack([gradient_2_x, gradient_2_y, gradient_2_z], axis=2)
        return gradient, gradient_2

    def compute_jacobian_autograd(self, inputs, coords, second_derivative=False, precision=tf.float32):
        """Compute the Jacobian matrix of the output wrt the input."""
        if second_derivative:
            jacobian_matrix_x, jacobian_matrix_2_x = self.gradient_gradient(inputs, coords, 0, precision=precision)
            jacobian_matrix_y, jacobian_matrix_2_y = self.gradient_gradient(inputs, coords, 1, precision=precision)
            jacobian_matrix_z, jacobian_matrix_2_z = self.gradient_gradient(inputs, coords, 2, precision=precision)
            jacobian_matrix = tf.stack([jacobian_matrix_x, jacobian_matrix_y, jacobian_matrix_z], axis=2)
            jacobian_matrix_2 = tf.stack([jacobian_matrix_2_x, jacobian_matrix_2_y, jacobian_matrix_2_z], axis=2)
            return jacobian_matrix, jacobian_matrix_2
        else:
            jacobian_matrix_x = self.gradient(inputs, coords, 0, precision=precision)[:, :, None, :]
            jacobian_matrix_y = self.gradient(inputs, coords, 1, precision=precision)[:, :, None, :]
            jacobian_matrix_z = self.gradient(inputs, coords, 2, precision=precision)[:, :, None, :]
            jacobian_matrix = tf.concat([jacobian_matrix_x, jacobian_matrix_y, jacobian_matrix_z], axis=2)
            return jacobian_matrix, None

    # def compute_jacobian_finite_diff(self, deformation_field, mask_indices, grid_shape):
    #     """
    #     Compute the smoothness loss for a deformation field given the mask indices.
    #
    #     :param deformation_field: Tensor of shape (B, M, 3)
    #     :param mask_indices: Tensor of shape (B, M, 3), contains the indices within the grid
    #     :param grid_shape: Tuple representing the shape of the grid (H, W, D)
    #     :return: Smoothness loss
    #     """
    #     B = tf.shape(deformation_field)[0]
    #     M = tf.shape(deformation_field)[1]
    #
    #     # Compute central differences in the x, y, z directions
    #     shifts = tf.constant([
    #         [1, 0, 0], [-1, 0, 0],
    #         [0, 1, 0], [0, -1, 0],
    #         [0, 0, 1], [0, 0, -1]
    #     ], dtype=tf.int32)
    #
    #     # Create neighbors by shifting mask_indices
    #     mask_indices_expanded = tf.expand_dims(mask_indices, axis=2)  # Shape (B, M, 1, 3)
    #     neighbors = mask_indices_expanded + shifts  # Shape (B, M, 6, 3)
    #     neighbors = tf.clip_by_value(neighbors, 0, tf.constant(grid_shape) - 1)  # Ensure neighbors are within bounds
    #
    #     # Flatten the batch and point dimensions
    #     batch_indices = tf.range(B, dtype=tf.int32)[:, tf.newaxis, tf.newaxis, tf.newaxis]
    #     batch_indices = tf.tile(batch_indices, [1, M, 6, 1])  # Shape (B, M, 6, 1)
    #
    #     flat_neighbors = tf.reshape(neighbors, [-1, 3])  # Shape (B*M*6, 3)
    #     flat_batch_indices = tf.reshape(batch_indices, [-1, 1])  # Shape (B*M*6, 1)
    #
    #     # Combine batch indices with neighbor indices
    #     flat_neighbors_with_batch = tf.concat([flat_batch_indices, flat_neighbors], axis=1)
    #
    #     # Create a flat version of the deformation field
    #     flat_deformation_field = tf.reshape(deformation_field, [B * M, 3])  # Shape (B*M, 3)
    #
    #     # Repeat the deformation field for each neighbor direction
    #     deformation_field_repeated = tf.tile(flat_deformation_field, [6, 1])  # Shape (B*M*6, 3)
    #
    #     # Compute linear indices for gathering neighbors
    #     neighbor_indices = flat_neighbors_with_batch[:, 1] * grid_shape[1] * grid_shape[2] + \
    #                        flat_neighbors_with_batch[:, 2] * grid_shape[2] + \
    #                        flat_neighbors_with_batch[:, 3]
    #
    #     # Gather neighbor values
    #     neighbor_vals = tf.gather(deformation_field_repeated, neighbor_indices)
    #     neighbor_vals = tf.reshape(neighbor_vals, [B, M, 6, 3])
    #
    #     # Calculate central differences
    #     central_diffs = (neighbor_vals[:, :, ::2, :] - neighbor_vals[:, :, 1::2, :]) / 2.0  # Shape (B, M, 3, 3)
    #
    #     return central_diffs

    def compute_jacobian_regularization(self, jacobians, precision=tf.float32):
        # Compute the Jacobian matrix for each point
        # jacobians = tf.transpose(jacobians, perm=[0, 1, 3, 2])  # Shape (B, M, 3, 3)
        B, M = tf.shape(jacobians)[0], tf.shape(jacobians)[1]
        jacobians = tf.cast(jacobians, precision)

        jacobians = tf.tile(tf.eye(3, dtype=precision)[None, None, ...], (B, M, 1, 1)) + jacobians

        # Compute the singular values of the Jacobian matrix for each point
        jacobians = tf.cast(jacobians, tf.float32)
        singular_values = tf.linalg.svd(jacobians, compute_uv=False)
        singular_values = tf.cast(singular_values, precision)

        # Regularization term: penalize singular values deviating from 1
        singular_value_reg = tf.reduce_mean(tf.square(singular_values - 1))

        return singular_value_reg

    def compute_bending_energy_regularization(self, jacobians_2, precision=tf.float32):
        jacobians_2 = tf.cast(jacobians_2, precision)

        dx_xyz = jacobians_2[:, :, 0, :, :]
        dy_xyz = jacobians_2[:, :, 1, :, :]
        dz_xyz = jacobians_2[:, :, 2, :, :]

        dx_xyz = tf.square(dx_xyz)
        dy_xyz = tf.square(dy_xyz)
        dz_xyz = tf.square(dz_xyz)

        loss = (
                tf.reduce_mean(dx_xyz[:, :, :, 0])
                + tf.reduce_mean(dy_xyz[:, :, :, 1])
                + tf.reduce_mean(dz_xyz[:, :, :, 2])
        )
        loss += (
                2. * tf.reduce_mean(dx_xyz[:, :, :, 1])
                + 2. * tf.reduce_mean(dx_xyz[:, :, :, 2])
                + tf.reduce_mean(dy_xyz[:, :, :, 2])
        )

        return loss

    def compute_hyper_elastic_loss(self, jacobians, alpha_l=1, alpha_a=1, alpha_v=1, precision=tf.float32):
        """Compute the hyper-elastic regularization loss."""
        # Compute the Jacobian matrix for each point
        # jacobians = tf.transpose(jacobians, perm=[0, 1, 3, 2])  # Shape (B, M, 3, 3)
        B, M = tf.shape(jacobians)[0], tf.shape(jacobians)[1]

        jacobians = tf.cast(jacobians, precision)
        grad_u = jacobians
        grad_y = tf.tile(tf.eye(3, dtype=precision)[None, None, ...], (B, M, 1, 1)) + grad_u

        # Compute length loss
        length_loss = tf.linalg.norm(grad_u, axis=(2, 3))
        length_loss = tf.pow(length_loss, 2.)
        length_loss = tf.reduce_mean(length_loss)
        length_loss = 0.5 * alpha_l * length_loss

        # Compute cofactor matrices for the area loss
        cofactors_row_1 = []
        cofactors_row_2 = []
        cofactors_row_3 = []

        # Compute elements of cofactor matrices one by one (Ugliest solution ever?)
        cofactors_row_1.append(compute_determinant_2x2(grad_y[:, :, 1:, 1:]))
        cofactors_row_1.append(compute_determinant_2x2(grad_y[:, :, 1:, 0::2]))
        cofactors_row_1.append(compute_determinant_2x2(grad_y[:, :, 1:, :2]))
        cofactors_row_2.append(compute_determinant_2x2(grad_y[:, :, 0::2, 1:]))
        cofactors_row_2.append(compute_determinant_2x2(grad_y[:, :, 0::2, 0::2]))
        cofactors_row_2.append(compute_determinant_2x2(grad_y[:, :, 0::2, :2]))
        cofactors_row_3.append(compute_determinant_2x2(grad_y[:, :, :2, 1:]))
        cofactors_row_3.append(compute_determinant_2x2(grad_y[:, :, :2, 0::2]))
        cofactors_row_3.append(compute_determinant_2x2(grad_y[:, :, :2, :2]))
        cofactors_row_1 = tf.stack(cofactors_row_1, axis=-1)
        cofactors_row_2 = tf.stack(cofactors_row_2, axis=-1)
        cofactors_row_3 = tf.stack(cofactors_row_3, axis=-1)
        cofactors = tf.concat([cofactors_row_1[:, :, None, :],
                               cofactors_row_2[:, :, None, :],
                               cofactors_row_3[:, :, None, :]], axis=2)

        # Compute area loss
        area_loss = tf.pow(cofactors, 2.)
        area_loss = tf.reduce_sum(area_loss, axis=2)
        area_loss = area_loss - 1.
        area_loss = tf.reduce_max(tf.concat((area_loss, tf.zeros_like(area_loss)), axis=-1), axis=-1)
        area_loss = tf.pow(area_loss, 2.)
        area_loss = tf.reduce_mean(tf.reduce_sum(area_loss, axis=-1))  # sum over dimension 1 and then 0
        area_loss = alpha_a * area_loss

        # # Compute volume loss
        volume_loss = compute_determinant_3x3(grad_y)
        # volume_loss = tf.matmul(tf.pow(volume_loss - 1., 4.)[..., None], tf.pow(volume_loss, -2.)[:, None, :])
        fn = lambda x: tf.matmul(tf.pow(x - 1., 4.)[None, :], tf.pow(x, -2.)[None, :], transpose_b=True)
        volume_loss = tf.map_fn(fn, volume_loss, fn_output_signature=precision)
        volume_loss = tf.reduce_mean(volume_loss)
        volume_loss = alpha_v * volume_loss

        # Compute total loss
        loss = length_loss + area_loss + volume_loss

        return loss

    def compute_smoothness_regularization(self, jacobians):
        # Compute the L2 norm of the spatial derivatives
        smoothness_reg = tf.reduce_mean(tf.square(jacobians))

        return smoothness_reg

    def compute_div_regularization(self, D):
        # D shape: (B, M, 3, 3)
        # Compute divergence: d_phi_x/dx + d_phi_y/dy + d_phi_z/dz
        div = D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]
        # Compute E_div: sum of squared divergences
        E_div = tf.reduce_mean(div ** 2)
        return E_div

    def compute_rot_regularization(self, D):
        # D shape: (B, M, 3, 3)
        # Compute curl components
        curl_x = D[..., 2, 1] - D[..., 1, 2]
        curl_y = D[..., 0, 2] - D[..., 2, 0]
        curl_z = D[..., 1, 0] - D[..., 0, 1]
        # Compute E_rot: sum of squared curls
        E_rot = tf.reduce_mean(curl_x ** 2 + curl_y ** 2 + curl_z ** 2)
        return E_rot

    def compute_field_loss(self, inputs, second_derivative=True, precision=tf.float32):

        # Better performance and memory saved
        indices = tf.range(tf.shape(self.generator.scaled_coords)[0], dtype=tf.int32)
        indices = tf.random.shuffle(indices)
        indices = indices[:1000]
        coords = tf.gather(self.generator.scaled_coords, indices, axis=0)

        # Compute Jacobians
        jacobians, jacobians_2 = self.compute_jacobian_autograd(inputs, coords, second_derivative=second_derivative,
                                                                precision=precision)
        jacobians = tf.cast(jacobians, tf.float32)
        jacobians_2 = tf.cast(jacobians_2, tf.float32)

        # Jacobian regularization
        jacobian_reg = self.compute_jacobian_regularization(jacobians, precision=precision)
        # smoothness_reg = self.compute_smoothness_regularization(jacobians)
        # div_reg = self.compute_div_regularization(derivatives)
        # rot_reg = self.compute_rot_regularization(derivatives)
        hyper_elastic_reg = self.compute_hyper_elastic_loss(jacobians, precision=precision)

        # Jacobian-Jacobian regularization
        if second_derivative:
            bending_energy_reg = self.compute_bending_energy_regularization(jacobians_2, precision=precision)
        else:
            bending_energy_reg = 0.0

        return (jacobian_reg + 1.0 * bending_energy_reg + 1.0 * hyper_elastic_reg) / self.generator.half_xsize

    def compute_field_norm(self, d_x, d_y, d_z):
        return tf.reduce_sum(tf.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)) / (self.generator.xsize *
                                                                            self.generator.xsize *
                                                                            self.generator.xsize)

    def call(self, input_features):
        if allow_open3d and self.generator.ref_is_struct:
            # To know this weights exist
            coords = tf.zeros((1, 3), self.precision)
            _ = self.nsearch(coords, coords, 1.0)
            _ = self.conv(tf.ones((tf.shape(coords)[0], 1), self.precision), coords, coords, self.extent)

        if self.mode == "spa":
            B, C = tf.shape(input_features)[0], tf.shape(self.generator.scaled_coords)[0]

            indexes = tf.zeros(B, dtype=tf.int32)
            x, euler, shifts = self.encoder_exp(input_features)
            het_x = self.activation(self.z_space_x(x))
            het_y = self.activation(self.z_space_y(x))
            het_z = self.activation(self.z_space_z(x))
            het = tf.stack([het_x, het_y, het_z], axis=-1)

            basis = self.basis_decoder(self.generator.scaled_coords)
            basis = tf.tile(basis[None, ...], (B, 1, 1))
            field = tf.matmul(basis, het)

            if self.compute_delta:
                volume_basis = self.basis_volume_decoder(self.generator.scaled_coords)
                volume_basis = tf.tile(volume_basis[None, ...], (B, 1, 1))
                delta_volume = tf.squeeze(tf.matmul(volume_basis, tf.reduce_mean(het, axis=-1, keepdims=True)))
                # delta_volume = tf.squeeze(tf.matmul(volume_basis, tf.reshape(het, (B, 3 * self.latDim, 1))))
            else:
                delta_volume = 0.0

            encoded = [tf.transpose(field, (1, 0, 2)), delta_volume, self.delta_euler(euler), self.delta_shifts(shifts)]
        elif self.mode == "tomo":
            B, C = tf.shape(input_features[0])[0], tf.shape(self.generator.scaled_coords)[0]

            indexes = tf.zeros(B, dtype=tf.int32)
            x, latent = self.encoder_exp(input_features)
            # _ = self.encoder_clean(input_features)
            # _ = self.encoder_ctf(input_features)

            het = self.z_space(latent)
            het_tiled = tf.tile(het[:, None, :], (1, C, 1))
            coords_tiled = tf.tile(self.generator.scaled_coords[None, :, :], (B, 1, 1))
            coords_het = tf.concat([coords_tiled, het_tiled], axis=-1)

            field = self.basis_decoder(coords_het)
            encoded = [tf.transpose(field, (1, 0, 2)), self.delta_euler(x), self.delta_shifts(x)]

        return self.phys_decoder([encoded, indexes])

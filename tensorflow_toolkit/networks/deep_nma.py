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
from tensorflow.keras import layers

from tensorflow_toolkit.utils import computeCTF, euler_matrix_batch
from tensorflow_toolkit.layers.residue_conv2d import ResidueConv2D


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, refinePose, architecture="convnn"):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        l2 = tf.keras.regularizers.l2(1e-3)
        # shift_activation = lambda y: 2 * tf.keras.activations.tanh(y)

        encoder_inputs = tf.keras.Input(shape=(input_dim, input_dim, 1))

        x = tf.keras.layers.Flatten()(encoder_inputs)

        if architecture == "mlpnn":
            for _ in range(12):
                x = layers.Dense(1024, activation='relu', kernel_regularizer=l2)(x)
            x = layers.Dropout(0.3)(x)
            x = layers.BatchNormalization()(x)

        elif architecture == "convnn":
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

            # x = tf.keras.layers.Conv2D(4, 5, activation="relu", strides=(2, 2), padding="same")(x)
            # x = tf.keras.layers.Conv2D(8, 5, activation="relu", strides=(2, 2), padding="same")(x)
            # x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(x)
            # x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=(2, 2), padding="same")(x)
            # x = tf.keras.layers.Flatten()(x)
            # x = tf.keras.layers.Dropout(.1)(x)
            # x = tf.keras.layers.BatchNormalization()(x)

        if refinePose:
            c_nma = layers.Dense(latent_dim, activation="linear", name="c_nma")(x)
            delta_euler = layers.Dense(3, activation="linear", name="delta_euler")(x)
            delta_shifts = layers.Dense(2, activation="linear", name="delta_shifts")(x)
            self.encoder = tf.keras.Model(encoder_inputs,
                                          [c_nma, delta_euler, delta_shifts], name="encoder")
        else:
            c_nma = layers.Dense(latent_dim, activation="linear", name="c_nma")(x)
            self.encoder = tf.keras.Model(encoder_inputs, c_nma, name="encoder")

    def call(self, x):
        return self.encoder(x)


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim, generator, CTF="apply"):
        super(Decoder, self).__init__()
        self.generator = generator
        self.CTF = CTF

        c_nma = tf.keras.Input(shape=(latent_dim,))
        if self.generator.refinePose:
            delta_euler = tf.keras.Input(shape=(3,))
            delta_shifts = tf.keras.Input(shape=(2,))

        # Compute deformation field
        d_x, d_y, d_z = layers.Lambda(self.generator.computeDeformationField, trainable=True)(c_nma)

        # Apply deformation field
        c_x = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 0), trainable=True)(d_x)
        c_y = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 1), trainable=True)(d_y)
        c_z = layers.Lambda(lambda inp: self.generator.applyDeformationField(inp, 2), trainable=True)(d_z)

        # Bond and angle
        bondk = layers.Lambda(lambda inp: self.generator.calc_bond(inp), trainable=True)([c_x, c_y, c_z])
        anglek = layers.Lambda(lambda inp: self.generator.calc_angle(inp), trainable=True)([c_x, c_y, c_z])

        # Apply alignment
        if self.generator.refinePose:
            c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentDeltaEuler(inp, 0), trainable=True) \
                ([c_x, c_y, c_z, delta_euler])
            c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentDeltaEuler(inp, 1), trainable=True) \
                ([c_x, c_y, c_z, delta_euler])
        else:
            c_r_x = layers.Lambda(lambda inp: self.generator.applyAlignmentMatrix(inp, 0), trainable=True)(
                [c_x, c_y, c_z])
            c_r_y = layers.Lambda(lambda inp: self.generator.applyAlignmentMatrix(inp, 1), trainable=True)(
                [c_x, c_y, c_z])

        # Apply shifts
        if self.generator.refinePose:
            c_r_s_x = layers.Lambda(lambda inp: self.generator.applyDeltaShifts(inp, 0), trainable=True) \
                ([c_r_x, delta_shifts])
            c_r_s_y = layers.Lambda(lambda inp: self.generator.applyDeltaShifts(inp, 1), trainable=True) \
                ([c_r_y, delta_shifts])
        else:
            c_r_s_x = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 0), trainable=True)(c_r_x)
            c_r_s_y = layers.Lambda(lambda inp: self.generator.applyShifts(inp, 1), trainable=True)(c_r_y)

        # Scatter image and bypass gradient
        decoded = layers.Lambda(self.generator.scatterImgByPass, trainable=True)([c_r_s_x, c_r_s_y])

        # Gaussian filter image
        decoded = layers.Lambda(self.generator.gaussianFilterImage)(decoded)

        if self.CTF == "apply":
            # CTF filter image
            decoded = layers.Lambda(self.generator.ctfFilterImage)(decoded)

        if self.generator.refinePose:
            self.decoder = tf.keras.Model([c_nma, delta_euler, delta_shifts], [decoded, bondk, anglek], name="decoder")
        else:
            self.decoder = tf.keras.Model(c_nma, [decoded, bondk, anglek], name="decoder")

    def call(self, x):
        decoded = self.decoder(x)
        # self.generator.def_coords = [d_x, d_y, d_z]
        return decoded


class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, architecture="convnn", CTF="apply", multires=(2, ),
                 l_bond=0.01, l_angle=0.01, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.multires = [tf.constant([int(generator.xsize / mr), int(generator.xsize / mr)], dtype=tf.int32) for mr in
                         multires]
        self.l_bond = l_bond
        self.l_angle = l_angle
        self.encoder = Encoder(generator.n_modes, generator.xsize,
                               generator.refinePose, architecture=architecture)
        self.decoder = Decoder(generator.n_modes, generator, CTF=CTF)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.img_loss_tracker = tf.keras.metrics.Mean(name="img_loss")
        self.bond_loss_tracker = tf.keras.metrics.Mean(name="bond_loss")
        self.angle_loss_tracker = tf.keras.metrics.Mean(name="angle_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.img_loss_tracker,
            self.bond_loss_tracker,
            self.angle_loss_tracker
        ]

    def train_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
        if self.decoder.generator.refinePose:
            self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
            self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
            self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)
        else:
            rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
            tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
            psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)
            self.decoder.generator.r = euler_matrix_batch(rot_batch, tilt_batch, psi_batch)

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
            encoded = self.encoder(images)
            decoded, bondk, anglek = self.decoder(encoded)

            img_loss = self.decoder.generator.cost_function(images, decoded)

            # Multiresolution loss
            multires_loss = 0.0
            for mr in self.multires:
                images_mr = self.decoder.generator.downSampleImages(images, mr)
                decoded_mr = self.decoder.generator.downSampleImages(decoded, mr)
                multires_loss += self.decoder.generator.cost_function(images_mr, decoded_mr)
            multires_loss = multires_loss / len(self.multires)

            # Bond and angle losses
            bond_loss = tf.sqrt(tf.reduce_mean(tf.keras.losses.MSE(self.decoder.generator.bond0, bondk)))
            angle_loss = tf.sqrt(tf.reduce_mean(tf.keras.losses.MSE(self.decoder.generator.angle0, anglek)))

            total_loss = img_loss + multires_loss + self.l_bond * bond_loss + self.l_angle * angle_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.img_loss_tracker.update_state(img_loss)
        self.angle_loss_tracker.update_state(angle_loss)
        self.bond_loss_tracker.update_state(bond_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "img_loss": self.img_loss_tracker.result(),
            "bond": self.bond_loss_tracker.result(),
            "angle": self.angle_loss_tracker.result(),
        }

    def test_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

        # Precompute batch alignments
        if self.decoder.generator.refinePose:
            self.decoder.generator.rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
            self.decoder.generator.tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
            self.decoder.generator.psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)
        else:
            rot_batch = tf.gather(self.decoder.generator.angle_rot, data[1], axis=0)
            tilt_batch = tf.gather(self.decoder.generator.angle_tilt, data[1], axis=0)
            psi_batch = tf.gather(self.decoder.generator.angle_psi, data[1], axis=0)
            self.decoder.generator.r = euler_matrix_batch(rot_batch, tilt_batch, psi_batch)

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

        encoded = self.encoder(images)
        decoded, bondk, anglek = self.decoder(encoded)

        img_loss = self.decoder.generator.cost_function(images, decoded)

        # Multiresolution loss
        multires_loss = 0.0
        for mr in self.multires:
            images_mr = self.decoder.generator.downSampleImages(images, mr)
            decoded_mr = self.decoder.generator.downSampleImages(decoded, mr)
            multires_loss += self.decoder.generator.cost_function(images_mr, decoded_mr)
        multires_loss = multires_loss / len(self.multires)

        # Bond and angle losses
        bond_loss = tf.sqrt(tf.reduce_mean(tf.keras.losses.MSE(self.decoder.generator.bond0, bondk)))
        angle_loss = tf.sqrt(tf.reduce_mean(tf.keras.losses.MSE(self.decoder.generator.angle0, anglek)))

        total_loss = img_loss + multires_loss + self.l_bond * bond_loss + self.l_angle * angle_loss

        self.total_loss_tracker.update_state(total_loss)
        self.img_loss_tracker.update_state(img_loss)
        self.angle_loss_tracker.update_state(angle_loss)
        self.bond_loss_tracker.update_state(bond_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "img_loss": self.img_loss_tracker.result(),
            "bond": self.bond_loss_tracker.result(),
            "angle": self.angle_loss_tracker.result(),
        }

    def predict_step(self, data):
        self.decoder.generator.indexes = data[1]
        self.decoder.generator.current_images = data[0]

        images = data[0]

        # Update batch_size (in case it is incomplete)
        batch_size_scope = tf.shape(data[0])[0]

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

        encoded = self.encoder(images)
        return encoded

    def call(self, input_features):
        return self.decoder(self.encoder(input_features))

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

from tensorflow_toolkit.utils import computeCTF
from tensorflow_toolkit.layers.siren import Sine, SIRENInitializer


class Encoder(tf.keras.Model):
    def __init__(self, input_dim, architecture="convnn", maxAngleDiff=5., maxShiftDiff=2.):
        super(Encoder, self).__init__()

        images = tf.keras.Input(shape=(input_dim, input_dim, 1))

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

        rows = layers.Dense(256, activation="relu")(x)
        for _ in range(2):
            rows = layers.Dense(256, activation="relu")(rows)
        # rows = layers.Dense(3, activation=sigmoid_diff, trainable=refPose)(rows)
        rows = layers.Dense(3, activation=Sine(w0=1.0), kernel_initializer=SIRENInitializer())(rows)
        # rows = layers.Dense(6, activation="linear", trainable=refPose)(rows)

        shifts = layers.Dense(256, activation="relu")(x)
        for _ in range(2):
            shifts = layers.Dense(256, activation="relu")(shifts)
        # shifts = layers.Dense(2, activation=sigmoid_diff, trainable=refPose)(shifts)
        shifts = layers.Dense(2, activation=Sine(w0=1.0), kernel_initializer=SIRENInitializer())(shifts)

        self.encoder = tf.keras.Model(images, [maxAngleDiff * rows, maxShiftDiff * shifts], name="encoder")

    def call(self, x):
        encoded = self.encoder(x)
        return encoded


class Decoder(tf.keras.Model):
    def __init__(self, generator, CTF="apply"):
        super(Decoder, self).__init__()
        self.generator = generator
        self.CTF = CTF

        alignment = tf.keras.Input(shape=(3,))
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
                decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(decoded)
        else:
            if self.CTF == "apply":
                # CTF filter image
                decoded_ctf = layers.Lambda(self.generator.ctfFilterImage)(scatter_images)

        self.decoder = tf.keras.Model([alignment, shifts], decoded_ctf, name="decoder")

    def call(self, x):
        decoded = self.decoder(x)
        return decoded


class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, architecture="convnn", CTF="apply", n_gradients=1,
                 maxAngleDiff=5., maxShiftDiff=2., multires=(2, ), **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.CTF = CTF if generator.applyCTF == 1 else None
        self.multires = [tf.constant([int(generator.xsize / mr), int(generator.xsize / mr)], dtype=tf.int32) for mr in
                         multires]
        self.encoder = Encoder(generator.xsize, architecture=architecture,
                               maxAngleDiff=maxAngleDiff, maxShiftDiff=maxShiftDiff)
        self.decoder = Decoder(generator, CTF=CTF)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_loss_tracker = tf.keras.metrics.Mean(name="test_loss")

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
            encoded = self.encoder(images)
            decoded = self.decoder(encoded)

            # Multiresolution loss
            multires_loss = 0.0
            for mr in self.multires:
                images_mr = self.decoder.generator.downSampleImages(images, mr)
                decoded_mr = self.decoder.generator.downSampleImages(decoded, mr)
                multires_loss += self.decoder.generator.cost_function(images_mr, decoded_mr)
            multires_loss = multires_loss / len(self.multires)

            total_loss = self.decoder.generator.cost_function(images, decoded) + multires_loss

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

        encoded = self.encoder(images)
        decoded = self.decoder(encoded)

        # Multiresolution loss
        multires_loss = 0.0
        for mr in self.multires:
            images_mr = self.decoder.generator.downSampleImages(images, mr)
            decoded_mr = self.decoder.generator.downSampleImages(decoded, mr)
            multires_loss += self.decoder.generator.cost_function(images_mr, decoded_mr)
        multires_loss = multires_loss / len(self.multires)

        total_loss = self.decoder.generator.cost_function(images, decoded) + multires_loss

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
        return self.decoder(self.encoder(input_features))

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

        # Decode inputs
        encoded = self.encoder(images)

        return encoded

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
import numpy as np


class SpaceEncoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(SpaceEncoder, self).__init__()
        input_data = tf.keras.Input(shape=(input_dim,))

        x = tf.keras.layers.Dense(1024, activation="relu")(input_data)
        for _ in range(3):
            aux_x = tf.keras.layers.Dense(1024, activation="relu")(x)
            x = tf.keras.layers.Add()([x, aux_x])

        x = tf.keras.layers.Dense(latent_dim, activation="linear")(x)

        self.encoder = tf.keras.Model(input_data, x)

    def call(self, x):
        latent = self.encoder(x)
        return latent


class SpaceDecoder(tf.keras.Model):
    def __init__(self, output_dim, lat_dim):
        super(SpaceDecoder, self).__init__()
        input_data = tf.keras.Input(shape=(lat_dim,))

        x = tf.keras.layers.Dense(1024, activation="relu")(input_data)
        for _ in range(3):
            aux_x = tf.keras.layers.Dense(1024, activation="relu")(x)
            x = tf.keras.layers.Add()([x, aux_x])

        x = tf.keras.layers.Dense(output_dim, activation="linear")(x)

        self.encoder = tf.keras.Model(input_data, x)

    def call(self, x):
        latent = self.encoder(x)
        return latent


class AutoEncoder(tf.keras.Model):
    def __init__(self, generator, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.generator = generator
        self.space_encoders = [SpaceEncoder(input_dim, generator.lat_dim) for input_dim in generator.space_dims]
        self.space_decoders = [SpaceDecoder(input_dim, generator.lat_dim) for input_dim in generator.space_dims]
        self.loss_tracker = [tf.keras.metrics.Mean(name="total_loss"), tf.keras.metrics.Mean(name="encoder_loss")]
        self.decoder_loss_tracker = [tf.keras.metrics.Mean(name="space_decoder_%d" % idx)
                                     for idx in range(len(generator.space_dims))]
        self.test_loss_tracker = [tf.keras.metrics.Mean(name="test_loss"),]
        self.loss_tracker += self.decoder_loss_tracker
        self.loss_tracker += self.test_loss_tracker

        # Variables for prediction
        self.encoder_idx = None
        self.decoder_idx = None

    @property
    def metrics(self):
        return self.loss_tracker

    def train_step(self, data):
        inputs = data[0]
        encoder_losses = []
        decoder_losses = []
        total_losses = []

        for idx, space_decoder in enumerate(self.space_decoders):
            weights_to_train = []
            with tf.GradientTape() as tape:

                # Encode spaces
                space_encoded = [space_encoder(input_data)
                                 for space_encoder, input_data in zip(self.space_encoders, inputs)]

                # Decode spaces
                space_decoded = []
                for input_features in space_encoded:
                    space_decoded.append(space_decoder(input_features))

                # Encoder losses (single space)
                encoder_loss_1 = self.generator.compute_encoder_loss(space_encoded)

                # Encoder losses (keep distances)
                encoder_loss_2 = self.generator.compute_shannon_loss(inputs, space_encoded)

                # Encoder loss
                encoder_loss = encoder_loss_1 + encoder_loss_2
                encoder_losses.append(encoder_loss)

                # Decoder losses
                decoder_loss = self.generator.compute_decoder_loss(inputs[idx], space_decoded)
                decoder_losses.append(decoder_loss)

                # Total loss
                total_loss = encoder_loss + decoder_loss
                total_losses.append(total_loss)

            # Get Submodel weights
            for space_encoder in self.space_encoders:
                weights_to_train += space_encoder.trainable_weights
            weights_to_train += space_decoder.trainable_weights

            grads = tape.gradient(total_loss, weights_to_train)
            self.optimizer.apply_gradients(zip(grads, weights_to_train))

        self.loss_tracker[0].update_state(sum(total_losses))
        self.loss_tracker[1].update_state(sum(encoder_losses))
        loss_dict = {"loss": self.loss_tracker[0].result(), "enc_loss": self.loss_tracker[1].result()}
        for idx in range(len(self.space_decoders)):
            shift_idx = idx + 2
            self.loss_tracker[shift_idx].update_state(decoder_losses[idx])
            loss_dict["dec_loss_%d" % (idx + 1)] = self.loss_tracker[shift_idx].result()
        return loss_dict

    def test_step(self, data):
        inputs = data[0]
        encoder_losses = []
        decoder_losses = []
        total_losses = []

        for idx, space_decoder in enumerate(self.space_decoders):
            # Encode spaces
            space_encoded = [space_encoder(input_data)
                             for space_encoder, input_data in zip(self.space_encoders, inputs)]

            # Decode spaces
            space_decoded = []
            for input_features in space_encoded:
                space_decoded.append(space_decoder(input_features))

            # Encoder losses (single space)
            encoder_loss_1 = self.generator.compute_encoder_loss(space_encoded)

            # Encoder losses (keep distances)
            encoder_loss_2 = self.generator.compute_shannon_loss(inputs, space_encoded)

            # Encoder loss
            encoder_loss = encoder_loss_1 + encoder_loss_2
            encoder_losses.append(encoder_loss)

            # Decoder losses
            decoder_loss = self.generator.compute_decoder_loss(inputs[idx], space_decoded)
            decoder_losses.append(decoder_loss)

            # Total loss
            total_loss = encoder_loss + decoder_loss
            total_losses.append(total_loss)

        self.loss_tracker[0].update_state(sum(total_losses))
        self.loss_tracker[1].update_state(sum(encoder_losses))
        loss_dict = {"loss": self.loss_tracker[0].result(), "enc_loss": self.loss_tracker[1].result()}
        for idx in range(len(self.space_decoders)):
            shift_idx = idx + 2
            self.loss_tracker[shift_idx].update_state(decoder_losses[idx])
            loss_dict["dec_loss_%d" % (idx + 1)] = self.loss_tracker[shift_idx].result()
        return loss_dict

    def call(self, input_features):
        space_encoded = [space_encoder(input_data)
                         for space_encoder, input_data in zip(self.space_encoders, input_features)]
        space_decoded = [space_decoder(encoded)
                         for space_decoder, encoded in zip(self.space_decoders, space_encoded)]
        return space_decoded

    def encode_space(self, input_features, input_encoder_idx):
        encoded = self.space_encoders[input_encoder_idx].predict(input_features)
        return encoded

    def decode_space(self, input_features, output_decoder_idx):
        decoded = self.space_decoders[output_decoder_idx].predict(input_features)
        return decoded

    def find_encoder(self, data):
        lowest_error = None
        encoder_idx = None
        idx = 0
        for encoder, decoder in zip(self.space_encoders, self.space_decoders):
            autoencoder = tf.keras.Sequential([encoder, decoder])
            encoder_input_shape = encoder.layers[0].input_shape
            if data.shape[1] == encoder_input_shape[1]:
                decoded_space = autoencoder.predict(data, verbose=0)
                error = np.mean(self.generator.rmse(data, decoded_space))
                if lowest_error is None or error < lowest_error:
                    lowest_error = error
                    encoder_idx = idx
            idx += 1
        return encoder_idx

    def predict(self, data, encoder_idx, decoder_idx):
        self.encoder_idx, self.decoder_idx = encoder_idx, decoder_idx
        self.predict_function = None
        decoded = super().predict(data)
        return decoded

    def predict_step(self, data):
        encoder, decoder = self.space_encoders[self.encoder_idx], self.space_decoders[self.decoder_idx]
        decoded = decoder(encoder(data))
        self.encoder_idx, self.decoder_idx = None, None
        return decoded

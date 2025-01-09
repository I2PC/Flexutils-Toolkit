#!/usr/bin/env python
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


import os
import numpy as np
import shutil
import glob
import re
from xmipp_metadata.metadata import XmippMetaData

import tensorflow as tf

from deprecated.generator_deep_pose import Generator
from deprecated.deep_pose import AutoEncoder
# from tensorflow_toolkit.datasets.dataset_template import sequence_to_data_pipeline, create_dataset
from tensorflow_toolkit.utils import epochs_from_iterations


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def apply_perturbation(model, strength=0.01):
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            weights = layer.get_weights()
            perturbed_weights = [w + np.random.normal(0, strength, w.shape) for w in weights]
            layer.set_weights(perturbed_weights)

class ResetOptimizerCallback(tf.keras.callbacks.Callback):
    def __init__(self, reset_epochs, new_learning_rate):
        super(ResetOptimizerCallback, self).__init__()
        self.reset_epochs = reset_epochs
        self.new_learning_rate = new_learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.reset_epochs:
            idx = self.reset_epochs.index(epoch)
            # Re-instantiate the optimizer with a new learning rate
            tf.keras.backend.set_value(self.model.optimizer.lr, 0.0001)
            # self.model.optimizer = tf.keras.optimizers.Adam(learning_rate=self.new_learning_rate[idx])

class AddNoiseToWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def on_batch_end(self, batch, logs=None):
        for weight in self.model.trainable_weights:
            noise = tf.random.normal(weight.shape, stddev=self.stddev)
            weight.assign_add(noise)


def train(outPath, md_file, batch_size, shuffle, step, splitTrain, epochs, cost,
          radius_mask, smooth_mask, architecture="convnn", weigths_file=None,
          ctfType="apply", pad=2, sr=1.0, applyCTF=1, lr=1e-5, multires=[2, 4, 8],
          jit_compile=True, tensorboard=True):

    try:
        # Create data generator
        generator = Generator(md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                              step=step, splitTrain=splitTrain, cost=cost, radius_mask=radius_mask,
                              smooth_mask=smooth_mask, pad_factor=pad,
                              sr=sr, applyCTF=applyCTF)

        # Create validation generator
        if splitTrain < 1.0:
            generator_val = Generator(md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                                      step=step, splitTrain=(splitTrain - 1.0), cost=cost, radius_mask=radius_mask,
                                      smooth_mask=smooth_mask, pad_factor=pad,
                                      sr=sr, applyCTF=applyCTF)
        else:
            generator_val = None

        # Tensorflow data pipeline
        # generator_dataset, generator = sequence_to_data_pipeline(generator)
        # dataset = create_dataset(generator_dataset, generator, batch_size=batch_size)

        # Train model
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        #     autoencoder = AutoEncoder(generator, architecture=architecture)
        # autoencoder = []
        # for _ in range(1):
        #     autoencoder.append(AutoEncoder(generator, architecture=architecture, CTF=ctfType, multires=multires,
        #                        maxAngleDiff=10., maxShiftDiff=1.0))
        autoencoder = AutoEncoder(generator, architecture=architecture, CTF=ctfType, multires=False,
                                  maxAngleDiff=10., maxShiftDiff=1.0)

        # Fine tune a previous model
        if weigths_file:
            autoencoder.build(input_shape=(None, generator.xsize, generator.xsize, 1))
            autoencoder.load_weights(weigths_file)

        # if tf.__version__ == '2.11.0':
        #     # optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-4)
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # else:
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Callbacks list
        callbacks = []

        # Create a callback that saves the model's weights
        initial_epoch = 0
        checkpoint_path = os.path.join(outPath, "training", "cp-{epoch:04d}.hdf5")
        if not os.path.isdir(os.path.dirname(checkpoint_path)):
            os.mkdir(os.path.dirname(checkpoint_path))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        callbacks.append(cp_callback)

        # Callbacks list
        if tensorboard:
            # Tensorboard callback
            log_dir = os.path.join(outPath, "logs")
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                                  write_graph=True, write_steps_per_second=True)

            callbacks.append(tensorboard_callback)

        checkpoint = os.path.join(outPath, "training")
        if os.path.isdir(checkpoint):
            files = glob.glob(os.path.join(checkpoint, "*"))
            if len(files) > 1:
                files.sort()
                latest = files[-2]
                autoencoder.build(input_shape=(None, generator.xsize, generator.xsize, 1))
                autoencoder.load_weights(latest)
                latest = os.path.basename(latest)
                initial_epoch = int(re.findall(r'\d+', latest)[0]) - 1

        # Callbacks list
        callbacks = [cp_callback]
        if tensorboard:
            callbacks.append(tensorboard_callback)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        autoencoder.compile(optimizer=optimizer, jit_compile=jit_compile)

        if generator_val is not None:
            autoencoder.fit(generator, validation_data=generator_val, epochs=epochs, validation_freq=2,
                                 callbacks=callbacks, initial_epoch=initial_epoch)
        else:
            # optim_epochs = list(range(0, epochs, 50))
            # optim_lr = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
            # optim_callback = ResetOptimizerCallback(optim_epochs, optim_lr)
            # callbacks.append(optim_callback)
            autoencoder.fit(generator, epochs=epochs,
                            callbacks=callbacks, initial_epoch=initial_epoch)

        # autoencoder[0].decoder.generator.angle_rot = np.zeros(len(generator.metadata))
        # autoencoder[0].decoder.generator.angle_tilt = np.zeros(len(generator.metadata))
        # autoencoder[0].decoder.generator.angle_psi = np.zeros(len(generator.metadata))
        #
        # for idx in range(5):
        #     # apply_perturbation(autoencoder[idx], strength=0.1)
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        #     autoencoder[idx].compile(optimizer=optimizer, jit_compile=jit_compile)
        #
        #     if generator_val is not None:
        #         autoencoder[idx].fit(generator, validation_data=generator_val, epochs=epochs, validation_freq=2,
        #                              callbacks=callbacks, initial_epoch=initial_epoch)
        #     else:
        #         autoencoder[idx].fit(generator, epochs=epochs,
        #                              callbacks=callbacks, initial_epoch=initial_epoch)
        #
        #     alignment, shifts = autoencoder[idx].predict(generator)
        #     euler_angles = np.zeros((alignment.shape[0], 3))
        #     idy = 0
        #     for matrix in alignment:
        #         euler_angles[idy] = xmippEulerFromMatrix(matrix)
        #         idy += 1
        #
        #     generator = Generator(md_file=md_file, shuffle=shuffle, batch_size=batch_size,
        #                           step=step, splitTrain=splitTrain, cost=cost, radius_mask=radius_mask,
        #                           smooth_mask=smooth_mask, pad_factor=pad,
        #                           sr=sr, applyCTF=applyCTF)
        #     generator.angle_rot = tf.constant(euler_angles[:, 0], dtype=tf.float32)
        #     generator.angle_tilt = tf.constant(euler_angles[:, 1], dtype=tf.float32)
        #     generator.angle_psi = tf.constant(euler_angles[:, 2], dtype=tf.float32)
        #     autoencoder.append(AutoEncoder(generator, architecture=architecture, CTF=ctfType, multires=multires,
        #                        maxAngleDiff=10., maxShiftDiff=1.0))

    except tf.errors.ResourceExhaustedError as error:
        msg = "GPU memory has been exhausted. Usually this can be solved by " \
              "downsampling further your particles or by decreasing the batch size. " \
              "Please, modify any of these two options in the form and try again."
        print(msg)
        raise error

    # Save model
    autoencoder.save_weights(os.path.join(outPath, "deep_pose_model.h5"))

    # Remove checkpoints
    shutil.rmtree(checkpoint)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--max_samples_seen', type=int, required=False)
    parser.add_argument('--cost', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--radius_mask', type=float, required=False, default=2)
    parser.add_argument('--smooth_mask', action='store_true')
    parser.add_argument('--weigths_file', type=str, required=False, default=None)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--apply_ctf', type=int, required=True)
    parser.add_argument('--multires', type=str, required=False)
    parser.add_argument('--jit_compile', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.max_samples_seen:
        n_samples = len(XmippMetaData(args.md_file))
        del XmippMetaData
        epochs = epochs_from_iterations(args.max_samples_seen, n_samples, args.batch_size)
    elif args.epochs:
        epochs = args.epochs
    else:
        raise ValueError("Error: Either parameter --epochs or --max_samples_seen is needed")

    multires = [int(x) for x in args.multires.split(",")]

    inputs = {"md_file": args.md_file, "outPath": args.out_path,
              "batch_size": args.batch_size, "shuffle": args.shuffle,
              "step": args.step, "splitTrain": args.split_train, "epochs": epochs,
              "cost": args.cost, "radius_mask": args.radius_mask, "smooth_mask": args.smooth_mask,
              "architecture": args.architecture, "weigths_file": args.weigths_file, "ctfType": args.ctf_type, "pad": args.pad,
              "sr": args.sr, "applyCTF": args.apply_ctf, "lr": args.lr, "multires": multires,
              "jit_compile": args.jit_compile, "tensorboard": args.tensorboard}

    # Initialize volume slicer
    train(**inputs)

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
import shutil
import glob
import re
from importlib.metadata import version
from scipy.stats import entropy
from scipy.interpolate import NearestNDInterpolator
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "0"
if version("tensorflow") >= "2.16.0":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

from tensorflow_toolkit.generators.generator_flex_consensus import Generator
from tensorflow_toolkit.networks.flex_consensus import AutoEncoder
# from tensorflow_toolkit.datasets.dataset_template import sequence_to_data_pipeline, create_dataset
from tensorflow_toolkit.utils import epochs_from_iterations


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def train(outPath, dataPath, latDim, batch_size, shuffle, splitTrain, epochs, lr=1e-5, tensorboard=True):

    try:
        # Read data
        data_files = glob.glob(os.path.join(dataPath, "*.txt"))
        sort_nicely(data_files)
        spaces = [np.loadtxt(file) for file in data_files]

        # Create data generator
        generator = Generator(spaces, latent_dim=latDim, batch_size=batch_size,
                              shuffle=shuffle, splitTrain=splitTrain)

        # Create validation generator
        if splitTrain < 1.0:
            generator_val = Generator(spaces, latent_dim=latDim, batch_size=batch_size,
                                      shuffle=shuffle, splitTrain=(splitTrain - 1.0))
        else:
            generator_val = None

        # Tensorflow data pipeline
        # generator_dataset, generator = sequence_to_data_pipeline(generator)
        # dataset = create_dataset(generator_dataset, generator, batch_size=batch_size)

        # Train model
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            autoencoder = AutoEncoder(generator)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Callbacks list
        callbacks = []

        # Create a callback that saves the model's weights
        initial_epoch = 0
        if tf.__version__ < "2.16.0" or os.environ["TF_USE_LEGACY_KERAS"] == "1":
            checkpoint_path = os.path.join(outPath, "training", "cp-{epoch:04d}.hdf5")
        else:
            checkpoint_path = os.path.join(outPath, "training", "cp-{epoch:04d}.weights.h5")
        if not os.path.isdir(os.path.dirname(checkpoint_path)):
            os.mkdir(os.path.dirname(checkpoint_path))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        callbacks.append(cp_callback)

        # Tensorboard callback
        if tensorboard:
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
                _ = autoencoder(generator[0][0])
                autoencoder.load_weights(latest)
                latest = os.path.basename(latest)
                initial_epoch = int(re.findall(r'\d+', latest)[0]) - 1

        autoencoder.compile(optimizer=optimizer, jit_compile=False)
        optimizer.build(autoencoder.trainable_variables)

        if generator_val is not None:
            autoencoder.fit(generator, validation_data=generator_val, epochs=epochs, validation_freq=2,
                            callbacks=callbacks, initial_epoch=initial_epoch)
        else:
            autoencoder.fit(generator, epochs=epochs,
                            callbacks=callbacks, initial_epoch=initial_epoch)
    except tf.errors.ResourceExhaustedError as error:
        msg = "GPU memory has been exhausted. Usually this can be solved by " \
              "by decreasing the batch size. Please, modify these " \
              "option in the form and try again."
        print(msg)
        raise error

    # Save model
    if tf.__version__ < "2.16.0" or os.environ["TF_USE_LEGACY_KERAS"] == "1":
        autoencoder.save_weights(os.path.join(outPath, "network", "flex_consensus_model.h5"))
    else:
        autoencoder.save_weights(os.path.join(outPath, "network", "flex_consensus_model.weights.h5"))

    # Remove checkpoints
    shutil.rmtree(checkpoint)

    # Get templates for future matching
    consensus_error = 0.0
    representation_error = 0.0
    mean_std_consensus_error = np.zeros(2)
    mean_std_representation_error = np.zeros((len(spaces), 2))
    for idx in range(len(spaces)):
        encoded_1 = autoencoder.encode_space(spaces[idx], idx)
        for idy in range(len(spaces)):
            encoded_2 = autoencoder.encode_space(spaces[idy], idy)
            consensus_error += generator.rmse(encoded_1, encoded_2)
            representation_error += generator.rmse(spaces[idx], autoencoder.decode_space(encoded_2, idx))
        representation_error = representation_error / len(spaces)
        mean_std_representation_error[idx] = np.array([np.mean(representation_error), np.std(representation_error)])
        representation_error = 0.0
    consensus_error = consensus_error / (len(spaces) * len(spaces))
    mean_std_consensus_error[0], mean_std_consensus_error[1] = np.mean(consensus_error), np.std(consensus_error)

    # Save means and stds of computed errors
    np.save(os.path.join(outPath, "mean_std_consensus_error.npy"), mean_std_consensus_error)
    np.save(os.path.join(outPath, "mean_std_representation_error.npy"), mean_std_representation_error)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--lat_dim', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--max_samples_seen', type=int, required=False)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.max_samples_seen:
        file = glob.glob(os.path.join(args.data_path, "*.txt"))[0]
        n_samples = len(np.loadtxt(file))
        epochs = epochs_from_iterations(args.max_samples_seen, n_samples, args.batch_size)
    elif args.epochs:
        epochs = args.epochs
    else:
        raise ValueError("Error: Either parameter --epochs or --max_samples_seen is needed")

    inputs = {"dataPath": args.data_path, "outPath": args.out_path, "latDim": args.lat_dim,
              "batch_size": args.batch_size, "shuffle": args.shuffle,
              "splitTrain": args.split_train, "epochs": epochs, "lr": args.lr, "tensorboard": args.tensorboard}

    # Initialize volume slicer
    train(**inputs)

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
import re
from glob import glob
from scipy.stats import entropy
from scipy.interpolate import NearestNDInterpolator

import numpy as np
import tensorflow as tf

from toolkit.tensorflow_toolkit.generators.generator_flex_consensus import Generator
from toolkit.tensorflow_toolkit.networks.flex_consensus import AutoEncoder

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


def train(outPath, dataPath, latDim, batch_size, shuffle, splitTrain, epochs):
    # Read data
    data_files = glob(os.path.join(dataPath, "*.txt"))
    sort_nicely(data_files)
    spaces = [np.loadtxt(file) for file in data_files]

    # Create data generator
    generator = Generator(spaces, latent_dim=latDim, batch_size=batch_size,
                          shuffle=shuffle, splitTrain=splitTrain)

    # Train model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        autoencoder = AutoEncoder(generator)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    autoencoder.compile(optimizer=optimizer)
    optimizer.build(autoencoder.trainable_variables)
    autoencoder.fit(generator, epochs=epochs)

    # Save model
    autoencoder.save_weights(os.path.join(outPath, "network", "flex_consensus_model"))

    # Get templates for future matching
    best_mean = [None, None]
    best_entropy = [None, None]
    error_interpolators = {}
    for idx in range(len(spaces)):
        data = spaces[idx]
        for idy in range(len(spaces)):
            if idx != idy:
                decoded = autoencoder.predict(data, encoder_idx=idx, decoder_idx=idy)
                error = generator.rmse(spaces[idy], decoded)
                error_interpolators[f"{idx}_{idy}"] = NearestNDInterpolator([*decoded], error)
                mean_error = np.mean(error)
                entropy_error = entropy(error)
                if best_mean[0] is None or mean_error < best_mean[0]:
                    best_mean[0] = mean_error
                    best_mean[1] = error
                if best_entropy[0] is None or entropy_error < best_entropy[0]:
                    best_entropy[0] = entropy_error
                    best_entropy[1] = error

    # Save best templates
    np.savetxt(os.path.join(outPath, "template_mean_error.txt"), best_mean[1])
    np.savetxt(os.path.join(outPath, "template_entropy_error.txt"), best_entropy[1])

    # Save interpolators
    np.save(os.path.join(outPath, "error_interpolators.npy"), error_interpolators)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--lat_dim', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"dataPath": args.data_path, "outPath": args.out_path, "latDim": args.lat_dim,
              "batch_size": args.batch_size, "shuffle": args.shuffle,
              "splitTrain": args.split_train, "epochs": args.epochs}

    # Initialize volume slicer
    train(**inputs)

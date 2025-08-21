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
from pathlib import Path
from importlib.metadata import version

os.environ["TF_USE_LEGACY_KERAS"] = "0"
if version("tensorflow") >= "2.16.0":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

from tensorflow_toolkit.generators.generator_flex_consensus import Generator
from tensorflow_toolkit.networks.flex_consensus import AutoEncoder

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


def predict(outPath, dataPath, weigths_file, latDim):
    # Read data
    data_files = glob(os.path.join(dataPath, "*.txt"))
    sort_nicely(data_files)
    spaces = [np.loadtxt(file) for file in data_files]

    # Load template histograms
    templates_data_path = Path(weigths_file).parent.parent

    # Load data for generator (needed to create the network properly)
    data_files = glob(os.path.join(templates_data_path, "data", "*.txt"))
    sort_nicely(data_files)
    spaces_network = [np.loadtxt(file) for file in data_files]

    # Load mean and std
    mean_std_cons = np.load(os.path.join(templates_data_path, "mean_std_consensus_error.npy"))
    mean_std_rep = np.load(os.path.join(templates_data_path, "mean_std_representation_error.npy"))

    # Create data generator
    generator = Generator(spaces_network, latent_dim=latDim, batch_size=64,
                          shuffle=False, splitTrain=1.0)

    # Save some memory
    del spaces_network

    # Load model
    autoencoder = AutoEncoder(generator)
    _ = autoencoder(generator[0][0])
    autoencoder.load_weights(weigths_file)

    # Predict step
    data = spaces[0]
    print("Finding best encoder...")
    idx = autoencoder.find_encoder(data)
    encoded = autoencoder.encode_space(data, input_encoder_idx=idx, batch_size=1024)
    print("Best encoder is %d" % idx)
    representation_error = np.zeros(data.shape[0])
    consensus_error = np.zeros(data.shape[0])
    num_decoders = len(autoencoder.space_decoders)
    for idy in range(num_decoders):
        # if idx != idy:
        decoded = autoencoder.decode_space(encoded, output_decoder_idx=idy, batch_size=1024)
        encoded_aux = autoencoder.encode_space(decoded, input_encoder_idx=idy, batch_size=1024)
        consensus_error += generator.rmse(encoded, encoded_aux)
    consensus_error = consensus_error / num_decoders
    decoded = autoencoder.predict(data, encoder_idx=idx, decoder_idx=idx)
    representation_error += generator.rmse(data, decoded)
    consensus_error = generator.logistic_transform_std_shift(consensus_error, mean_std_cons[0], mean_std_cons[1])
    representation_error = generator.logistic_transform_std_shift(representation_error, mean_std_rep[idx, 0], mean_std_rep[idx, 1])

    # Save matched error distributions
    np.save(os.path.join(outPath, "consensus_latents.npy"), encoded)
    np.save(os.path.join(outPath, "representation_error.npy"), representation_error)
    np.save(os.path.join(outPath, "consensus_error.npy"), consensus_error)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--weigths_file', type=str, required=True)
    parser.add_argument('--lat_dim', type=int, required=True)
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"dataPath": args.data_path, "outPath": args.out_path, "latDim": args.lat_dim,
              "weigths_file": args.weigths_file}

    # Initialize volume slicer
    predict(**inputs)

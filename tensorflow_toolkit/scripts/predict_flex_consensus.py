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
    template_mean_error = np.loadtxt(os.path.join(templates_data_path, "template_mean_error.txt"))
    template_entropy_error = np.loadtxt(os.path.join(templates_data_path, "template_entropy_error.txt"))

    # Load data for generator (needed to create the network properly)
    data_files = glob(os.path.join(templates_data_path, "data", "*.txt"))
    sort_nicely(data_files)
    spaces_network = [np.loadtxt(file) for file in data_files]

    # Load interpolators
    error_interpolators = np.load(os.path.join(templates_data_path, "error_interpolators.npy"), allow_pickle=True)[()]

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
    print("Best encoder is %d" % idx)
    error_matched_mean, error_matched_entropy = np.zeros(data.shape[0]), np.zeros(data.shape[0])
    num_decoders = len(autoencoder.space_decoders)
    for idy in range(num_decoders):
        if idx != idy:
            decoded = autoencoder.predict(data, encoder_idx=idx, decoder_idx=idy)
            error = error_interpolators[f"{idx}_{idy}"](decoded)
            error_matched_mean += generator.hist_match(error, template_mean_error)
            error_matched_entropy += generator.hist_match(error, template_entropy_error)
    error_matched_mean = error_matched_mean / (num_decoders - 1)
    error_matched_entropy = error_matched_entropy / (num_decoders - 1)

    # Save matched error distributions
    np.savetxt(os.path.join(outPath, "error_matched_mean.txt"), error_matched_mean)
    np.savetxt(os.path.join(outPath, "error_matched_entropy.txt"), error_matched_entropy)


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

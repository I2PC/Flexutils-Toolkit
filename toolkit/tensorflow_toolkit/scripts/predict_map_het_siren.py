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
import mrcfile
from pathlib import Path
from sklearn.cluster import KMeans
from xmipp_metadata.image_handler import ImageHandler

import tensorflow as tf

from tensorflow_toolkit.generators.generator_het_siren import Generator
from tensorflow_toolkit.networks.het_siren import AutoEncoder


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def predict(weigths_file, het_file, out_path, allCoords=False, filter=False, **kwargs):
    x_het = np.loadtxt(het_file)
    if len(x_het.shape) == 1:
        x_het = x_het.reshape((1, -1))
    md_file = Path(Path(weigths_file).parent.parent, "input_particles.xmd")

    # Create data generator
    generator = Generator(md_file=md_file, step=kwargs.pop("step"), shuffle=False)

    # Load model
    autoencoder = AutoEncoder(generator, het_dim=x_het.shape[1], **kwargs)
    autoencoder.load_weights(weigths_file).expect_partial()

    # Decode maps
    decoded_maps = autoencoder.eval_volume_het(x_het, allCoords=allCoords, filter=filter)
    for idx, decoded_map in enumerate(decoded_maps):
        decoded_path = Path(out_path, 'decoded_map_class_%d.mrc' % (idx + 1))
        ImageHandler().write(decoded_map, decoded_path, overwrite=True)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weigths_file', type=str, required=True)
    parser.add_argument('--het_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)

    args = parser.parse_args()

    if hasattr(args, "gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"weigths_file": args.weigths_file, "het_file": args.het_file,
              "out_path": args.out_path, "step": args.step}

    # Initialize volume slicer
    predict(**inputs)

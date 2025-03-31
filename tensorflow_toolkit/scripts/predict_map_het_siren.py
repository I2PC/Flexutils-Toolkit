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
import h5py
import mrcfile
from importlib.metadata import version
from pathlib import Path
from sklearn.cluster import KMeans
from xmipp_metadata.image_handler import ImageHandler

os.environ["TF_USE_LEGACY_KERAS"] = "0"
if version("tensorflow") >= "2.16.0":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

from tensorflow_toolkit.generators.generator_het_siren import Generator
from tensorflow_toolkit.networks.het_siren import AutoEncoder


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def predict(weigths_file, het_file, out_path, allCoords=False, filter=True, architecture="convnn",
            poseReg=0.0, ctfReg=0.0, refPose=True, use_hyper_network=True, **kwargs):
    x_het = np.loadtxt(het_file)
    if len(x_het.shape) == 1:
        x_het = x_het.reshape((1, -1))
    md_file = Path(Path(weigths_file).parent.parent, "input_particles.xmd")

    # Get xsize from weights file
    f = h5py.File(weigths_file, 'r')
    xsize = int(np.sqrt(f["encoder"]["dense"]["kernel:0"].shape[0]))

    # Create data generator
    generator = Generator(md_file=md_file, step=kwargs.pop("step"), shuffle=False,
                          xsize=xsize)

    # Load model
    autoencoder = AutoEncoder(generator, het_dim=x_het.shape[1], architecture=architecture,
                              poseReg=poseReg, ctfReg=ctfReg, refPose=refPose, use_hyper_network=use_hyper_network,
                              **kwargs)
    if generator.mode == "spa":
        data = np.zeros((1, generator.xsize, generator.xsize, 1))
    elif generator.mode == "tomo":
        data = [np.zeros((1, generator.xsize, generator.xsize, 1)), np.zeros((1, generator.sinusoid_table.shape[1]))]
    _ = autoencoder(data)
    autoencoder.load_weights(weigths_file)

    # Decode maps
    decoded_maps = autoencoder.eval_volume_het(x_het, allCoords=allCoords, filter=filter, add_to_original=True)
    for idx, decoded_map in enumerate(decoded_maps):
        decoded_path = Path(out_path, 'decoded_map_class_%02d.mrc' % (idx + 1))
        ImageHandler().write(decoded_map, decoded_path, overwrite=True)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weigths_file', type=str, required=True)
    parser.add_argument('--het_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--pose_reg', type=float, required=False, default=0.0)
    parser.add_argument('--ctf_reg', type=float, required=False, default=0.0)
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--use_hyper_network', action='store_true')
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"weigths_file": args.weigths_file, "het_file": args.het_file,
              "out_path": args.out_path, "step": args.step, "architecture": args.architecture,
              "poseReg": args.pose_reg, "ctfReg": args.ctf_reg, "refPose": args.refine_pose,
              "use_hyper_network": args.use_hyper_network}

    # Initialize volume slicer
    predict(**inputs)

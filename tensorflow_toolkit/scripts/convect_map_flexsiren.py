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
from pathlib import Path
from importlib.metadata import version
from xmipp_metadata.image_handler import ImageHandler

if version("tensorflow") >= "2.16.0":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

from tensorflow_toolkit.generators.generator_flexsiren import Generator
from tensorflow_toolkit.networks.flexsiren_basis import AutoEncoder


def predict(weigths_file, het_file, out_path, architecture="mlpnn",
            poseReg=0.0, ctfReg=0.0, refinePose=True, **kwargs):
    x_het = np.loadtxt(het_file)
    if len(x_het.shape) == 1:
        x_het = x_het.reshape((1, -1))
    md_file = Path(Path(weigths_file).parent.parent, "input_particles.xmd")

    # Get xsize from weights file
    f = h5py.File(weigths_file, 'r')
    xsize = int(np.sqrt(f["encoder"]["dense"]["kernel:0"].shape[0]))

    # Create data generator
    generator = Generator(md_file=md_file, step=kwargs.pop("step"), shuffle=False,
                          xsize=xsize, refinePose=refinePose)

    # Load model
    autoencoder = AutoEncoder(generator, latDim=x_het.shape[1], architecture=architecture,
                              poseReg=poseReg, ctfReg=ctfReg, **kwargs)
    if generator.mode == "spa":
        autoencoder.build(input_shape=(None, generator.xsize, generator.xsize, 1))
    elif generator.mode == "tomo":
        autoencoder.build(input_shape=[(None, generator.xsize, generator.xsize, 1),
                                       [None, generator.sinusoid_table.shape[1]]])
    autoencoder.load_weights(weigths_file)

    # Convect maps
    convected_maps = autoencoder.convect_maps(x_het)
    for idx, decoded_map in enumerate(convected_maps):
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
    parser.add_argument('--refine_pos', type=float, required=False, default=0.0)
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
              "poseReg": args.pose_reg, "ctfReg": args.ctf_reg, "refinePose": args.refine_pose}

    # Initialize volume slicer
    predict(**inputs)

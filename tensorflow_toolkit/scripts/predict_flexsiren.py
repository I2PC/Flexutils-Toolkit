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
from importlib.metadata import version

if version("tensorflow") >= "2.16.0":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorboard.plugins import projector

from xmipp_metadata.metadata import XmippMetaData


# from tensorflow_toolkit.datasets.dataset_template import sequence_to_data_pipeline, create_dataset


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def predict(md_file, weigths_file, latDim, refinePose, architecture, ctfType, pad=2,
            sr=1.0, applyCTF=1, poseReg=0.0, ctfReg=0.0):

    # We need to import network and generators here instead of at the beginning of the script to allow Tensorflow
    # get the right GPUs set in CUDA_VISIBLE_DEVICES
    from tensorflow_toolkit.generators.generator_flexsiren import Generator
    from tensorflow_toolkit.networks.flexsiren import AutoEncoder

    # Create data generator
    generator = Generator(md_file=md_file, shuffle=False, batch_size=32,
                          step=1, splitTrain=1.0, refinePose=refinePose, pad_factor=pad,
                          sr=sr, applyCTF=applyCTF)

    # Tensorflow data pipeline
    # generator_dataset, generator = sequence_to_data_pipeline(generator)
    # dataset = create_dataset(generator_dataset, generator, shuffle=False, batch_size=32)

    # Load model
    autoencoder = AutoEncoder(generator, latDim=latDim, architecture=architecture, CTF=ctfType,
                              poseReg=poseReg, ctfReg=ctfReg)
    _ = autoencoder(next(iter(generator.return_tf_dataset()))[0])
    autoencoder.load_weights(weigths_file)

    # Get Zernike3DSpace
    # zernike_space = []
    # delta_euler = []
    # delta_shifts = []

    # Metadata
    metadata = XmippMetaData(md_file)

    # Predict step
    print("------------------ Predicting Zernike3D coefficients... ------------------")
    encoded, c_lnm = autoencoder.predict(generator.return_tf_dataset())

    # Get encoded data in right format
    z_space = encoded[0]

    if refinePose:
        delta_euler = encoded[1]
        delta_shifts = encoded[2]

    # Tensorboard projector
    log_dir = os.path.join(os.path.dirname(md_file), "network", "logs")
    if os.path.isdir(log_dir):
        zernike_space_norm = z_space / np.amax(np.linalg.norm(z_space, axis=1))
        weights = tf.Variable(zernike_space_norm, name="z_space")
        checkpoint = tf.train.Checkpoint(zernike_space=weights)
        checkpoint.save(os.path.join(log_dir, "z_space.ckpt"))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = os.path.join("z_space", ".ATTRIBUTES", "VARIABLE_VALUE")
        projector.visualize_embeddings(log_dir, config)

    # Save space to metadata file
    metadata[:, 'zCoefficients'] = np.asarray([",".join(item) for item in z_space.astype(str)])
    metadata[:, 'bCoefficients'] = np.asarray([",".join(item) for item in c_lnm.astype(str)])

    if refinePose:
        delta_euler = np.vstack(delta_euler)
        delta_shifts = np.vstack(delta_shifts)

        metadata[:, 'delta_angle_rot'] = delta_euler[:, 0]
        metadata[:, 'delta_angle_tilt'] = delta_euler[:, 1]
        metadata[:, 'delta_angle_psi'] = delta_euler[:, 2]
        metadata[:, 'delta_shift_x'] = delta_shifts[:, 0]
        metadata[:, 'delta_shift_y'] = delta_shifts[:, 1]

    metadata.write(md_file, overwrite=True)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--weigths_file', type=str, required=True)
    parser.add_argument('--lat_dim', type=int, required=True)
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--pose_reg', type=float, required=False, default=0.0)
    parser.add_argument('--ctf_reg', type=float, required=False, default=0.0)
    parser.add_argument('--apply_ctf', type=int, required=True)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"md_file": args.md_file, "weigths_file": args.weigths_file,
              "latDim": args.lat_dim, "refinePose": args.refine_pose,
              "architecture": args.architecture, "ctfType": args.ctf_type,
              "pad": args.pad, "sr": args.sr, "applyCTF": args.apply_ctf,
              "poseReg": args.pose_reg, "ctfReg": args.ctf_reg}

    # Initialize volume slicer
    predict(**inputs)

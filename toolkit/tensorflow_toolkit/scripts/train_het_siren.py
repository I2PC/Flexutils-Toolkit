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

import tensorflow as tf

from toolkit.tensorflow_toolkit.generators.generator_het_siren import Generator
from toolkit.tensorflow_toolkit.networks.het_siren import AutoEncoder


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def train(outPath, md_file, batch_size, shuffle, step, splitTrain, epochs, cost,
          radius_mask, smooth_mask, refinePose, architecture="convnn", weigths_file=None,
          ctfType="apply", pad=2, sr=1.0, applyCTF=1, hetDim=10, l1Reg=0.5):

    # Create data generator
    generator = Generator(md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                          step=step, splitTrain=splitTrain, cost=cost, radius_mask=radius_mask,
                          smooth_mask=smooth_mask, pad_factor=pad, sr=sr,
                          applyCTF=applyCTF)

    # Train model
    autoencoder = AutoEncoder(generator, architecture=architecture, CTF=ctfType, refPose=refinePose,
                              het_dim=hetDim, l1_lambda=l1Reg)

    # Fine tune a previous model
    if weigths_file:
        autoencoder.load_weights(weigths_file)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    autoencoder.compile(optimizer=optimizer)
    autoencoder.fit(generator, epochs=epochs)

    # Save model
    autoencoder.save_weights(os.path.join(outPath, "het_siren_model"))


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--cost', type=str, required=True)
    parser.add_argument('--l1_reg', type=float, required=True)
    parser.add_argument('--het_dim', type=int, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--radius_mask', type=float, required=False, default=2)
    parser.add_argument('--smooth_mask', action='store_true')
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--weigths_file', type=str, required=False, default=None)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--apply_ctf', type=int, required=True)
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"md_file": args.md_file, "outPath": args.out_path,
              "batch_size": args.batch_size, "shuffle": args.shuffle,
              "step": args.step, "splitTrain": args.split_train, "epochs": args.epochs,
              "cost": args.cost, "radius_mask": args.radius_mask, "smooth_mask": args.smooth_mask,
              "refinePose": args.refine_pose, "architecture": args.architecture,
              "weigths_file": args.weigths_file, "ctfType": args.ctf_type, "pad": args.pad,
              "sr": args.sr, "applyCTF": args.apply_ctf, "hetDim": args.het_dim,
              "l1Reg": args.l1_reg}

    # Initialize volume slicer
    train(**inputs)
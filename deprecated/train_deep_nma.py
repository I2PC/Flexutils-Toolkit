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
import glob
import shutil
from xmipp_metadata.metadata import XmippMetaData

import tensorflow as tf

from deprecated.generator_deep_nma import Generator
from deprecated.deep_nma import AutoEncoder
# from tensorflow_toolkit.datasets.dataset_template import sequence_to_data_pipeline, create_dataset
from tensorflow_toolkit.utils import epochs_from_iterations


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def train(outPath, md_file, n_modes, batch_size, shuffle, splitTrain, epochs, cost,
          radius_mask, smooth_mask, refinePose, architecture="convnn", ctfType="apply", pad=2,
          sr=1.0, applyCTF=1, lr=1e-5, multires=[2, 4, 8], angReg=0.01, bondReg=0.01):

    try:
        # Create data generator
        generator = Generator(n_modes=n_modes, md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                              step=1, splitTrain=splitTrain, cost=cost, radius_mask=radius_mask,
                              smooth_mask=smooth_mask, refinePose=refinePose, pad_factor=pad,
                              sr=sr, applyCTF=applyCTF)

        # Create validation generator
        if splitTrain < 1.0:
            generator_val = Generator(n_modes=n_modes, md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                                      step=1, splitTrain=(splitTrain - 1.0), cost=cost, radius_mask=radius_mask,
                                      smooth_mask=smooth_mask, refinePose=refinePose, pad_factor=pad,
                                      sr=sr, applyCTF=applyCTF)
        else:
            generator_val = None

        # Tensorflow data pipeline
        # generator_dataset, generator = sequence_to_data_pipeline(generator)
        # dataset = create_dataset(generator_dataset, generator, batch_size=batch_size)

        # Train model
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        autoencoder = AutoEncoder(generator, architecture=architecture, CTF=ctfType, multires=multires,
                                  l_angle=angReg, l_bond=bondReg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Create a callback that saves the model's weights
        initial_epoch = 0
        checkpoint_path = os.path.join(outPath, "training", "cp-{epoch:04d}.hdf5")
        if not os.path.isdir(os.path.dirname(checkpoint_path)):
            os.mkdir(os.path.dirname(checkpoint_path))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        # Tensorboard callback
        log_dir = os.path.join(outPath, "logs")
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                              write_graph=True, write_steps_per_second=True)

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

        autoencoder.compile(optimizer=optimizer, jit_compile=True)

        if generator_val is not None:
            autoencoder.fit(generator, validation_data=generator_val, epochs=epochs, validation_freq=2,
                            callbacks=[cp_callback, tensorboard_callback], initial_epoch=initial_epoch)
        else:
            autoencoder.fit(generator, epochs=epochs,
                            callbacks=[cp_callback, tensorboard_callback], initial_epoch=initial_epoch)
    except tf.errors.ResourceExhaustedError as error:
        msg = "GPU memory has been exhausted. Usually this can be solved by " \
              "downsampling further your particles or by decreasing the batch size. " \
              "Please, modify any of these two options in the form and try again."
        print(msg)
        raise error

    # Save model
    autoencoder.save_weights(os.path.join(outPath, "deep_nma_model.h5"))

    # Remove checkpoints
    shutil.rmtree(checkpoint)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--n_modes', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--max_samples_seen', type=int, required=False)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--cost', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--radius_mask', type=float, required=False, default=2)
    parser.add_argument('--smooth_mask', action='store_true')
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--apply_ctf', type=int, required=True)
    parser.add_argument('--multires', type=str, required=True)
    parser.add_argument('--angle_reg', type=float, required=True)
    parser.add_argument('--bond_reg', type=float, required=True)
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

    inputs = {"md_file": args.md_file, "outPath": args.out_path, "n_modes": args.n_modes,
              "batch_size": args.batch_size, "shuffle": args.shuffle,
              "splitTrain": args.split_train, "epochs": epochs,
              "cost": args.cost, "radius_mask": args.radius_mask, "smooth_mask": args.smooth_mask,
              "refinePose": args.refine_pose, "architecture": args.architecture,
              "ctfType": args.ctf_type, "pad": args.pad, "sr": args.sr,
              "applyCTF": args.apply_ctf, "lr": args.lr, "multires": multires,
              "angReg": args.angle_reg, "bondReg": args.bond_reg}

    # Initialize volume slicer
    train(**inputs)

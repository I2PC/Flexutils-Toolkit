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
import shutil
import glob
from importlib.metadata import version
from xmipp_metadata.metadata import XmippMetaData

if version("tensorflow") >= "2.16.0":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras import mixed_precision

# from tensorflow_toolkit.datasets.dataset_template import sequence_to_data_pipeline, create_dataset
from tensorflow_toolkit.utils import epochs_from_iterations

# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def train(outPath, md_file, latDim, batch_size, shuffle, step, splitTrain, epochs, cost,
          radius_mask, smooth_mask, refinePose, architecture="convnn", ctfType="apply", pad=2,
          sr=1.0, applyCTF=1, lr=1e-5, jit_compile=True, regNorm=1e-4, regBond=0.01, regAngle=0.01, regClashes=None,
          tensorboard=True, weigths_file=None, precision="mixed_float16"):

    # We need to import network and generators here instead of at the beginning of the script to allow Tensorflow
    # get the right GPUs set in CUDA_VISIBLE_DEVICES
    assert precision in ["float32", "mixed_float16"]
    mixed_precision.set_global_policy(precision)
    precision = tf.float32 if precision == "float32" else tf.float16
    from tensorflow_toolkit.generators.generator_flexsiren import Generator
    from tensorflow_toolkit.networks.flexsiren import AutoEncoder

    try:
        # Create data generator
        generator = Generator(md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                              step=step, splitTrain=splitTrain, cost=cost, radius_mask=radius_mask,
                              smooth_mask=smooth_mask, refinePose=refinePose, pad_factor=pad,
                              sr=sr, applyCTF=applyCTF, precision=precision)

        # Create validation generator
        if splitTrain < 1.0:
            generator_val = Generator(md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                                      step=step, splitTrain=(splitTrain - 1.0), cost=cost, radius_mask=radius_mask,
                                      smooth_mask=smooth_mask, refinePose=refinePose, pad_factor=pad,
                                      sr=sr, applyCTF=applyCTF, precision=precision)
        else:
            generator_val = None

        # Tensorflow data pipeline
        # generator_dataset, generator = sequence_to_data_pipeline(generator)
        # dataset = create_dataset(generator_dataset, generator, batch_size=batch_size)

        # Train model
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        if regClashes is not None:
            autoencoder = AutoEncoder(generator, latDim=latDim, architecture=architecture, CTF=ctfType, l_bond=regBond,
                                      l_angle=regAngle, l_clashes=regClashes, jit_compile=jit_compile, precision=precision)
            jit_compile = False
        else:
            autoencoder = AutoEncoder(generator, latDim=latDim, architecture=architecture, CTF=ctfType, l_bond=regBond,
                                      l_angle=regAngle, l_clashes=regClashes, jit_compile=False, precision=precision)

        # Fine tune a previous model
        if weigths_file:
            if generator.mode == "spa":
                autoencoder.build(input_shape=(None, generator.xsize, generator.xsize, 1))
            elif generator.mode == "tomo":
                autoencoder.build(input_shape=[(None, generator.xsize, generator.xsize, 1),
                                               [None, generator.sinusoid_table.shape[1]]])
            autoencoder.load_weights(weigths_file)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        # Callbacks list
        callbacks = []

        # Create a callback that saves the model's weights
        initial_epoch = 0
        checkpoint_path = os.path.join(outPath, "training", "cp-{epoch:04d}.hdf5")
        if not os.path.isdir(os.path.dirname(checkpoint_path)):
            os.mkdir(os.path.dirname(checkpoint_path))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        callbacks.append(cp_callback)

        # Callbacks list
        if tensorboard:
            # Tensorboard callback
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
                if generator.mode == "spa":
                    autoencoder.build(input_shape=(None, autoencoder.xsize, autoencoder.xsize, 1))
                elif generator.mode == "tomo":
                    autoencoder.build(input_shape=[(None, autoencoder.xsize, autoencoder.xsize, 1),
                                                   [None, generator.sinusoid_table.shape[1]]])
                autoencoder.load_weights(latest)
                latest = os.path.basename(latest)
                initial_epoch = int(re.findall(r'\d+', latest)[0]) - 1

        autoencoder.compile(optimizer=optimizer, jit_compile=jit_compile)

        if generator_val is not None:
            autoencoder.fit(generator, validation_data=generator_val, epochs=epochs, validation_freq=2,
                            callbacks=callbacks, initial_epoch=initial_epoch)
        else:
            autoencoder.fit(generator, epochs=epochs,
                            callbacks=callbacks, initial_epoch=initial_epoch)
    except tf.errors.ResourceExhaustedError as error:
        msg = "GPU memory has been exhausted. Usually this can be solved by " \
              "downsampling further your particles or by decreasing the batch size. " \
              "Please, modify any of these two options in the form and try again."
        print(msg)
        raise error

    # Save model
    autoencoder.save_weights(os.path.join(outPath, "flexsiren_model.h5"))

    # Remove checkpoints
    shutil.rmtree(checkpoint)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--lat_dim', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--max_samples_seen', type=int, required=False)
    parser.add_argument('--cost', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--radius_mask', type=float, required=False, default=2)
    parser.add_argument('--smooth_mask', action='store_true')
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--weigths_file', type=str, required=False, default=None)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--apply_ctf', type=int, required=True)
    parser.add_argument('--jit_compile', action='store_true')
    parser.add_argument('--regNorm', type=float, default=0.0001)
    parser.add_argument('--regBond', type=float, default=0.01)
    parser.add_argument('--regAngle', type=float, default=0.01)
    parser.add_argument('--regClashes', type=float, default=None)
    parser.add_argument('--tensorboard', action='store_true')
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

    inputs = {"md_file": args.md_file, "outPath": args.out_path, "latDim": args.lat_dim,
              "batch_size": args.batch_size, "shuffle": args.shuffle,
              "step": args.step, "splitTrain": args.split_train, "epochs": epochs,
              "cost": args.cost, "radius_mask": args.radius_mask, "smooth_mask": args.smooth_mask,
              "refinePose": args.refine_pose, "architecture": args.architecture,
              "ctfType": args.ctf_type, "pad": args.pad, "sr": args.sr,
              "applyCTF": args.apply_ctf, "lr": args.lr, "jit_compile": args.jit_compile,
              "regNorm": args.regNorm, "regBond": args.regBond, "regAngle": args.regAngle,
              "regClashes": args.regClashes, "tensorboard": args.tensorboard, "weigths_file": args.weigths_file}

    # Initialize volume slicer
    train(**inputs)

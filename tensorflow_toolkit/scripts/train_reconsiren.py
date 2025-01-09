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
import numpy as np
from math import ceil
from importlib.metadata import version

from sklearn.cluster import KMeans
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

os.environ["TF_USE_LEGACY_KERAS"] = "0"
# if version("tensorflow") >= "2.16.0":
#     os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
# from tensorflow.keras import mixed_precision

from tensorflow_toolkit.utils import epochs_from_iterations, xmippEulerFromMatrix


def train(outPath, md_file, batch_size, shuffle, splitTrain, epochs, only_pose=False, n_candidates=6,
          architecture="convnn", weigths_file=None, ctfType=None, pad=4, sr=1.0, applyCTF=0, l1Reg=0.5,
          tvReg=0.1, mseReg=0.1, udLambda=0.000001, unLambda=0.0001, only_pos=False, useHet=False,
          jit_compile=True, tensorboard=True):
    # We need to import network and generators here instead of at the beginning of the script to allow Tensorflow
    # get the right GPUs set in CUDA_VISIBLE_DEVICES
    from tensorflow_toolkit.generators.generator_reconsiren import Generator
    from tensorflow_toolkit.networks.reconsiren import AutoEncoder

    try:
        # Create data generator
        generator = Generator(md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                              step=1, splitTrain=splitTrain, cost="mse", pad_factor=pad, sr=sr,
                              applyCTF=0)
        generator_pred = Generator(md_file=md_file, shuffle=False, batch_size=batch_size,
                                   step=1, splitTrain=splitTrain, cost="mse", pad_factor=pad, sr=sr,
                                   applyCTF=0)

        # Create validation generator
        if splitTrain < 1.0:
            generator_val = Generator(md_file=md_file, shuffle=shuffle, batch_size=batch_size,
                                      step=1, splitTrain=splitTrain, cost="mse", pad_factor=pad, sr=sr,
                                      applyCTF=applyCTF)
        else:
            generator_val = None

        strategy = tf.distribute.get_strategy()  # Default strategy

        with strategy.scope():
            # Train model
            autoencoder = AutoEncoder(generator, architecture=architecture, CTF=None,
                                      l1_lambda=l1Reg, tv_lambda=tvReg, mse_lambda=mseReg, un_lambda=unLambda,
                                      ud_lambda=udLambda, only_pose=only_pose, n_candidates=n_candidates,
                                      only_pos=only_pos, multires=None, useHet=useHet)

            # Fine tune a previous model
            if weigths_file:
                _ = autoencoder(next(iter(generator.return_tf_dataset()))[0])
                autoencoder.load_weights(weigths_file)

            if only_pose:
                optimizer_encoder = [tf.keras.optimizers.RMSprop(learning_rate=1e-5),
                                     tf.keras.optimizers.Adam(learning_rate=1e-5)]
                optimizer_decoder = tf.keras.optimizers.Adam(learning_rate=1e-5)
                optimizer_het = [tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                                 tf.keras.optimizers.Adam(learning_rate=1e-4)]
            else:
                if only_pos:
                    optimizer_encoder = [tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                                         tf.keras.optimizers.Adam(learning_rate=1e-5)]
                    optimizer_het = [tf.keras.optimizers.Adam(learning_rate=1e-4),
                                     tf.keras.optimizers.Adam(learning_rate=1e-4)]
                    optimizer_decoder = tf.keras.optimizers.Adam(learning_rate=1e-3)  # Classes
                else:
                    optimizer_encoder = [tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                                         tf.keras.optimizers.Adam(learning_rate=1e-5)]
                    optimizer_het = [tf.keras.optimizers.Adam(learning_rate=1e-4),
                                     tf.keras.optimizers.Adam(learning_rate=1e-4)]
                    optimizer_decoder = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Particles

            # Callbacks list
            callbacks = []

            # Create a callback that saves the model's weights
            initial_epoch = 0
            if tf.__version__ < "2.16.0" or os.environ["TF_USE_LEGACY_KERAS"] == "1":
                checkpoint_path = os.path.join(outPath, "training", "cp-{epoch:04d}.hdf5")
            else:
                checkpoint_path = os.path.join(outPath, "training", "cp-{epoch:04d}.weights.h5")
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
                    autoencoder.build(input_shape=(None, generator.xsize, generator.xsize, 1))
                    autoencoder.load_weights(latest)
                    latest = os.path.basename(latest)
                    initial_epoch = int(re.findall(r'\d+', latest)[0]) - 1

            autoencoder.compile(e_optimizer=optimizer_encoder, d_optimizer=optimizer_decoder, het_optimizer=optimizer_het,
                                jit_compile=jit_compile)

            steps = ceil(epochs / 5)
            md = XmippMetaData(md_file)

            train_dataset = generator.return_tf_dataset(preShuffle=True)
            predict_dataset = generator_pred.return_tf_dataset()
            if generator_val is not None:
                validation_dataset = generator_val.return_tf_dataset()

            for idx in range(steps):
                if generator_val is not None:
                    autoencoder.fit(train_dataset, validation_data=validation_dataset, epochs=5, validation_freq=2,
                                    callbacks=callbacks, initial_epoch=initial_epoch)
                else:
                    autoencoder.fit(train_dataset, epochs=5,
                                    initial_epoch=initial_epoch)  # Adding callback (checkpoint) fail?

                r, shifts, imgs, het, loss, loss_cons = autoencoder.predict(predict_dataset)

                # Rotation matrix to euler angles
                euler_angles = np.zeros((r.shape[0], 3))
                idx = 0
                for matrix in r:
                    euler_angles[idx] = xmippEulerFromMatrix(matrix)
                    idx += 1

                # Save metadata
                md[:, 'angleRot'] = euler_angles[:, 0]
                md[:, 'angleTilt'] = euler_angles[:, 1]
                md[:, 'anglePsi'] = euler_angles[:, 2]
                md[:, 'shiftX'] = shifts[:, 0]
                md[:, 'shiftY'] = shifts[:, 1]
                md[:, "reproj_cons_error"] = loss_cons

                # Save map
                if not autoencoder.only_pose:

                    decoded_map = autoencoder.eval_volume(filter=True, het=None)
                    decoded_path = os.path.join(outPath, f'decoded_map_no_het.mrc')
                    ImageHandler().write(decoded_map, decoded_path, overwrite=True)

                if useHet:
                    kmeans = KMeans(n_clusters=20).fit(het)
                    labels = kmeans.predict(het)
                    unique_labels = np.unique(labels)
                    centers = kmeans.cluster_centers_
                    decoded_maps = autoencoder.eval_volume(filter=True, het=centers)
                    idx = 0
                    for decoded_map in decoded_maps:
                        decoded_path = os.path.join(outPath, f'decoded_map_{unique_labels[idx]:02}.mrc')
                        ImageHandler().write(decoded_map, decoded_path, overwrite=True)
                        idx += 1
                    md[:, "cluster_labels"] = labels
                    md[:, "reproj_het_error"] = loss

                md.write(os.path.join(outPath, f'metadata_pred_angles.xmd'), overwrite=True)

    except tf.errors.ResourceExhaustedError as error:
        msg = "GPU memory has been exhausted. Usually this can be solved by " \
              "downsampling further your particles or by decreasing the batch size. " \
              "Please, modify any of these two options in the form and try again."
        print(msg)
        raise error

    # Save model
    if tf.__version__ < "2.16.0" or os.environ["TF_USE_LEGACY_KERAS"] == "1":
        autoencoder.save_weights(os.path.join(outPath, "reconsiren_model.h5"))
    else:
        autoencoder.save_weights(os.path.join(outPath, "reconsiren_model.weights.h5"))

    # Remove checkpoints
    shutil.rmtree(checkpoint)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    # parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--split_train', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--max_samples_seen', type=int, required=False)
    parser.add_argument('--l1_reg', type=float, required=True)
    parser.add_argument('--tv_reg', type=float, required=True)
    parser.add_argument('--mse_reg', type=float, required=True)
    parser.add_argument('--ud_lambda', type=float, required=True)
    parser.add_argument('--un_lambda', type=float, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    # parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--only_pose', action='store_true')
    parser.add_argument('--only_pos', action='store_true')
    parser.add_argument('--heterogeneous', action='store_true')
    parser.add_argument('--n_candidates', type=int, required=True)
    parser.add_argument('--weigths_file', type=str, required=False, default=None)
    parser.add_argument('--sr', type=float, required=True)
    # parser.add_argument('--apply_ctf', type=int, required=True)
    parser.add_argument('--jit_compile', action='store_true')
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

    inputs = {"md_file": args.md_file, "outPath": args.out_path,
              "batch_size": args.batch_size, "shuffle": args.shuffle,
              "splitTrain": args.split_train, "epochs": epochs,
              "architecture": args.architecture, "weigths_file": args.weigths_file, "ctfType": None,
              "pad": args.pad, "sr": args.sr, "applyCTF": 0,
              "l1Reg": args.l1_reg, "tvReg": args.tv_reg, "mseReg": args.mse_reg,
              "udLambda": args.ud_lambda, "unLambda": args.un_lambda,
              "jit_compile": args.jit_compile, "tensorboard": args.tensorboard,
              "only_pose": args.only_pose, "only_pos": args.only_pos, "n_candidates": args.n_candidates,
              "useHet": args.heterogeneous}

    # Initialize volume slicer
    train(**inputs)

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
from pathlib import Path

from sklearn.cluster import KMeans
from xmipp_metadata.image_handler import ImageHandler
from xmipp_metadata.metadata import XmippMetaData

os.environ["TF_USE_LEGACY_KERAS"] = "0"
import tensorflow as tf

from tensorflow_toolkit.generators.generator_reconsiren import Generator
from tensorflow_toolkit.networks.reconsiren import AutoEncoder
from tensorflow_toolkit.utils import xmippEulerFromMatrix


def predict(md_file, weigths_file, architecture, ctfType, pad=2, sr=1.0, n_candidates=6,
            applyCTF=1, filter=True, only_pose=False, only_pos=False, useHet=False):
    # Create data generator
    generator = Generator(md_file=md_file, shuffle=False, batch_size=32,
                          step=1, splitTrain=1.0, cost="mse", pad_factor=pad, sr=sr,
                          applyCTF=0)

    # Load model
    autoencoder = AutoEncoder(generator, architecture=architecture, CTF=None,
                              l1_lambda=0.0, tv_lambda=0.0, mse_lambda=0.0, un_lambda=0.0001,
                              ud_lambda=0.000001, only_pose=only_pose, n_candidates=n_candidates,
                              only_pos=only_pos, useHet=useHet)
    _ = autoencoder(next(iter(generator.return_tf_dataset()))[0])
    autoencoder.load_weights(weigths_file)

    # Metadata
    metadata = XmippMetaData(md_file)

    # Dataset
    predict_dataset = generator.return_tf_dataset()

    # Get poses
    print("------------------ Predicting angles and shifts... ------------------")
    r, shifts, imgs, het, loss, loss_cons = autoencoder.predict(predict_dataset)

    # Rotation matrix to euler angles
    euler_angles = np.zeros((r.shape[0], 3))
    idx = 0
    for matrix in r:
        euler_angles[idx] = xmippEulerFromMatrix(matrix)
        idx += 1

    # Get map
    print("------------------ Decoding volume... ------------------")
    decoded_map = autoencoder.eval_volume(filter=True)

    if useHet:
        kmeans = KMeans(n_clusters=20).fit(het)
        centers = kmeans.cluster_centers_
        labels = kmeans.predict(het)
        unique_labels = np.unique(labels)
        het_maps = autoencoder.eval_volume(filter=True, het=centers)

    # Save space to metadata file
    metadata[:, 'angleRot'] = euler_angles[:, 0]
    metadata[:, 'angleTilt'] = euler_angles[:, 1]
    metadata[:, 'anglePsi'] = euler_angles[:, 2]
    metadata[:, 'shiftX'] = shifts[:, 0]
    metadata[:, 'shiftY'] = shifts[:, 1]
    metadata[:, "reproj_cons_error"] = loss_cons

    # Save map
    decoded_path = Path(Path(md_file).parent, 'decoded_map.mrc')
    ImageHandler().write(decoded_map, decoded_path, overwrite=True)

    if useHet:
        idx = 0
        for het_map in het_maps:
            decoded_path = os.path.join(Path(md_file).parent, f'decoded_map_{unique_labels[idx]:02}.mrc')
            ImageHandler().write(het_map, decoded_path, overwrite=True)
            idx += 1
        metadata[:, "cluster_labels"] = labels
        metadata[:, "reproj_het_error"] = loss

    metadata.write(md_file, overwrite=True)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--weigths_file', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--apply_filter', action='store_true')
    parser.add_argument('--only_pose', action='store_true')
    parser.add_argument('--only_pos', action='store_true')
    parser.add_argument('--heterogeneous', action='store_true')
    parser.add_argument('--n_candidates', type=int, required=True)
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"md_file": args.md_file, "weigths_file": args.weigths_file,
              "architecture": args.architecture, "ctfType": None, "pad": args.pad, "sr": args.sr,
              "applyCTF": 0, "filter": args.apply_filter,
              "only_pose": args.only_pose, "only_pos": args.only_pos, "n_candidates": args.n_candidates,
              "useHet": args.heterogeneous}

    # Initialize volume slicer
    predict(**inputs)

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
from threadpoolctl import threadpool_limits, threadpool_info

import tensorflow as tf
from tensorboard.plugins import projector

from tensorflow_toolkit.generators.generator_het_siren import Generator
from tensorflow_toolkit.networks.het_siren import AutoEncoder
# from tensorflow_toolkit.datasets.dataset_template import sequence_to_data_pipeline, create_dataset


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def predict(md_file, weigths_file, refinePose, architecture, ctfType, pad=2, sr=1.0,
            applyCTF=1, filter=False, hetDim=10, numVol=20):
    # Create data generator
    generator = Generator(md_file=md_file, shuffle=False, batch_size=16,
                          step=1, splitTrain=1.0, pad_factor=pad, sr=sr,
                          applyCTF=applyCTF)

    # Tensorflow data pipeline
    # generator_dataset, generator = sequence_to_data_pipeline(generator)
    # dataset = create_dataset(generator_dataset, generator, shuffle=False, batch_size=16)

    # Load model
    autoencoder = AutoEncoder(generator, architecture=architecture, CTF=ctfType, refPose=refinePose,
                              het_dim=hetDim)
    autoencoder.build(input_shape=(None, generator.xsize, generator.xsize, 1))
    autoencoder.load_weights(weigths_file)

    # Get poses
    print("------------------ Predicting particles... ------------------")
    alignment, shifts, het = autoencoder.predict(generator, predict_mode="het")

    # Get map
    pool_info = threadpool_info()
    with threadpool_limits(limits=1, user_api=pool_info[1]["user_api"]):
        kmeans = KMeans(n_clusters=numVol).fit(het)
    centers = kmeans.cluster_centers_
    print("------------------ Decoding volume... ------------------")
    decoded_maps = autoencoder.eval_volume_het(centers, filter=filter, allCoords=True)

    # Tensorboard projector
    log_dir = os.path.join(os.path.dirname(md_file), "network", "logs")
    if os.path.isdir(log_dir):
        het_norm = het / np.amax(np.linalg.norm(het, axis=1))
        weights = tf.Variable(het_norm, name="het_space")
        checkpoint = tf.train.Checkpoint(het_space=weights)
        checkpoint.save(os.path.join(log_dir, "het_space.ckpt"))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = os.path.join("het_space", ".ATTRIBUTES", "VARIABLE_VALUE")
        projector.visualize_embeddings(log_dir, config)

    # Save space to metadata file
    alignment = np.vstack(alignment)
    shifts = np.vstack(shifts)
    het = np.vstack(het)

    generator.metadata[:, 'latent_space'] = np.asarray([",".join(item) for item in het.astype(str)])
    generator.metadata[:, 'delta_angle_rot'] = alignment[:, 0]
    generator.metadata[:, 'delta_angle_tilt'] = alignment[:, 1]
    generator.metadata[:, 'delta_angle_psi'] = alignment[:, 2]
    generator.metadata[:, 'delta_shift_x'] = shifts[:, 0]
    generator.metadata[:, 'delta_shift_y'] = shifts[:, 1]

    generator.metadata.write(md_file, overwrite=True)

    # Save map
    for idx, decoded_map in enumerate(decoded_maps):
        decoded_path = Path(Path(md_file).parent, 'decoded_map_class_%02d.mrc' % (idx + 1))
        ImageHandler().write(decoded_map, decoded_path, overwrite=True)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True)
    parser.add_argument('--weigths_file', type=str, required=True)
    parser.add_argument('--refine_pose', action='store_true')
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--het_dim', type=int, required=True)
    parser.add_argument('--ctf_type', type=str, required=True)
    parser.add_argument('--pad', type=int, required=False, default=2)
    parser.add_argument('--sr', type=float, required=True)
    parser.add_argument('--apply_ctf', type=int, required=True)
    parser.add_argument('--apply_filter', action='store_true')
    parser.add_argument('--num_vol', type=int, required=True)
    parser.add_argument('--gpu', type=str)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"md_file": args.md_file, "weigths_file": args.weigths_file,
              "refinePose": args.refine_pose, "architecture": args.architecture,
              "ctfType": args.ctf_type, "pad": args.pad, "sr": args.sr,
              "applyCTF": args.apply_ctf, "filter": args.apply_filter,
              "hetDim": args.het_dim, "numVol": args.num_vol}

    # Initialize volume slicer
    predict(**inputs)

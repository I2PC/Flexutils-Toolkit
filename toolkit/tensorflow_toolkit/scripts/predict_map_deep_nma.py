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
from scipy.ndimage import gaussian_filter

import tensorflow as tf

from toolkit.tensorflow_toolkit.generators.generator_deep_nma import Generator


# # os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu_instance in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu_instance, True)


def predict(weigths_file, nma_file, out_path, sr=1.0):
    c_nma = np.loadtxt(nma_file)
    if len(c_nma.shape) == 1:
        c_nma = c_nma.reshape((1, -1))
    md_file = Path(Path(weigths_file).parent.parent, "input_particles.xmd")
    basis_file = Path(Path(weigths_file).parent.parent, "nma_basis.anm.npz")

    # Create data generator
    generator = Generator(n_modes=c_nma.shape[1], md_file=md_file, shuffle=False, basis_file=basis_file)

    # Decode maps
    for idx in range(c_nma.shape[0]):
        # Get deformation field
        d_f = (generator.U @ c_nma[idx].T).reshape(-1, 3)

        # Get moved coords
        c_moved = generator.coords + d_f

        # Coords to indices
        c_moved /= sr
        c_moved[:, 0] += generator.xmipp_origin[0]
        c_moved[:, 1] += generator.xmipp_origin[1]
        c_moved[:, 2] += generator.xmipp_origin[2]
        c_moved = np.round(c_moved).astype(int)

        # Place values on grid
        volume = np.zeros((generator.xsize, generator.xsize, generator.xsize), dtype=np.float32)
        np.add.at(volume, (c_moved[:, 2], c_moved[:, 1], c_moved[:, 0]), generator.values)

        # Filter map
        volume = gaussian_filter(volume, sigma=1.)

        decoded_path = Path(out_path, 'decoded_map_class_%d.mrc' % (idx + 1))
        with mrcfile.new(decoded_path, overwrite=True) as mrc:
            mrc.set_data(volume)


if __name__ == '__main__':
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weigths_file', type=str, required=True)
    parser.add_argument('--nma_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--sr', type=float, required=True)

    args = parser.parse_args()

    if hasattr(args, "gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    inputs = {"weigths_file": args.weigths_file, "nma_file": args.nma_file,
              "out_path": args.out_path, "sr": args.sr}

    # Initialize volume slicer
    predict(**inputs)

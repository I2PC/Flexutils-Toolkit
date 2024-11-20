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
import subprocess
from tqdm import tqdm

import numpy as np
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

import tensorflow as tf

from tensorflow_toolkit.utils import epochs_from_iterations, computeBasis


def compute_distance_matrix(outPath, references_file, targets_file, L1, L2, batch_size, epochs, cost,
                            architecture="convnn", lr=1e-5, jit_compile=True, regNorm=1e-4, gpu=0, thr=8):

    # Load references and targers (MMap to save memory)
    references = np.load(references_file, mmap_mode="r")
    targets = np.load(targets_file, mmap_mode="r")

    # Find box sizes
    reference_bx = int(np.round(np.power(references.shape[-1], 1. / 3.)))
    target_bx = int(np.round(np.power(targets.shape[-1], 1. / 3.)))

    # Generate projection gallery
    ih = ImageHandler()
    md_file = os.path.join(outPath, "proj_metadata.xmd")
    proj_all, angles_all = [], []
    volIds = np.repeat(np.arange(targets.shape[0]), 20)
    if not os.path.isfile(md_file):
        for volume in tqdm(targets, desc="Projecting volumes: "):
            if volume.ndim == 2:
                volume = np.reshape(volume, (target_bx, target_bx, target_bx))
            volume = ih.scaleSplines(data=volume, finalDimension=64)
            proj, angles = ih.generateProjections(20, volume=volume,
                                                  n_jobs=thr, useFourier=True)
            proj_all.append(proj)
            angles_all.append(angles)
        proj_all = np.vstack(proj_all)
        angles_all = np.vstack(angles_all)
        ImageHandler().write(proj_all, os.path.join(outPath, "projections.mrcs"))
        md = XmippMetaData(os.path.join(outPath, "projections.mrcs"), angles=angles_all,
                           subtomo_labels=volIds)
    else:
        md = XmippMetaData(md_file)
        md.table.drop(columns="zernikeCoefficients", errors='ignore')

    # Call Zernike3Deep training script for all volumes
    dist_mat = np.zeros([references.shape[0], targets.shape[0]])
    idx = 0
    for volume in references:
        # Prepare data
        md.write(filename=md_file, overwrite=True)
        if volume.ndim == 2:
              volume = np.reshape(volume, (reference_bx, reference_bx, reference_bx))
        volume = ih.scaleSplines(data=volume, finalDimension=64)
        ImageHandler().write(volume, os.path.join(outPath, "volume.mrc"), overwrite=True)
        mask = ih.generateMask(inputFn=os.path.join(outPath, "volume.mrc"),
                               iterations=50, boxsize=64, smoothStairEdges=False)
        ImageHandler().write(mask, os.path.join(outPath, "mask.mrc"), overwrite=True)

        # Train and predict
        conda_base = subprocess.check_output("conda info --base", shell=True, text=True).strip()
        weights_file = os.path.join(outPath, "zernike3deep_model.h5")
        if jit_compile:
            subprocess.check_call(f'eval "$({conda_base}/bin/conda shell.bash hook)" && '
                                  f"conda activate flexutils-tensorflow && train_zernike3deep.py "
                                  f"--md_file {md_file} --out_path {outPath} --L1 {L1} --L2 {L2} "
                                  f"--batch_size {batch_size} --lr {lr} --epochs {epochs} "
                                  f"--architecture {architecture} --cost {cost} --jit_compile "
                                  f"--regNorm {regNorm} --apply_ctf 0 --shuffle --step 1 --split_train 1 "
                                  f"--ctf_type apply --sr 1.0 --pose_reg 0.0 --ctf_reg 0.0 --gpu {gpu}", shell=True)
        else:
            subprocess.check_call(f'eval "$({conda_base}/bin/conda shell.bash hook)" && '
                                  f"conda activate flexutils-tensorflow && train_zernike3deep.py "
                                  f"--md_file {md_file} --out_path {outPath} --L1 {L1} --L2 {L2} "
                                  f"--batch_size {batch_size} --lr {lr} --epochs {epochs} "
                                  f"--architecture {architecture} --cost {cost} "
                                  f"--regNorm {regNorm} --apply_ctf 0 --shuffle --step 1 --split_train 1 "
                                  f"--ctf_type apply --sr 1.0 --pose_reg 0.0 --ctf_reg 0.0 --gpu {gpu}", shell=True)
        subprocess.check_call(f'eval "$({conda_base}/bin/conda shell.bash hook)" && '
                              f"conda activate flexutils-tensorflow && predict_zernike3deep.py "
                              f"--md_file {md_file} --weigths_file {weights_file} --L1 {L1} "
                              f"--L2 {L2} --architecture {architecture} --pose_reg 0.0 --ctf_reg 0.0 "
                              f"--ctf_type apply --sr 1.0 --apply_ctf 0 --gpu {gpu}", shell=True)

        # Prepare data
        boxsize = volume.shape[0]
        r = 0.5 * boxsize
        coords = np.asarray(np.where(mask == 1))
        coords = np.transpose(np.asarray([coords[2, :], coords[1, :], coords[0, :]]))

        # Compute deformation distance
        Z = computeBasis(L1=int(L1), L2=int(L2), pos=coords - r, r=r)
        A = np.asarray([np.fromstring(item, sep=',') for item in
                        XmippMetaData(file_name=md_file)[:, 'zernikeCoefficients']])
        A = A[::20]
        size = int(A.shape[1] / 3)
        A = np.stack([A[:, :size], A[:, size:2 * size], A[:, 2 * size:]], axis=1)
        d_f = Z[None, ...] @ np.transpose(A, (0, 2, 1))
        dist_mat[idx, :] = np.mean(np.sqrt(np.mean(d_f * d_f, axis=-1)), axis=-1)
        idx += 1

    # Save resulting distance matrix
    np.save(os.path.join(outPath, "dist_mat.npy"), dist_mat)


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--references_file', type=str, required=True)
    parser.add_argument('--targets_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--L1', type=int, default=7)
    parser.add_argument('--L2', type=int,default=7)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--cost', type=str, default="corr")
    parser.add_argument('--architecture', type=str, default="mlpnn")
    parser.add_argument('--jit_compile', action='store_true')
    parser.add_argument('--regNorm', type=float, default=0.001)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--thr', type=int)

    args = parser.parse_args()

    inputs = {"references_file": args.references_file, "targets_file": args.targets_file,
              "outPath": args.out_path, "L1": args.L1,
              "L2": args.L2, "batch_size": args.batch_size, "epochs": args.epochs,
              "cost": args.cost, "architecture": args.architecture,
              "lr": args.lr, "jit_compile": args.jit_compile,
              "regNorm": args.regNorm, "gpu": args.gpu, "thr": args.thr}

    # Initialize volume slicer
    compute_distance_matrix(**inputs)

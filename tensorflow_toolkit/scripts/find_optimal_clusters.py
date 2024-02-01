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
import os.path

import matplotlib.pyplot as plt
import numpy as np
from inspect import getmembers, isfunction
import seaborn as sns
import pandas as pd

from tensorflow_toolkit.utils.Clustering.utils import getClusterMethod
import tensorflow_toolkit.utils.Clustering.methods as mth


def clusterAnalysis(dataFile, outPath, maxClusters=15, clusterMethod="KMeans"):
    # Load data
    flex_space = np.loadtxt(dataFile)

    # Clustering method choice
    fn_cluster = getClusterMethod(clusterMethod)

    # Exclude this methods
    exclude_list = [""]

    # Get dict of analysis methods
    methods = getmembers(mth, isfunction)
    methods = dict((name.replace("Analysis", ""), [fn, 0]) for name, fn in methods
                   if not name.replace("Analysis", "") in exclude_list and "Analysis" in name)

    # Clustering analysis
    for key in methods:
        methods[key][1] = methods[key][0](flex_space, fn_cluster, maxClusters=maxClusters, outPath=outPath)

    # Scatter plot of auto-clusters
    sns.set_theme()
    df = pd.DataFrame([(key, methods[key][1]) for key in methods], columns=["Method", "Best_K"])
    df = df.groupby(['Method', 'Best_K']).size().reset_index(name='Count')
    fig, ax = plt.subplots(figsize=(30, 5))
    sns.scatterplot(
        data=df, x="Method", y="Best_K", size='Count',
        sizes=(100, 200), legend="full", ax=ax
    )
    ax.grid(axis='y')
    sns.despine(left=True, bottom=True)
    plt.savefig(os.path.join(outPath, 'auto_clustering_results.png'), bbox_inches='tight')
    plt.close()

    # Save results
    df.to_csv(os.path.join(outPath, "auto_clustering_results.csv"))


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--max_clusters', type=int, required=True)
    parser.add_argument('--cluster_method', type=str, required=True)

    args = parser.parse_args()

    inputs = {"dataFile": args.data_file, "outPath": args.out_path,
              "maxClusters": args.max_clusters, "clusterMethod": args.cluster_method}

    # Initialize volume slicer
    clusterAnalysis(**inputs)

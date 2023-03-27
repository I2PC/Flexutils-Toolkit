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
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score


def gapStatisticAnalysis(data, clust_model, outPath="", nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic
    Params:
        data: ndarry of shape (n_samples, n_features)
        clust_model: Clustering method to be used
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    print("-------------- Running Gap statistic analysis... --------------")
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(tqdm(range(1, maxClusters))):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = clust_model(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = clust_model(k)
        km.fit(data)

        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = pd.concat([resultsdf, pd.DataFrame({'clusterCount': [k], 'gap': [gap]})],
                              ignore_index=True)

    print("-------------- ...Done --------------")

    # Best k
    best_k = gaps.argmax() + 1

    # Save gap statistic plot
    plt.plot(resultsdf['clusterCount'], resultsdf['gap'], linestyle='-', marker='o', color='b')
    plt.axvline(x=best_k, color='k', linestyle='--')
    plt.xlabel('K')
    plt.ylabel('Gap Statistic')
    plt.savefig(os.path.join(outPath, 'gap_statistic.png'), bbox_inches='tight')
    plt.close()

    return best_k


def elbowAnalysis(data, clust_model, outPath="", maxClusters=15):
    print("-------------- Running Elbow analysis... --------------")

    visualizer = KElbowVisualizer(clust_model(), k=(1, maxClusters), timings=True)
    visualizer.fit(data)

    print("-------------- ...Done --------------")

    # Save elbow plot
    plt.savefig(os.path.join(outPath, 'elbow.png'), bbox_inches='tight')
    plt.close()

    return visualizer.elbow_value_


def silhouetteAnalysis(data, clust_model, outPath="", maxClusters=15):
    print("-------------- Running Silhouette analysis... --------------")

    visualizer = KElbowVisualizer(clust_model(), k=(2, maxClusters), timings=True, metric='silhouette')
    visualizer.fit(data)

    print("-------------- ...Done --------------")

    # Save silhouette plot
    plt.savefig(os.path.join(outPath, 'silhouette.png'), bbox_inches='tight')
    plt.close()

    return visualizer.elbow_value_


def chAnalysis(data, clust_model, outPath="", maxClusters=15):
    print("-------------- Running Calinski Harabasz analysis... --------------")

    visualizer = KElbowVisualizer(clust_model(), k=(2, maxClusters), timings=True,
                                  metric='calinski_harabasz')
    visualizer.fit(data)

    print("-------------- ...Done --------------")

    # Save silhouette plot
    plt.savefig(os.path.join(outPath, 'calinski_harabasz.png'), bbox_inches='tight')
    plt.close()

    return visualizer.elbow_value_


def dbAnalysis(data, clust_model, outPath="", maxClusters=15):

    def get_kmeans_score(data, clust_model, center):
        '''
        returns the kmeans score regarding Davies Bouldin for points to centers
        INPUT:
            data - the dataset you want to fit kmeans to
            clust_model - Clustering method to be used
            center - the number of centers you want (the k value)
        OUTPUT:
            score - the Davies Bouldin score for the kmeans model fit to the data
        '''
        # instantiate kmeans
        kmeans = clust_model(n_clusters=center)
        # Then fit the model to your data using the fit method
        model = kmeans.fit_predict(data)

        # Calculate Davies Bouldin score
        score = davies_bouldin_score(data, model)

        return score

    # Analysis loop
    print("-------------- Running Davies Bouldin analysis... --------------")

    scores = []
    centers = list(range(2, maxClusters))
    for center in centers:
        scores.append(get_kmeans_score(data, clust_model, center))

    print("-------------- ...Done --------------")

    # Best k
    best_k = centers[scores.index(max(scores))]

    # Save Davies Bouldin plot
    plt.plot(centers, scores, linestyle='-', marker='o', color='b')
    plt.axvline(x=best_k, color='k', linestyle='--')
    plt.xlabel('K')
    plt.ylabel('Davies Bouldin score')
    plt.title('Davies Bouldin score vs. K')
    plt.savefig(os.path.join(outPath, 'davies_bouldin.png'), bbox_inches='tight')
    plt.close()

    return best_k

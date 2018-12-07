import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

import matplotlib.patches as mpatches
from thex_model.data_maps import code_cat


def plot_cluster_evals(cluster_classes, plot_title=None):
    """
    Plots cluster evaluation in horizontal bar plot. Each bar (y-axis) is a cluster with color corresponding to dominant transient class. The percent of that class captured by the cluster is the value of the bar (x-axis). 
    :param cluster_classes: Mapping of cluster numbers to class dominance info. Comes from clustering_performance.evaluate_clusters
    """
    rcParams['figure.figsize'] = 6, 10
    default_fontsize = 12
    cluster_indices = np.arange(len(cluster_classes))

    accuracies = []
    dominant_classes = []
    for key in cluster_classes.keys():
        # % of total class captured by this cluster
        perc_captured = cluster_classes[key][1]
        accuracies.append(perc_captured)
        dominant_classes.append(cluster_classes[key][0])

    # Color bars based on dominant transient class
    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dominant_classes))

    barlist = plt.barh(y=cluster_indices, width=accuracies,
                       height=1, color=colors, label=dominant_classes)

    plt.yticks(cluster_indices, cluster_classes.keys(), fontsize=default_fontsize)
    plt.xlabel('% of Total Class Captured by Cluster', fontsize=default_fontsize)
    plt.ylabel('Cluster', fontsize=default_fontsize)
    title = plot_title if plot_title else "Transient Class Dominance \n and Completeness in Clusters"
    plt.title(title, fontsize=16)

    # create legend
    unique_patches = []
    unique_classes = set()
    for index, c in enumerate(list(colors)):
        dominant_class = dominant_classes[index]
        if dominant_class not in unique_classes:
            unique_classes.add(dominant_class)
            patch = mpatches.Patch(color=c, label=code_cat[dominant_class])
            unique_patches.append(patch)
    plt.legend(handles=unique_patches, loc='best')

    plt.show()

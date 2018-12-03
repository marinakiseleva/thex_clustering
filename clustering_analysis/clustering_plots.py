import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

import matplotlib.patches as mpatches
from thex_model.data_maps import code_cat


def plot_cluster_evals(cluster_classes):
    """
    Plots cluster evaluation in horizontal bar plot. Each bar (y-axis) is a cluster, with the maximum frequency transient class noted, and the % of that class captured by the cluster being the value of the bar (x-axis). 
    """
    rcParams['figure.figsize'] = 10, 10
    default_fontsize = 12
    print(cluster_classes)

    # num_classes = len(cluster_classes)
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
    plt.title("Transient Class Dominance and Completeness in Clusters", fontsize=16)

    # Legend: create legend
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from thex_model.data_maps import code_cat


def plot_kmeans(kmeans, reduced_data):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.show()


def plot_2d_evals(embedding, data, unique_classes):
    """
    Plots 2d embedding of data, with different markers for different classes
    Can handle up to 36 classes
    """
    # Marker styles and colors
    unique_markers = [".", "+", "v", "s", "P", "p",
                      "*", "d", "^", "<", ">", "1", "2", "3", "4", "h", "H", "o"]
    mcount = len(unique_markers) - 1
    colors = ["red", "blue", "green"]

    # maps transient type to unique marker/color combo
    ttype_marker = {}
    markers = []  # for legend
    for index, ttype_code in enumerate(data['transient_type'].unique()):
        marker_type = unique_markers[index]
        marker_color = colors[index % 3]
        ttype_marker[ttype_code] = [marker_type, marker_color]
        # Keep track of markers for legend
        p = mlines.Line2D([], [], color=marker_color, marker=marker_type,
                          linestyle='', markersize=10, label=code_cat[ttype_code])
        markers.append(p)

    rcParams['figure.figsize'] = 6, 6
    x = embedding[:, 0]
    y = embedding[:, 1]
    for index, ttype_code in data['transient_type'].iteritems():
        marker_type, marker_color = ttype_marker[ttype_code]
        plt.scatter(x[index], y[index], marker=marker_type, color=marker_color)

    plt.legend(handles=markers, loc='best')

    plt.show()


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

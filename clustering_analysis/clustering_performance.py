import math
import numpy as np
from thex_data import data_plot
from thex_data.data_consts import code_cat, cat_code, TARGET_LABEL
# from clustering_plots import plot_cluster_evals


def get_max_cluster_class(cluster_num, cluster_map, data):
    """
    Returns most frequent class, and corresponding % of the total that this class is in this cluster (cluster_num)
    :param cluster_num: Particular cluster number to filter on
    :param cluster_map: Entire map of clusters
    :param data: Corresponding data that was trained on 
    """

    cluster_data = cluster_map.loc[cluster_map.cluster == cluster_num]
    indices = list(cluster_data.data_index)
    ttype_freq = {}  # Map transient class code to frequency in cluster
    for i in indices:
        cur_ttype = data.loc[i][TARGET_LABEL]
        if cur_ttype in ttype_freq:
            ttype_freq[cur_ttype] += 1
        else:
            ttype_freq[cur_ttype] = 1
    max_class = max(ttype_freq, key=ttype_freq.get)
    max_count = ttype_freq[max_class]
    return max_class, max_count, ttype_freq


def evaluate_clusters(k, cluster_map, data):
    """
    Get most frequent class per cluster, and how much of the total class is in that cluster. Clusters with different dominant classes & high percentage of the total class signify potential special class patterns.
    :param k: Number of clusters
    :param cluster_map: dataframe of data_index and cluster number
    :param data: corresponding dataframe of data
    :return cluster_classes: Map of clusters (numbers) to the value, percentage, and count of the most frequent class
    """
    # get map of transient classes to frequency
    map_counts = data_plot.map_counts_types(data)
    cluster_classes = {}
    # Iterate through each cluster
    for cluster_num in range(0, k):
        class_num, freq, ttype_freq = get_max_cluster_class(
            cluster_num, cluster_map, data)
        class_total = map_counts[class_num]
        perc_class = freq / class_total
        cluster_classes[cluster_num] = [int(class_num), round(perc_class, 4), freq]

    return cluster_classes

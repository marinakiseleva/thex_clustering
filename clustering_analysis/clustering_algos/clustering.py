import numpy as np
import pandas as pd


def get_num_clusters(cluster_map):
    return cluster_map['cluster'].max()


def get_cluster_map(clustering_algo, data):
    cluster_map = pd.DataFrame()
    if type(data) == np.ndarray:
        data = pd.DataFrame(data)

    cluster_map['data_index'] = data.index.values
    cluster_map['cluster'] = clustering_algo.labels_
    return cluster_map

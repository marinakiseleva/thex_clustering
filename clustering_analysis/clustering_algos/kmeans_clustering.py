from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def train_kmeans_clustering(k, data):
    """
    Run k-means with k clusters on train data
    """
    kmeans = KMeans(n_clusters=k, init='random', n_init=40, max_iter=500,
                    tol=1e-6, precompute_distances=True, random_state=10, copy_x=True).fit(data)
    return kmeans


def get_cluster_map(kmeans, data):
    cluster_map = pd.DataFrame()
    if type(data) == np.ndarray:
        data = pd.DataFrame(data)

    cluster_map['data_index'] = data.index.values
    cluster_map['cluster'] = kmeans.labels_
    return cluster_map


def test_kmeans_clustering(kmeans, data):
    """
    Tests kmeans algorithm with test data
    """
    return kmeans.predict(data)


def run_kmeans(k, train, test=None):
    """
    Runs k means with k clusters, using training and testing data. Returns map of indices in training data to the cluster that it was assigned.
    """
    print("Running k means with k = " + str(k))
    kmeans = train_kmeans_clustering(k, train)

    return get_cluster_map(kmeans, train)

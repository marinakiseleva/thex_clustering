from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def train_kmeans_clustering(k, data):
    """
    Run k-means with k clusters on train data
    """
    kmeans = KMeans(n_clusters=k).fit(data)
    return kmeans


def test_kmeans_clustering(kmeans, data):
    """
    Tests kmeans algorithm with test data
    """
    return kmeans.predict(data)


def run_kmeans(k, train, test=None):
    """
    Runs k means with k clusters, using training and testing data.
    """
    print("Running k means with k = " + str(k))
    kmeans = train_kmeans_clustering(k, train)
    # predictions = test_kmeans_clustering(kmeans, test)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = train.index.values
    cluster_map['cluster'] = kmeans.labels_
    return cluster_map

from sklearn.cluster import DBSCAN
from clustering_algos.clustering import get_cluster_map


def init_dbscan(eps=60, min_samples=3):
    """
    Run k-means with k clusters on train data
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

    return dbscan


def run_dbscan(dbscan, data):
    # dbscan = init_DBSCAN(data, eps=2, min_samples=3)
    dbscan.fit(data)
    return get_cluster_map(dbscan, data)

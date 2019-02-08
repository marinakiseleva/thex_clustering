import argparse

from thex_data import data_init
from thex_data import data_clean
from thex_data.data_prep import get_data
from thex_data import data_plot
from thex_data.data_consts import TARGET_LABEL

from model_performance.init_classifier import *

from clustering_algos.kmeans_clustering import train_kmeans_clustering
from clustering_algos.dbscan_clustering import init_dbscan
from clustering_algos.kmeans_clustering import run_kmeans
from clustering_algos.dbscan_clustering import run_dbscan
from clustering_performance import evaluate_clusters
from clustering_plots import plot_kmeans
from clustering_plots import plot_cluster_evals
from clustering_plots import plot_cluster_classes
from clustering_algos.clustering import get_num_clusters

from reduction_algos.tsne_reduction import run_tsne
from reduction_algos.umap_reduction import run_umap


def prep_data(cols, incl_redshift=False, debug=False):
    """
    Cleans and prepares data for analysis by normalizing transient classes into preset groups (by preset mapping). Returns training and testing data (with ttype in it.)

    """
    df = get_data(cols, incl_redshift).reset_index(drop=True)
    unique_classes = len(list(df[TARGET_LABEL].unique()))
    data_plot.plot_ttype_distribution(df)
    print("Unique classes: " + str(unique_classes))
    print("Training size " + str(df.shape[0]))
    return df, unique_classes


def run_analysis(data, unique_classes):
    """
    Runs each algorithm and plots results: KMeans alone, T-SNE, UMAP, and DBSCAN
    """
    train_data = data.drop([TARGET_LABEL], axis=1)
    # KMeans ###############################
    cluster_map = run_kmeans(k=unique_classes, train=train_data)
    cluster_classes = evaluate_clusters(unique_classes, cluster_map, data)

    # # TSNE ###############################
    embedding = run_tsne(data=train_data)
    reduced_cluster_map = run_kmeans(k=unique_classes, train=embedding)
    evaluate_clusters(unique_classes, reduced_cluster_map, data)

    # Plot 2D KMeans Clusters of t-SNE Reduced data
    kmeans = train_kmeans_clustering(unique_classes, embedding)
    plot_kmeans(kmeans, embedding, data, unique_classes)

    # UMAP ###############################
    embedding = run_umap(train_data)
    reduced_cluster_map = run_kmeans(k=unique_classes, train=embedding)
    evaluate_clusters(unique_classes, reduced_cluster_map, data)
    # Plot 2D KMeans Clusters of UMAP Reduced data
    kmeans = train_kmeans_clustering(unique_classes, embedding)
    plot_kmeans(kmeans, embedding, data, unique_classes)

    # DBSCAN ###############################
    dbscan = init_dbscan(eps=1.5, min_samples=10)
    cluster_map = run_dbscan(dbscan, embedding)
    cluster_classes = evaluate_clusters(get_num_clusters(cluster_map), cluster_map, data)
    plot_cluster_evals(cluster_classes, plot_title="DBSCAN Clustering")
    plot_cluster_classes(embedding, data, plot_title="DBSCAN Clustering")


def main():
    """
    Main runner of analysis. Expects data columns to filter input on. Pass in columns as space-delimited texts, like this:
    python run_analysis.py -cols PS1_gKmag PS1_rKmag PS1_iKmag PS1_zmag PS1_zKmag PS1_yKmag
    col_name:
        MPAJHU
        HyperLEDA
        NSA
        NED_GALEX
        NED_SDSS
        NED_IRASE
        AllWISE
        GalaxyZoo
    """
    col_list, incl_redshift = collect_args()
    train, unique_classes = prep_data(col_list, incl_redshift, True)

    if train.shape[0] == 0:
        print("No data to run on -- try adjusting columns or filters.")
        return -1

    run_analysis(train, unique_classes)


if __name__ == '__main__':
    main()

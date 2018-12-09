import argparse

from thex_model import data_init
from thex_model import data_clean
from thex_model import data_prep
from thex_model import data_plot

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


def prep_data(df, cols, debug=False):
    """
    Cleans and prepares data for analysis by normalizing transient classes into preset groups (by preset mapping). Returns training and testing data (with ttype in it.)

    """
    df = data_clean.group_cts(df)
    if debug:
        print("Grouped ct rows: " + str(df.shape[0]))

    df = data_prep.filter_columns(df, col_list=cols, incl_redshift=False)

    # Take difference between magnitude columns to get 'colors' from spectral samples
    df = data_prep.derive_diffs(df.copy())

    df.dropna(axis='rows', how='any', inplace=True)  # Drop rows with any NULL values

    # Randomly subsample any over-represented classes down to 100
    df = data_prep.sub_sample(df, count=100, col_val='transient_type')

    # Filter on top 5, most frequent classes (in order to reduce class imbalance)
    df = data_prep.filter_top_classes(df, top=5)

    df.reset_index(drop=True, inplace=True)

    unique_classes = len(list(df.transient_type.unique()))
    if debug:
        data_plot.plot_ttype_distribution(df)
        print("Unique classes: " + str(unique_classes))
        print("Training size " + str(df.shape[0]))

    return df, unique_classes


def run_analysis(data, unique_classes):
    """
    Runs each algorithm and plots results: KMeans alone, T-SNE, UMAP, and DBSCAN
    """
    train_data = data.drop(['transient_type'], axis=1)

    # KMeans ###############################
    # cluster_map = run_kmeans(k=unique_classes, train=train_data)
    # cluster_classes = evaluate_clusters(unique_classes, cluster_map, data)

    # # TSNE ###############################
    embedding = run_tsne(data=train_data)
    reduced_cluster_map = run_kmeans(k=unique_classes, train=embedding)
    evaluate_clusters(unique_classes, reduced_cluster_map, data)

    # Plot 2D KMeans Clusters of t-SNE Reduced data
    kmeans = train_kmeans_clustering(unique_classes, embedding)
    plot_kmeans(kmeans, embedding, data, unique_classes)

    # UMAP ###############################
    # embedding = run_umap(train_data)
    # reduced_cluster_map = run_kmeans(k=unique_classes, train=embedding)
    # evaluate_clusters(unique_classes, reduced_cluster_map, data)
    # # Plot 2D KMeans Clusters of UMAP Reduced data
    # kmeans = train_kmeans_clustering(unique_classes, embedding)
    # plot_kmeans(kmeans, embedding, data, unique_classes)

    # DBSCAN ###############################
    # dbscan = init_dbscan(eps=1.5, min_samples=10)
    # cluster_map = run_dbscan(dbscan, embedding)
    # cluster_classes = evaluate_clusters(get_num_clusters(cluster_map), cluster_map, data)
    # plot_cluster_evals(cluster_classes, plot_title="DBSCAN Clustering")
    # plot_cluster_classes(embedding, data, plot_title="DBSCAN Clustering")


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
    parser = argparse.ArgumentParser(description='Classify transients')
    parser.add_argument('-col_name', '--col_name', nargs='+',
                        help='Pass in string by which columns will be selected. For example: AllWISE will use all AlLWISE columns.', required=False)
    parser.add_argument('-cols', '--cols', nargs='+',
                        help='Pass in specific column names to filter on.', required=False)
    parser.add_argument('-d', '--debug', type=bool, nargs='?',
                        const=False, help='Boolean flag for debugging', required=False)

    args = parser.parse_args()
    df = data_init.collect_data("../data_input/THEx-catalog.v0_0_3.fits")

    # Initialize columns
    if args.col_name is None and args.cols is None:
        print("Either col_name or cols needs to be passed in. Exiting.")
        return -1
    col_list = []
    if args.col_name is not None:
        # Make list of column/feature names; exlcude _e_ (errors)
        col_list = [col for col in list(df) if any(
            col_val in col and "_e_" not in col for col_val in args.col_name)]
    elif args.cols is not None:
        col_list = args.cols

    if args.debug:
        print("Running on cols " + str(col_list))
        print("Init data pull rows: " + str(df.shape[0]))

    train, unique_classes = prep_data(df, col_list, args.debug)
    if train.shape[0] == 0:
        print("No data to run on -- try adjusting columns or filters.")
        return -1

    run_analysis(train, unique_classes)


if __name__ == '__main__':
    main()

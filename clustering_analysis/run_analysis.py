import argparse

from thex_model import data_init
from thex_model import data_clean
from thex_model import data_prep
from thex_model import data_plot

from clustering_algos.kmeans_clustering import run_kmeans
from clustering_performance import evaluate_clusters
from clustering_plots import plot_cluster_evals


def prep_data(df, cols, debug=False):
    """
    Cleans and prepares data for analysis by normalizing transient classes into preset groups (by preset mapping). Returns training and testing data (with ttype in it.)

    """
    df = data_clean.group_cts(df)
    if debug:
        print("Grouped ct rows: " + str(df.shape[0]))
    df = data_prep.filter_columns(df, col_list=cols, incl_redshift=False)

    df = data_prep.derive_diffs(df.copy())

    df.dropna(axis='rows', how='any', inplace=True)  # Drop rows with any NULL values
    df.reset_index(drop=True, inplace=True)
    df = data_prep.sub_sample(df, count=100, col_val='transient_type')

    df = data_prep.filter_top_classes(df, top=5)
    unique_classes = len(list(df.transient_type.unique()))
    if debug:
        data_plot.plot_ttype_distribution(df)
        print("Unique classes: " + str(unique_classes))
        print("Training size " + str(df.shape[0]))

    return df, unique_classes

    # train = df.sample(frac=0.7, random_state=200)
    # test = df.drop(train.index)
    # train.reset_index(drop=True, inplace=True)
    # test.reset_index(drop=True, inplace=True)
    # return train, test, unique_classes


def main():
    parser = argparse.ArgumentParser(description='Classify transients')
    # Pass in columns as space-delimited texts, like this:
    # PS1_gKmag PS1_rKmag PS1_iKmag PS1_zmag PS1_zKmag PS1_yKmag
    parser.add_argument('-cols', '--cols', nargs='+',
                        help='<Required> Set flag', required=True)
    parser.add_argument('-d', '--debug', type=bool, nargs='?',
                        const=False, help='Boolean for debugging')

    args = parser.parse_args()
    df = data_init.collect_data("../data_input/THEx-catalog.v0_0_3.fits")
    # print(list(df))
    print("Init data pull rows: " + str(df.shape[0]))
    train, unique_classes = prep_data(df, args.cols, args.debug)
    train_data = train.drop(['transient_type'], axis=1)

    cluster_map = run_kmeans(k=unique_classes, train=train_data)

    cluster_classes = evaluate_clusters(unique_classes, cluster_map, train)
    plot_cluster_evals(cluster_classes)


if __name__ == '__main__':
    main()

# 'AllWISE_W2mag', 'AllWISE_W2mag_Err', 'AllWISE_W3mag', 'AllWISE_W3mag_Err', 'AllWISE_W4mag', 'AllWISE_W4mag_Err', 'AllWISE_Jmag', 'AllWISE_Jmag_Err', 'AllWISE_Hmag', 'AllWISE_Hmag_Err', 'AllWISE_Kmag', 'AllWISE_Kmag_Err', 'AllWISE_IsExtSrc', 'AllWISE_IsVar',

 # 'Firefly_Chabrier_MILES_age_lightW', 'Firefly_Chabrier_MILES_metallicity_lightW', 'Firefly_Chabrier_MILES_stellar_mass', 'Firefly_Chabrier_MILES_spm_EBV', 'Firefly_Chabrier_MILES_nComponentsSSP', 'Firefly_Salpeter_MILES_age_lightW', 'Firefly_Salpeter_MILES_metallicity_lightW', 'Firefly_Salpeter_MILES_stellar_mass', 'Firefly_Salpeter_MILES_spm_EBV', 'Firefly_Salpeter_MILES_nComponentsSSP', 'Firefly_Kroupa_MILES_age_lightW', 'Firefly_Kroupa_MILES_metallicity_lightW', 'Firefly_Kroupa_MILES_stellar_mass', 'Firefly_Kroupa_MILES_spm_EBV', 'Firefly_Kroupa_MILES_nSSP', 'WiscPCA_MSTELLAR_MEDIAN', 'WiscPCA_MSTELLAR_ERR', 'WiscPCA_VDISP_MEDIAN', 'WiscPCA_VDISP_ERR', 'WiscPCA_CALPHA_NORM',

 #  'GalaxyZoo2_Ncl', 'GalaxyZoo2_Nvot', 'GalaxyZoo2_Smooth', 'GalaxyZoo2_FeatureOrDisk', 'GalaxyZoo2_StarOrArtifact', 'GalaxyZoo2_EdgeOn', 'GalaxyZoo2_NotEdgeOn', 'GalaxyZoo2_Bar', 'GalaxyZoo2_NoBar', 'GalaxyZoo2_Spiral', 'GalaxyZoo2_NoSpiral', 'GalaxyZoo2_NoBulge', 'GalaxyZoo2_JustNoticableBulge', 'GalaxyZoo2_ObviousBulge', 'GalaxyZoo2_DominantBulge', 'GalaxyZoo2_SomethingOdd', 'GalaxyZoo2_NothingOdd', 'GalaxyZoo2_SmoothAndCompletelyRound', 'GalaxyZoo2_SmoothAndInbetweenRoundness', 'GalaxyZoo2_SmoothAndCigarShape', 'GalaxyZoo2_Ring', 'GalaxyZoo2_LensOrArc', 'GalaxyZoo2_DisturbedGalaxy', 'GalaxyZoo2_IrregularGalaxy', 'GalaxyZoo2_SomethingElseOdd', 'GalaxyZoo2_Merger', 'GalaxyZoo2_DustLane', 'GalaxyZoo2_EdgeOnBulgeRounded', 'GalaxyZoo2_EdgeOnBulgeBoxy', 'GalaxyZoo2_EdgeOnBulgeNone', 'GalaxyZoo2_TightlyWoundArms', 'GalaxyZoo2_MediumWoundArms', 'GalaxyZoo2_LooselyWoundArms', 'GalaxyZoo2_OneArm', 'GalaxyZoo2_TwoArms', 'GalaxyZoo2_ThreeArms', 'GalaxyZoo2_FourArms', 'GalaxyZoo2_MultipleArms', 'GalaxyZoo2_AmbigousArms'

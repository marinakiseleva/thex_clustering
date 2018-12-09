# thex_clustering
Clustering analysis for final project in INFO 521 using THEx data.


## Development setup
There are several modules that must be installed in order for this module to work. All can be pip installed except for another custom-made model for THEx: thex_model. That is available here: 
https://github.com/marinakiseleva/thex_model
External depenencies include:
- numpy
- pandas
- matplotlib
- umap from https://github.com/lmcinnes/umap
- scikit-learn

You also need to have the main data source on which this runs: THEx FITS data file. This can be obtained through a request as it is not available publicly yet. 

## Running
Simply run like this:
python run_analysis.py -cols colname_1 colname_2

where colname_1 and colname_2 are the names of columns on which to filter the FITS file on. There is no limit on the number of columns that can be passed, this is just an example. There is also a debug flag.  
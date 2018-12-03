# thex_clustering
Clustering work for final project in INFO 521 and THEx.


## Development setup
You need to have thex_model installed:
https://github.com/marinakiseleva/thex_model
As well as external dependencies like:
- numpy
- pandas
- matplotlib

You also need to have the main data source on which this runs: THEx FITS data file. This can be obtained through a request as it is not available publicly yet. 

## Running
Simply run like this:
python run_analysis.py -cols colname_1 colname_2

where colname_1 and colname_2 are the names of columns on which to filter the FITS file on. There is no limit on the number of columns that can be passed, this is just an example. There is also a debug flag.  
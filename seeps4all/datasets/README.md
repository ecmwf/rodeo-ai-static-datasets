## Sripts to build SEEPS4ALL from ECA&D and collocate raw forecasts

Scripts used to generate the SEEPS4ALL datasets.
They could be used to update the datasets with new dates or build a dataset for another weather parameter.

---


Starting from[ECA&D website](https://www.ecad.eu/dailydata/predefinedseries.php), one can build the SEEPS4ALL observation/seeps/climate dataset using the following scripts sequentially:

- [build_seeps4all_step1](https://github.com/ecmwf/rodeo-ai-static-datasets/blob/seeps/seeps4all/datasets/build_seeps4all_step1.py) # to build one info file (with clim, seeps, ...) per station (one file for each station separately)

- [build_seeps4all_step2](https://github.com/ecmwf/rodeo-ai-static-datasets/blob/seeps/seeps4all/datasets/build_seeps4all_step2.py) # to generate the obs_seeps and obs_clim used in the verifcatiom notebooks.

SEEPS4ALL observation datasets are available on the European Weather Cloud, so one would need to follow these steps only to build datasets including more years or for another weather variables.

The forecast datasets are generated here using grib files as input (please adpat as need if your forecast is in netcdf or any other format). When one would like to assess a new model or experiment the following scripts can be run to collocate forecasts and observations and generate data in a Zarr format.  The forecast dataset is used together with the observations in the verification scripts. To generate a new forecast dataset, one would use the following script:   

- [example_build_collocated_forecast](https://github.com/ecmwf/rodeo-ai-static-datasets/blob/seeps/seeps4all/datasets/example_build_collocated_forecast.py) # for a single deterministic forecast or

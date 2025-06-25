# :umbrella: SEEPS4ALL for the verification of precipitation forecasts
This repository provides tools to generate preprocessed data of daily precipitation in-situ observations derived from **ECA&D** as well as collocated foreacts, together with scripts to compute verification scores.

---

## :small_orange_diamond: What is ECA&D?

[ECA&D](https://www.ecad.eu/dailydata/predefinedseries.php)

## :small_orange_diamond: What is SEEPS4ALL?
SEEPS4ALL resembles scripts to build datasets (observation and forecasts) and notebooks to compute and plot verification metrics. 

The observation datasets are based on ECA&D. The weather variable of interest here is 24h-precipitation. The observation dataset covers the years 2022 to 2024 as it is now. Climatological statistics (including SEEPS coefficients) are computed over the period 1991-2020. The raw data is accessible from [ECA&D website](https://www.ecad.eu/dailydata/predefinedseries.php). Starting from this later, one can build the SEEPS4ALL observation/seeps/climate dataset using the following scripts sequentially:

- A1. gen/build_seeps4all_step1.py # to build one info file (with clim, seeps, ...) per station (one file for each station separately)

- A2. gen/build_seeps4all_step2 .py # to generate the obs_seeps and obs_clim used in the verifcatiom notebooks.

SEEPS4ALL observation datasets are already ready to be used, so one would need to follow steps A1 and A2 only to build datasets including more years or for other weather variables.

The forecast datasets are generated using grib files as input (please adpat as need if your foreacst is in netcdf or any other format). When one would like to assess a new model or experiment the following scripts can be run to collocate forecasts and observations and generate data in a Zarr format.  The forecast dataset is used together with the observations in the verification scripts. To generate a new forecast dataset, one would use the following script:   

- A3. gen/retrieve_single_forecast.py # for a single deterministic forecast or

- A4. gen/retrieve_ensemble_forecast.py # for an ensemble forecast.

Verification notebooks using SEEPS4ALL datasets as input (hese  notebooks can be run in parallel):
- B1. Verification of single deterministic forecast with SEEPS: notebook_1_verif_single.ipynb

- B2. Verification of single deterministic forecast using climate statistics: notebook_2_verif_single_climate.ipynb

- B3. Verification of single deterministic forecast after dressing with probabilistic scores: notebook_3_verif_single_dressed.ipynb

- B4. Verification of ensemble forecasts after with probabilistic scores: notebook_verif_4_single_ensemble.ipynb
 






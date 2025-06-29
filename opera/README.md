# 🌧️ OPERA Radar Precipitation Datasets for Anemoi

This repository provides preprocessed radar-based precipitation datasets derived from **OPERA**, made available in formats compatible with the [Anemoi](https://github.com/ecmwf/anemoi-datasets) machine learning framework.

---

## 📡 What is OPERA?

[OPERA](https://www.eumetnet.eu/observations/weather-radar-network/) is the European weather radar program coordinated by **EUMETNET**, responsible for harmonizing radar observations across national meteorological services.

We focus on **pan-European composite products** from OPERA, specifically:
- **1-hour accumulated total precipitation**
- Based on raw radar data originally at:
  - **15-minute temporal resolution**
  - **2 km spatial resolution**
  - Projected in **Lambert Azimuthal Equal-Area (LAEA)**

---

## 📦 Available Datasets

Each dataset contains:
- **TP**: Total Precipitation (https://codes.ecmwf.int/grib/param-db/228)
- **QI**: Quality Indicator (https://codes.ecmwf.int/grib/param-db/260690)
- **DM**: Data Mask (https://codes.ecmwf.int/grib/param-db/260691)

Available configurations:

| Dataset Type          | Temporal Resolution| Resolution         | Format      | Use Case                       |
|-----------------------|--------------------|--------------------|----------------------------------------------|
| 6-hour aggregated     | 6hr                | N320               | Zarr        | Medium-range NWP + ERA5        |
| 6-hour aggregated     | 6hr                | O96                | Zarr        | Medium-range NWP + ERA5        |
| 1-hour aggregated     | 15min              | N320               | Zarr        | Nowcasting                     |
| 1-hour aggregated     | 15min              | O96                | Zarr        | Nowcasting                     |
| 1-hour native         | 15min              | Native (2km, LAEA) | Zarr        | Training with Observational    |

**Note** - precipitation in OPERA composites is defined in milimeters. While in other datasets like ERA5 is expressed in meters.
In the 1hr and 6hr datasets units have been transformed to mm, while in the native dataset, the units have been kept to meters.
It is possible to apply a rescaling of the variable units using the 'rescale' flag. See [https://anemoi.readthedocs.io/projects/datasets/en/latest/datasets/using/selecting.html] (https://anemoi.readthedocs.io/projects/datasets/en/latest/datasets/using/selecting.html)

**Note:**  
The accumulation window for OPERA radar data **does not** align with the top of the hour.  For example, in a file like `2021-01-01T224500Z-rainfall_accumulation-composite-opera.h5`, the internal metadata shows:

```
[('enddate', b'20210101'),
 ('endtime', b'225000'),
 ('product', b'COMP'),
 ('quantity', b'ACRR'),
 ('startdate', b'20210101'),
 ('starttime', b'215000')]
 ```
which means the data represents accumulated values from **21:50 to 22:50 UTC**, even though the timestamp in the filename is `22:45`.  
so the accumulation window, which spans from **+5 to -45 minutes** around the raw data file timestamp. We have kept this behaviour.
Since 6hr datasets are designed to be matched with ERA5, aggregations have been computed considering top of hours windows.

---

## 🌀 About Anemoi

The datasets are available as [anemoi-datasets](https://github.com/ecmwf/anemoi-datasets).

**Anemoi** is an open-source framework co-developed by ECMWF and several European national meteorological services to build, train, and run data-driven weather forecasts. Its primary goal is to empower meteorological organisations to train machine learning (ML) models using their own data, simplifying the process through shared tools and workflows.

These datasets are provided as **Zarr** datasets, optimized for scalable and efficient storage and access in ML workflows.

To learn how to use **anemoi-training**, documentation is available at [https://anemoi.readthedocs.io/projects/training/en/latest/](https://anemoi.readthedocs.io/projects/training/en/latest/), and the repository [ecmwf/anemoi-configs](https://github.com/ecmwf/anemoi-configs) contains useful example configurations.

**Note** Anemoi also enables the definition of datasets for both regional and global models in two main ways: 

- Using OPERA as a global dataset, with regions outside the area of interest filled with ‘NaNs’ 

- Combining OPERA with a global dataset such as ERA5. It's possible to access a sample ERA5 dataset as explained at [https://anemoi.readthedocs.io/projects/training/en/latest/user-guide/download-era5-o96.html] (https://anemoi.readthedocs.io/projects/training/en/latest/user-guide/download-era5-o96.html)

While defining OPERA as a global dataset may seem counterintuitive, it can be useful for fine-tuning a model already trained on other global datasets like ERA5 or IMERG, especially when aiming to improve the model's performance over Europe. Example on how to define the configs for such cases can be found under 'config_opera_6hr_cutout.yaml' and 'config_opera_6hr_global.yaml'

**Note:**  
The 1-hour configs serve as a minimal working example for using Anemoi with the OPERA dataset only.  However, this should be considered a **non-realistic reference setup**, as Anemoi is designed to work with atmospheric datasets that include both pressure-level and surface variables.

**Note** The configs here consider the following versions of anemoi:
- anemoi-training: 0.5.1
- anemoi-models: 0.8.1
- anemoi-graphs: 0.6.2

If you are interested in trying newer versions of anemoi you can do updating them accordingly to the latest configs that can be found under https://github.com/ecmwf/anemoi-core/tree/main/training/src/anemoi/training/config.

---

## ☁️ Accessing the Datasets

These datasets are publicly hosted on ECMWF’s S3 bucket:

**Bucket URL**: `s3://ecmwf-rodeo-benchmark/opera`

You can use [`s3cmd`](https://s3tools.org/s3cmd) to download the files:

```bash
# Configure s3cmd (only needs to be done once)
s3cmd --configure

# List contents
s3cmd ls s3://ecmwf-rodeo-benchmark/opera/

# Download a specific dataset
s3cmd get --recursive s3://ecmwf-rodeo-benchmark/opera/[dataset_name]
```
where [dataset_name] can be:

- rodeo-opera-files-2km-2013-2023-15m-lambert-azimuthal-equal-area-with-preprocessing.zarr
- rodeo-opera-files-n320-2013-2023-15m.zarr
- rodeo-opera-files-n320-2013-2023-6h.zarr
- rodeo-opera-files-o96-2013-2023-15m.zarr
- rodeo-opera-files-o96-2013-2023-6h.zarr

All datataset have been generated using `anemoi-datasets`. Datasets are available at the native 2 km resolution, as well as at coarser resolutions reprojected onto reduced Gaussian grids, commonly used by IFS and AIFS, at O96 (approximately 1°) and N320 (approximately 31 km). Although coarsening may reduce some fine-scale features in the signal, these versions can significantly ease experimentation and exploration, particularly in the context of medium-range weather forecasting.

All reprojections were performed using ECMWF’s Meteorological Interpolation and Regridding (MIR) software package.
While OPERA composites offer spatially harmonised data, they may include outliers caused by echoes or radar interference. These artefacts have been retained in the signal, but custom pre-processing transformation, provided through the anemoi-transform package, have been developed to ensure physically consistent magnitudes across the three variables: total precipitation (tp), data quality index (dfqi), and radar data quality flag (rdqf). 
Details for those transformations can be found at [https://github.com/ecmwf/anemoi-transform/tree/main/src/anemoi/transform/filters] (https://github.com/ecmwf/anemoi-transform/tree/main/src/anemoi/transform/filters)

**Note:**  
While the instructions above use [`s3cmd`](https://s3tools.org/s3cmd), other tools like [`rclone`](https://rclone.org/s3/#configuration) can also be used to download large datasets. `rclone` generally offers better performance, reliability, and flexibility for large-scale dataset transfers, especially when recursively downloading from S3 buckets.

---

## 📄 License
This dataset is made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
For details check the LICENSE file
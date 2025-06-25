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
- **TP**: Total Precipitation  
- **QI**: Quality Indicator  
- **DM**: Data Mask

Available configurations:

| Dataset Type          | Resolution         | Format | Use Case                       |
|-----------------------|--------------------|--------|--------------------------------|
| 6-hour aggregated     | N320               | Zarr   | Medium-range NWP + ERA5        |
| 6-hour aggregated     | O96                | Zarr   | Medium-range NWP + ERA5        |
| 1-hour aggregated     | N320               | Zarr   | Nowcasting                     |
| 1-hour aggregated     | O96                | Zarr   | Nowcasting                     |
| 1-hour native         | Native (2km, LAEA) | Zarr   | Training with Observational    |

---

## 🌀 About Anemoi

The datasets are available as [anemoi-datasets](https://github.com/ecmwf/anemoi-datasets).

**Anemoi** is an open-source framework co-developed by ECMWF and several European national meteorological services to build, train, and run data-driven weather forecasts. Its primary goal is to empower meteorological organisations to train machine learning (ML) models using their own data, simplifying the process through shared tools and workflows.

These datasets are provided as **Zarr** datasets, optimized for scalable and efficient storage and access in ML workflows.

To learn how to use **anemoi-training**, documentation is available at [https://anemoi.readthedocs.io/projects/training/en/latest/](https://anemoi.readthedocs.io/projects/training/en/latest/), and the repository [ecmwf/anemoi-configs](https://github.com/ecmwf/anemoi-configs) contains useful example configurations.

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

---

## 📄 License
This dataset is made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
For details check the LICENSE file
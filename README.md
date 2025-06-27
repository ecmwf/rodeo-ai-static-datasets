# 🌍 Rodeo Static AI Datasets

This repository provides metadata, documentation, and examples for a set of curated static datasets intended to support AI and ML applications in weather forecasting. These datasets are hosted publicly and maintained by ECMWF as part of the Rodeo AI initiative.

---

## 📁 Repository structure

### [`seeps4all`](https://github.com/ecmwf/rodeo-ai-static-datasets/tree/seeps/seeps4all)
- Contains the ECAD / SEEPS4ALL dataset.
- Includes:
  - `README.md` describing dataset content and structure
  - Jupyter notebook for accessing and using the dataset in the context of **forecast verification**

### [`opera-anemoi`](https://github.com/ecmwf/rodeo-ai-static-datasets/opera)
- Provides OPERA-based datasets tailored for ML training using the **Anemoi** framework.
- Includes:
  - `README.md` detailing data preparation and usage
  - Examples configs for **training ML models** on the data using Anemoi Framework

---

## 🪣 Public S3 Bucket (ECMWF)

Datasets are publicly available via ECMWF’s S3 bucket 's3://ecmwf-rodeo-benchmark', as approved for open access.

### ECAD / SEEPS4ALL
- Format: **Zarr**
- Preprocessed for use in verification workflows

### OPERA 
Total Precipitation (TP) datasets containing:
- **TP** (total precipitation)
- **QI** (quality indicator)
- **DM** (Data mask)

Available configurations:
- **6-hour aggregated** TP at:
  - N320 resolution
  - O96 resolution
- **1-hour aggregated** TP at:
  - N320 resolution
  - O96 resolution
- **1-hour native-resolution** TP (no reprojection)

---

## 📚 Getting Started

Each dataset-specific repository includes:
- A detailed `README.md` with variable descriptions
- Python notebooks or scripts showing how to:
  - Load datasets
  - Use the data in ML pipelines (e.g. Anemoi)
  - Apply the data in forecast verification contexts (e.g. SEEPS score computation)

---

## 📌 License & Attribution

All datasets and scripts are released under open data licenses where applicable. Please check the individual repositories and metadata for specific licensing and citation information.

---

**🔗 Stay tuned for updates as new datasets and tools are added.**

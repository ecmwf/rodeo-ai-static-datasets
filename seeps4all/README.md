# :umbrella: SEEPS4ALL for the verification of precipitation forecasts

This repository provides scripts to compute verification scores for precipitation forecasts. 
Forecasts are compared against in-situ observations using climate statistics for the computation of scores and skill scores.

SEEPS4ALL comprises verification scripts but also a set of tools to generate climate statistics from **ECA&D** as well as collocated forecast and observation data.

An example of a dataset is provided using 1991-2020 for the climate statistics and 2022-2024 for the verification. **ECA&D** is updated regularly so one could update SEEPS4ALL data too if needed. 

---

## :black_square_button: What is ECA&D?

[ECA&D](https://www.ecad.eu) stands for European Climate Assessment & Dataset. 
ECA&D encompassess daily dataset needed to monitor and analyse changes in weather and climate extremes.

Here the focus in on:
- Daily precipitation
- Over Europe

## :black_square_button: What is SEEPS4ALL?

Scores specifically designed to assess the performance of precipitation forecasts have been developed over the years. One example is the Stable and Equitable Error in Probability Space ([SEEPS, Rodwell et al 2010](https://doi.org/10.1002/qj.656)). The computation of this score is however not straightforward because it requires information about the precipitation climatology at the verification locations. More generally, climate statistics are key to assessing forecasts for extreme precipitation and high-impact events. Here, we introduce SEEPS4ALL, a set of data and tools that democratize the use of  climate statistics for verification purposes. In particular, verification results for daily precipitation are showcased with both deterministic and probabilistic forecasts.

---

SEEPS4ALL resembles scripts to build datasets (observations and forecasts) as well as notebooks to compute and plot verification metrics. 

The observation datasets are based on ECA&D. The raw data is accessible from [ECA&D website](https://www.ecad.eu/dailydata/predefinedseries.php).  The weather variable of interest here is 24h-precipitation. 

SEEPS4ALL  observation dataset covers the years 2022 to 2024 (as for now). 

Climatological statistics are computed over the period 1991-2020. They corresponds to:
- the SEEPS parameters that is the thresholds and probabilities of occurence require to compute the score
- percentiles of the climatological distribution at level 1, 2, ..., 98, 99%.
These statistisc are time-of-the-year and station dependent.   

The verification allow the assessment of both deterministic and probabillistic forecasts.  

---

## :paperclip: Citation

```bibtex
@misc{seeps4all,
      title={SEEPS4ALL: an open dataset for the verification of daily precipitation forecasts using station climatology statistics.}, 
      author={{Ben Bouallegue} Zied and al},
      year={2025},
      journal={in preparation}
}
```


## 📄 License
This dataset is made available under the Creative Commons Attribution-Non-Commercial License.
For details check the LICENSE file
 






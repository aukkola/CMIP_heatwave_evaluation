# CMIP_heatwave_evaluation

Repository of Analysis Code for the Manuscript entitled: "CMIP6 Multi-Model Evaluation of Present-Day Heatwave Attributes" [Paper #2021GL095161R] by Hirsch et al. 2021 published in Geophysical Research Letters.

This repository contains all the analysis source code for Hirsch et al 2021 and can be used to reproduce the analysis of the manuscript. Note that all directory paths will need to be updated for successful implementation of the code.

Datasets to which the code applies:

  > global extent historical simulations from CMIP5 and CMIP6 (input variables daily maximum and daily minimum temperature)

Dependencies:

  >	Requires first calculating the EHF heatwave metrics using https://github.com/tammasloughran/ehfheatwaves/releases/tag/v1.2

File summary on python codes to reproduce the figures:

  > plot_cmip_skill.ipynb - a python notebook that uses the calculated EHF heatwave metrics for CMIP5 and CMIP6 to reproduce all analysis and figures in the manuscript

  > plot_mme_vs_sme.ipynb - the python notebook that creates the figures comparing the multi-model vs. the multi-member ensembles of CMIP6.

  > plot_cmip_warming_trend.ipynb - the python notebook used to check the long-term temperature trends for CMIP5 and CMIP6 that produces one of the supplementary figures

  > common_functions.py - python script that contains generic functions that calculating the skill metrics

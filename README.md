# Bachelor-Thesis-Bank-Failure-Prediction-using-CNNs
This repository contains the code and documentation for Michael Rayan’s Senior Project at University College Roosevelt, exploring the use of Convolutional Neural Networks (CNNs) for predicting bank failures using structured financial data.

## Project Structure

bank-failure-prediction/
├── preprocessing.py       # Data preprocessing and matching
├── train_initial.py       # Initial training to find optimal epochs
├── train_final.py         # Final training and evaluation
├── utils.py              # Utility functions
├── config/               # Configuration files
│   └── epochs_config.json
├── data/                 # Input data directory
│   ├── panel.dta
│   └── cbrdataT.dta
├── output/               # Preprocessed datasets
│   └── Banks_12/
├── plots/                # Initial training plots
└── results/              # Final results

## Overview

This project investigates whether CNNs, typically used for image processing, can effectively predict bank failures using numeric financial data.
By reshaping tabular balance sheet and profit–loss data into image-like matrices, CNNs are used to identify spatial and temporal patterns in banks’ financial health.
The analysis focuses on Russian banks (2004–2020) and frames the task as binary classification — predicting whether a bank is “alive” or “dead” (license revoked).

## Methodology

The workflow consists of the following steps:

- Data Matching:
  Each failed (“dead”) bank is matched to a similar surviving (“alive”) bank using Extended CAMEL variables to address class imbalance.
  
- Data Transformation:
  Financial indicators are reshaped into 2D arrays (grayscale image representations) suitable for CNN input.
  
- Model Architecture:
  A simplified CNN architecture is used, optimized for structured numeric data instead of natural images.

## Data Availability
The dataset used in this study is proprietary and cannot be shared publicly due to licensing and confidentiality agreements.

It combines data from:

- The Central Bank of Russia (CBR)
- Supplementary datasets described in Karas & Vernikov (2019)

As a result:

- The raw data files are not included in this repository.
  
- All preprocessing, matching, and modeling code is provided for transparency, but users must supply their own data if they wish to reproduce the analysis.
  
- Publicly available banking data may be used as a substitute for experimentation.

Evaluation:
Performance is measured using accuracy, loss, and ROC–AUC across different variable sets and time spans (3, 6, and 12 months).

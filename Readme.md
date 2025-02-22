# Artificial-intelligence-based decision support tools for the differential diagnosis of colitis

## Introduction

In here you may find the models described in our paper 
[Artificial-intelligence-based decision support tools for the differential diagnosis of colitis](https://doi.org/10.1111/eci.13960), 
used for descriminating IBD from infectious and ischemic colitis using endoscopic images and clinical data.

Three different models were trained and tested:
* a Convolutional Neural Network (CNN) to classify endoscopic images;
* a Gradient Boosted Decision Trees (GBDT) algorithm using five clinical parameters;
* and a hybrid approach (CNN+GBDT) using both endoscopic images and clinical data. 

The script 'demo.py' shows the users how these models can be applied.

## Data availability

Example images are given to show how the models can be used. 
The data used to train, validate, and test the models will be made available by the authors upon 
reasonable request. Datasets can only be shared after formal ethics approval. 

## Prerequisites

* Python (3.8)
* numpy (1.23.5)
* tensorflow (2.11.0)
* lightgbm (3.3.3)
* Pillow (9.3.0)
* pandas (1.5.2)

## Reference

Guimarães P, Finkler H, Reichert MC, Zimmer V, Grünhage F, Krawczyk M, Lammert F, Keller A, Casper M. Artificial‐intelligence‐based decision support tools for the differential diagnosis of colitis. European Journal of Clinical Investigation. 2023 Jun;53(6):e13960. DOI: [10.1111/eci.13960](https://doi.org/10.1111/eci.13960).

If you found this code useful, please cite our paper.

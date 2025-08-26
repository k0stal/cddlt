## CDDLT (Climate Downscaling Deep Learning Tools)

A utility package for climate downscaling using a single-image super-resolution approach.

- [Overview](#overview)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)

## Overview

CDDLT is a high-level interface for training and testing deep learning models for climate downscaling. The package includes pre-implemented models commonly used for super-resolution, such as `SRCNN`, `ESPCN`, `DeepESDtas`, and `SwinIR`.

It also provides ready-to-use PyTorch dataset interfaces for:

- `ReKIS` – Regional Climate Information System for Saxony, Saxony-Anhalt, and Thuringia.
- `CORDEX` - evaluation run of the regional climate model REMO2015 from the EURO-CORDEX.

The data itself is not included in the package.

## Models

The following models are currently implemented:

- **Interpolation** – Baseline for statistical downscaling.
- **SRCNN** (Super-Resolution Convolutional Neural Network)  
  [Learning a Deep Convolutional Network for Image Super-Resolution (Dong et al., 2016)](https://arxiv.org/abs/1501.00092)
- **ESPCN** (Efficient Sub-Pixel Convolutional Neural Network)  
  [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (Shi et al., 2016)](https://arxiv.org/abs/1609.05158)
- **DeepESDpr**
  [Downscaling multi-model climate projection ensembles with deep learning (DeepESD)](https://doi.org/10.5194/gmd-15-6747-2022)
- **DeepESD**
  [Downscaling multi-model climate projection ensembles with deep learning (DeepESD)](https://doi.org/10.5194/gmd-15-6747-2022)
- **EDSR**
  [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921)
- **FNO** (Fourier Neural Operator)  
  [Fourier Neural Operator for Parametric Partial Differential Equations (Li et al., 2021)](https://arxiv.org/abs/2010.08895)
- **SwinIR**
  [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

## Installation

This package is available on PyPI and can be installed via pip:

``` bash
pip install cddlt
```

Or from source:

``` bash
git clone https://github.com/k0stal/cddlt
cd cddlt
pip install .
```

## Usage

`cddlt.DLModule` provides a high-level interface.
Before training, DLModuel has to be configured:

``` python
model.configure(
    optimizer = ...,
    scheduler = ...,
    loss = ...,
    args = ...,
    metrics = {
        ...
    }
)
```

After configuring the model and providing training and validation dataloaders, you can train it using the `fit` method. The training process, including loss curves and metrics, is automatically logged to `TensorBoard`.

``` python
model.fit(
    train_loader = ...,
    dev_loader = ...,
    epochs = ...,
    ...
)
```

`cddlt.DLModule` saves the best model weights by default to the log directory.
Once training is complete, load the weights using `load_weights`

``` python
model.load_weights(...)
```

Then, provided the dataloader, the model can be used for inference:

``` python
predictions = model.predict(...)
```

## Datasets

The module includes prepared dataloaders for `ReKIS` and `CORDEX` datasets, assuming the underlying data is in a specified (TBD) NetCDF format.

### ReKIS

When creating a `ReKIS` dataset instance, you can select the corresponding date ranges, variables to predict, the resampling method used when upscaling the high-resolution image and whether the data should be standardized for training:

``` python
rekis = ReKIS(
    data_path=...,
    variables=["TM"],
    train_len=("2000-01-01", "2000-01-10"),
    dev_len=("2000-01-10", "2000-01-20"),
    test_len=("2000-01-20", "2000-02-01"),
    resampling="cubic_spline",
    standardize=True
)
```

### CORDEX

A `CORDEX` dataset instance is created analogously apart from standardization. We will be using the fitted standardization parameters from the `ReKIS` dataset:

``` python
cordex = CORDEX(
    data_path=...,
    variables=["TM"],
    dev_len=("2000-01-20", "2000-02-01"),
    test_len=("2000-02-01", "2000-03-01"),
    standardize=True,
    standard_params=...
)
```

### Notes

See the testing scenarios for example usage.

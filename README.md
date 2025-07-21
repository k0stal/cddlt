## CDDLT (Climate Downscaling Deep Learning Tools)

A utility package for climate downscaling using a single-image super-resolution approach.

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

## Overview

CDDLT is a high level interface for traning and testing climate downscaling deep learning models. Package comes with preimplemented models commonly used for super-resolution such as: `SRCNN`, `ESPCN`, `DeepESDpr`, `DeepESDtas` and `FNO`.

Package also includes prepared *PyTorch* dataset interfaces for `ReKIS` and `CORDEX`.

## Installation

This package is available on PyPI and can be installed via pip

```
pip install cddlt
```

or from source.

```
git clone https://github.com/k0stal/cddlt
cd cddlt
pip install .
```

## Usage

`cddlt.DLModule` supports high-level interface. First of all we have to `configure` our model.

```
model.configure(
    optimizer = ...,
    scheduler = ...,
    loss = ...,
    metrics = {
        ...
    }
    ...
)
```

After configuration, provided traning and validation dataloaders, we can train our model using the `fit` method. Trainig process, loss curves and metrics are automatically stored to `TensorBoard`.

```
model.fit(
    train_loader = ...,
    dev_loader = ...,
    epochs = ...,
    ...
)
```

#TBD
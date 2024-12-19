<h1 align='center'>Flexutils Toolkit</h1>

<p align="center">
        
<img alt="Supported Python versions" src="https://img.shields.io/badge/Supported_Python_Versions-3.8_%7C_3.9_%7C_3.10_%7C_3.11_%7C_3.12-blue">
<img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/I2PC/Flexutils-Toolkit/total">
<img alt="GitHub License" src="https://img.shields.io/github/license/I2PC/Flexutils-Toolkit">

</p>

<p align="center">
        
<img alt="Flexutils" width="300" src="https://github.com/scipion-em/scipion-em-flexutils/blob/devel/flexutils/icon.png">

</p>

This package provides a neural network environment integrating the deep learning algorithms needed by [scipion-em-flexutils](<https://github.com/scipion-em/scipion-em-flexutils>) plugin.

# Included Networks

- **Zernike3Deep**: Semi-classical neural network to analyze continuous heterogeneity with the Zernike3D basis
- **FlexSIREN**: Smooth and regularized deformation field estimated with neural networks
- **ReconSIREN**: Neural network for ab initio reconstruction and global angular assignment
- **HetSIREN**: Neural network heterogeneous reconstruction for real space
- **FlexConsensus**: Consensus neural network for conformational landscapes

# Installation

The Flexutils-Toolkit relies on Conda and Pip to install it correctly.

Before installation, Conda must be set in the PATH variable so it can be detected during installation. Once this requirement is met, the package be installed by running the file ``install.sh`` after cloning this repository.

# Know issues

For some GPUs and/or drivers, there exists a bug related to the initialization of the CuFFT libraries in Tensorflow:

```bash

  {{function_node __wrapped__rfft2d_device_/job:localhost/replica:0/task:0/device:gpu:0}} failed to create cufft batched plan with scratch allocator [op:rfft2d]

```

If the error appears, it can be solved by reinstalling the CuFFT libraries in the ``flexutils-tensorflow`` environment through conda.

To find the right library version, please, visit the following [page](<https://anaconda.org/nvidia/libcufft>).

An example for Cuda 11.8 is provided below:

```bash

  conda activate flexutils-tensorflow
  conda install -c "nvidia/label/cuda-11.8.0" libcufft
  conda deactivate

```

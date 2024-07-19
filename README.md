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
- **HomoSIREN**: Neural network homogeneous reconstruction for real space
- **HetSIREN**: Neural network heterogeneous reconstruction for real space
- **DeepPose**: Particle pose and shift refinement with neural networks
- **FlexConsensus**: Consensus neural network for conformational landscapes

# Installation

The Flexutils-Toolkit relies on Conda and Pip for its correct installation.

Before installing, Conda must be set in the PATH variable so it can be discovered during the installation. Once this requirement is met, the package can be either installed with ``pip install -e git+https://github.com/I2PC/Flexutils-Toolkit.git@master#egg=flexutils-toolkit`` or with ``pip install -e path/to/cloned/flexutils-toolkit`` after cloning this repository.

We recommend adding the flag `-v` to pip installation command to have a better tracking of the installation.

Additionally, the optional component `Open3D` can be installed to add extra functionalities during the network training phase. In order to install this package, the following requirements must be satisfied:

- CUDA must be installed in your system and properly added to the ``PATH`` and ``LD_LIBRARY_PATH`` variables
- You should check the following dependencies are installed in your system:

```bash

    sudo apt install xorg-dev libxcb-shm0 libglu1-mesa-dev python3-dev clang libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev libtbb-dev libosmesa6-dev libudev-dev autoconf libtool

```

If the previous requirements are not met, `Open3D` installation will be just skipped.

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

=======================
Flexutils Toolkit
=======================

This package provides a neural network environment integrating the deep learning algorithms needed by `scipion-em-flexutils <https://github.com/scipion-em/scipion-em-flexutils>`_ plugin

==========================
Included Networks
==========================

- **Zernike3Deep**: Semi-classical neural network to analyze continuous heterogeneity with the Zernike3D basis
- **DeepNMA**: Semi-classical neural network with automatic NMA selection directly from images to analyze continuous heterogeneity
- **HomoSIREN**: Neural network homogeneous reconstruction for real space
- **HetSIREN**: Neural network heterogeneous reconstruction for real space
- **DeepPose**: Particle pose and shift refinement with neural networks
- **FlexConsensus**: Consensus neural network for conformational landscapes

==========================
Installation
==========================

The Flexutils-Toolkit relies on Conda and Pip for its correct installation.

Before installing, Conda must be set in the PATH variable so it can be discovered during the installation. Once this requirement is met, the package can be either installed with ``pip install -e git+https://github.com/I2PC/Flexutils-Toolkit.git@master#egg=flexutils-toolkit`` or with ``pip install -e path/to/cloned/flexutils-toolkit`` after cloning this repository.

We recommend adding the flag `-v` to pip installation command to have a better tracking of the installation.

Additionally, the optional component `Open3D` can be installed to add extra functionalities during the network training phase. In order to install this package, the following requirements must be satisfied:

- CUDA must be installed in your system and properly added to the ``PATH`` and ``LD_LIBRARY_PATH`` variables
- You should check the following dependencies are installed in your system:

.. code-block::

    sudo apt install xorg-dev libxcb-shm0 libglu1-mesa-dev python3-dev clang libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev libtbb-dev libosmesa6-dev libudev-dev autoconf libtool

If the previous requirements are not met, `Open3D` installation will be just skipped.

==========================
Know issues
==========================

For some GPUs and/or drivers, there exists a bug related to the initialization of the CuFFT libraries in Tensorflow:

.. code-block::

  {{function_node __wrapped__rfft2d_device_/job:localhost/replica:0/task:0/device:gpu:0}} failed to create cufft batched plan with scratch allocator [op:rfft2d]

If the error appears, it can be solved by reinstalling the CuFFT libraries in the ``flexutils-tensorflow`` environment through conda.

To find the right library version, please, visit the following `page <https://anaconda.org/nvidia/libcufft>`_.

An example for Cuda 11.8 is provided below:

.. code-block::

  conda activate flexutils-tensorflow
  conda install -c "nvidia/label/cuda-11.8.0" libcufft
  conda deactivate

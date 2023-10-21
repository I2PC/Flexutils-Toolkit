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

==========================
Know issues
==========================

For some GPUs and/or drivers, there exists a bug related to the initialization of the CuFFT libraries in Tensorflow:

```
{{function_node __wrapped__rfft2d_device_/job:localhost/replica:0/task:0/device:gpu:0}} failed to create cufft batched plan with scratch allocator [op:rfft2d]
```

If the error appears, it can be solved by reinstalling the CuFFT libraries in the `flexutils-tensorflow` environment through conda.

To find the right library version, please, visit the following [page](https://anaconda.org/nvidia/libcufft).

An example for Cuda 11.8 is provided below:

```
conda activate flexutils-tensorflow
conda install -c "nvidia/label/cuda-11.8.0" libcufft
conda deactivate
```
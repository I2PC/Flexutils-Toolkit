=======================
Flexutils Tensorflow Toolkit
=======================

This package provides a Tensorflow environment integrating the deep learning algorithms needed by [scipion-em-flexutils](https://github.com/scipion-em/scipion-em-flexutils) plugin

==========================
Included Networks
==========================

- **Zernike3Deep**: Semi-classical neural network to analyze continuous heterogeneity with the Zernike3D basis
- **DeepNMA**: Semi-classical neural network with automatic NMA selection directly from images to analyze continuous heterogeneity
- **DeepPose**: Particle pose and shift refinement with neural networks

==========================
Installation
==========================

The Flexutils-Tensorflow toolkit relies on Conda and Pip for its correct installation.

Before installing, Conda must be set in the PATH variable so it can be discovered during the installation. Once this requirement is met, the package can be either installed with `pip install tensorflow-toolkit` or with `pip install -e path/to/cloned/tensorflow-toolkit` after cloning this repository.

We recommend adding the flag `-v` to pip installation command to have a better tracking of the installation.
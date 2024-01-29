#!/bin/bash

# List of required packages
required_packages=(
    xorg-dev
    libxcb-shm0
    libglu1-mesa-dev
    python3-dev
    clang
    libc++-dev
    libc++abi-dev
    libsdl2-dev
    ninja-build
    libxi-dev
    libtbb-dev
    libosmesa6-dev
    libudev-dev
    autoconf
    libtool
)

# Function to check if a package is installed
is_package_installed() {
    if dpkg -l $1 &> /dev/null; then
        return 0
    else
        echo "Package '$1' is not installed."
        return 1
    fi
}

# Check each package
missing_packages=()
for pkg in "${required_packages[@]}"; do
    if ! is_package_installed "$pkg"; then
        missing_packages+=("$pkg")
    fi
done

# Inform the user about missing packages
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "The following packages are missing:"
    for pkg in "${missing_packages[@]}"; do
        echo " - $pkg"
    done
    echo "Open3D functionalities will not be available. If you want them to be used, please, install the listed
    packages as sudo and rerun the scipion-em-flexutils plugin installation."
    exit 0
fi

# Activate conda in shell
if which conda | sed 's: ::g' &> /dev/null ; then
  CONDABIN=$(which conda | sed 's: ::g')
  eval "$($CONDABIN shell.bash hook)"
else
  echo "Conda not found in path - Exiting"
  exit 1
fi

# Clone Open3D
echo "##### Cloning Open3D... #####"
CURRENT_DIR = $(pwd)
cd ..
git clone https://github.com/isl-org/Open3D
echo "##### Done! #####"

# Create folders needed for the build
echo "##### Creating build folders... #####"
cd Open3D/
mkdir build
cd build/
echo "##### Done! #####"

# Create installation folder for open3D
echo "##### Create installation folder... #####"
mkdir ../open3d_install
echo "##### Done! #####"

# Clone Open3D ML
echo "##### Cloning Open3D-ML... #####"
git clone https://github.com/isl-org/Open3D-ML.git
echo "##### Done! #####"

# Ensure dependencies are installed (needs sudo)
#echo "##### Installing extra dependencies (needs SUDO)... #####"
#../util/install_deps_ubuntu.sh
#echo "##### Done! #####"

# Check Cuda is installed in the system
echo "##### Checking Cuda... #####"
nvcc -V || (echo "Cuda not installed in the system. Please, install Cuda." && exit)
echo "##### Done! #####"

# Activate Flexutils conda environment
echo "##### Getting Flexutils-Tensorflow python... #####"
conda activate flexutils-tensorflow
PYTHON_CONDA=$CONDA_PREFIX"/bin/python"
conda deactivate
echo "##### Done! #####"

# CMake call (including Tensorflow)
echo "##### Generating building files... #####"
conda activate flexutils-tensorflow
cmake -DBUILD_CUDA_MODULE=ON -DGLIBCXX_USE_CXX11_ABI=ON -DBUILD_TENSORFLOW_OPS=ON -DBUNDLE_OPEN3D_ML=ON -DOPEN3D_ML_ROOT=./Open3D-ML -DCMAKE_INSTALL_PREFIX=../open3d_install -DPython3_ROOT=$PYTHON_CONDA ..
echo "##### Done! #####"

# Install (needs Flexutils-Tensorflow environment)
echo "##### Installing Open3D... #####"
#conda activate flexutils-tensorflow
make -j12
echo "##### Done! #####"

# Install Python package in environment
echo "##### Installing Open3D in Flexutils-Tensorflow environment... #####"
make install-pip-package
echo "##### Done! #####"

# Deactivate Flexutils-Tensorflow environment
conda deactivate
cd "$CURRENT_DIR"
echo "##### Installation finished succesfully! #####"

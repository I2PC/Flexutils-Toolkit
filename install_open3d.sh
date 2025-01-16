#!/bin/bash

# Function to echo text in specified color using tput and printf
colored_echo() {
    local color=$1
    local text=$2

    # Define color codes using tput
    local black=$(tput setaf 0)
    local red=$(tput setaf 1)
    local green=$(tput setaf 2)
    local yellow=$(tput setaf 3)
    local blue=$(tput setaf 4)
    local magenta=$(tput setaf 5)
    local cyan=$(tput setaf 6)
    local white=$(tput setaf 7)
    local reset=$(tput sgr0)

    # Choose color based on input
    case $color in
        "black") color_code=$black ;;
        "red") color_code=$red ;;
        "green") color_code=$green ;;
        "yellow") color_code=$yellow ;;
        "blue") color_code=$blue ;;
        "magenta") color_code=$magenta ;;
        "cyan") color_code=$cyan ;;
        "white") color_code=$white ;;
        *) color_code=$reset ;; # Default to reset if no color match
    esac

    # Print the colored text
    printf "%b%s%b\n" "$color_code" "$text" "$reset"
}

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

# Compare versions
compare_versions() {
    local IFS=.
    local i
    local ver1=($1)
    local ver2=($2)

    # Determine the maximum length of the version arrays
    local max_length=${#ver1[@]}
    (( ${#ver2[@]} > max_length )) && max_length=${#ver2[@]}

    # Fill empty fields with zeros
    for ((i=0; i<max_length; i++)); do
        [[ -z ${ver1[i]} ]] && ver1[i]=0
        [[ -z ${ver2[i]} ]] && ver2[i]=0
    done

    # Compare version components
    for ((i=0; i<max_length; i++)); do
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 2
        fi
    done
    return 0
}

# Inform the user about missing packages
#if [ ${#missing_packages[@]} -ne 0 ]; then
#    colored_echo "yellow" "The following packages are missing:"
#    for pkg in "${missing_packages[@]}"; do
#        colored_echo "yellow" " - $pkg"
#    done
#    colored_echo "yellow" "Open3D functionalities will not be available. If you want them to be used, please, install the listed
#    packages as sudo and rerun the scipion-em-flexutils plugin installation."
#    exit 0
#fi

# Read input parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --condaBin) CONDABIN="$2"; shift ;; # Capture the first argument
        -h|--help) echo "Usage: $0 [--condaBin VALUE]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Activate conda in shell
if [[ ! -v CONDABIN ]]; then
  if which conda | sed 's: ::g' &> /dev/null ; then
    CONDABIN=$(which conda | sed 's: ::g')
    eval "$($CONDABIN shell.bash hook)"
  else
    colored_echo "red" "Conda not found in path - Exiting"
    exit 1
  fi
else
  eval "$($CONDABIN shell.bash hook)"
fi

# Clone Open3D
colored_echo "green" "##### Cloning Open3D... #####"
CURRENT_DIR=$(pwd)
cd ..
git clone https://github.com/isl-org/Open3D
colored_echo "green" "##### Done! #####"

# Create folders needed for the build
colored_echo "green" "##### Creating build folders... #####"
cd Open3D/
mkdir build
cd build/
colored_echo "green" "##### Done! #####"

# Create installation folder for open3D
colored_echo "green" "##### Create installation folder... #####"
mkdir ../open3d_install
colored_echo "green" "##### Done! #####"

# Clone Open3D ML
colored_echo "green" "##### Cloning Open3D-ML... #####"
git clone https://github.com/isl-org/Open3D-ML.git
colored_echo "green" "##### Done! #####"

# Ensure dependencies are installed (needs sudo)
#echo "##### Installing extra dependencies (needs SUDO)... #####"
#../util/install_deps_ubuntu.sh
#echo "##### Done! #####"

# Check Cuda is installed in the system
colored_echo "green" "##### Checking Nvidia Drivers version... #####"
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia_minimum_version="535.054.03"
    nvidia_driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1)
    # nvidia_driver_version=$(echo $nvidia_driver_version | sed 's/V//;s/\([0-9]*\.[0-9]*\).*/\1/')
#    nvidia_driver_version_cleaned=$(echo $nvidia_driver_version | awk -F. '{printf "%d%02d%02d", $1, $2, $3}')
    compare_versions "$version1" "$version2"
    result=$?
    if [ "$result" -ge 0 ]; then
        colored_echo "green" "Nvidia Drivers OK"
    else
        colored_echo "yellow" "The version of the Nvidia Drivers available in your system does not meet Open3D
        requirements. Your current version is $nvidia_driver_version, but you will need at least 535.054.03. Please,
        update your drivers and run the installation again to get Open3D."
        exit 0
    fi
else
    colored_echo "yellow" "Nvidia Drivers not found, exiting. To installed Open3D capabilities, please, install at
    least Nvidia Drivers v535.054.03 and run the installation again to get Open3D."
    exit 0
fi
colored_echo "green" "##### Done! #####"

# Activate Flexutils conda environment
colored_echo "green" "##### Getting Flexutils-Tensorflow python... #####"
conda activate flexutils-tensorflow
PYTHON_CONDA=$CONDA_PREFIX"/bin/python"
colored_echo "green" "##### Done! #####"

# Check each package
missing_packages=()
for pkg in "${required_packages[@]}"; do
    if ! is_package_installed "$pkg"; then
        missing_packages+=("$pkg")
    fi
done

# CMake call (including Tensorflow)
colored_echo "green" "##### Generating building files... #####"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
#cmake -DCMAKE_CXX_FLAGS="-Wno-error=unused-result" -DBUILD_CUDA_MODULE=ON -DGLIBCXX_USE_CXX11_ABI=ON -DBUILD_TENSORFLOW_OPS=ON -DBUNDLE_OPEN3D_ML=ON -DBUILD_GUI=ON -DBUILD_WEBRTC=ON -DBUILD_EXAMPLES=OFF -DOPEN3D_ML_ROOT=./Open3D-ML -DCMAKE_INSTALL_PREFIX=../open3d_install -DPython3_ROOT=$PYTHON_CONDA ..
cmake -DBUILD_CUDA_MODULE=ON -DGLIBCXX_USE_CXX11_ABI=ON -DBUILD_TENSORFLOW_OPS=ON -DBUNDLE_OPEN3D_ML=ON -DBUILD_GUI=ON -DBUILD_WEBRTC=ON -DBUILD_EXAMPLES=OFF -DOPEN3D_ML_ROOT=./Open3D-ML -DCMAKE_INSTALL_PREFIX=../open3d_install -DPython3_ROOT=$PYTHON_CONDA ..
colored_echo "green" "##### Done! #####"

# Build and install Python package in environment (needs Flexutils-Tensorflow environment)
colored_echo "green" "##### Installing Open3D in Flexutils-Tensorflow environment...... #####"
make install-pip-package -j12
colored_echo "green" "##### Done! #####"

# Deactivate Flexutils-Tensorflow environment
conda deactivate
cd "$CURRENT_DIR"
colored_echo "green" "##### Installation finished succesfully! #####"

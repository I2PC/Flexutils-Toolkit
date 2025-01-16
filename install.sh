#!/bin/bash

# Function to check if a Python package is installed
is_python_package_installed() {
  conda activate flexutils-tensorflow
    if pip list | grep -F "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
  conda deactivate
}

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

# Read input parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --condaBin) CONDABIN="$2"; shift ;; # Capture the first argument
        -h|--help) echo "Usage: $0 [--condaBin VALUE]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

colored_echo "green" "-------------- Installing Flexutils-toolkit --------------"

# Name of the Python package to check
python_package="open3d"

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

# Install pynvml package in current env (installation dependency)
export LD_LIBRARY_PATH=""
conda create -y -n install-temp python=3.9
conda activate install-temp
colored_echo "green" "Adding installation dependencies to current env..."
pip install pynvml packaging
colored_echo "green" "...Done"

# Run Conda installation
if python tensorflow_toolkit/build.py ; then
  colored_echo "green" "Environment: flexutils-tensorflow built succesfully"
else
  colored_echo "red" "Error when building flexutils-tensorflow - Exiting"
  if which conda | sed 's: ::g' ; then
    conda env remove -n flexutils-tensorflow -y
  fi
  exit 1
fi

# Deactivate and remove temporal installation environment
conda deactivate
conda env remove -n install-temp -y

# Install flexutils-toolkit in flexutils-tensorflow env
conda activate flexutils-tensorflow
pip install -e . -v
conda deactivate
colored_echo "green" "Package flexutils-toolkit succesfully installed in flexutils-tensorlfow env"
colored_echo "green" "-------------- Flexutils-toolkit installation finished! --------------"

# Install Open3D
if is_python_package_installed $python_package; then
  colored_echo "green" "Open3D package is already installed. Skipping..."
else
  colored_echo "green" "Installing Open3D..."
  bash ./install_open3d.sh
  colored_echo "green" "Done..."
fi

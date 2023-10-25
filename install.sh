echo "-------------- Installing Flexutils-toolkit --------------"

# Activate conda in shell
if which conda | sed 's: ::g' &> /dev/null ; then
  CONDABIN=$(which conda | sed 's: ::g')
  eval "$($CONDABIN shell.bash hook)"
else
  echo "Conda not found in path - Exiting"
  exit 1
fi

# Install pynvml package in current env (installation dependency)
echo "Adding installation dependencies to current env..."
pip install pynvml packaging
echo "...Done"

# Run Conda installation
if python tensorflow_toolkit/build.py ; then
  echo "Environment: flexutils-tensorflow built succesfully"
else
  echo "Error when building flexutils-tensorflow - Exiting"
  if which conda | sed 's: ::g' ; then
    conda env remove -n flexutils-tensorflow
  fi
  exit 1
fi

# Install flexutils-toolkit in flexutils-tensorflow env
conda activate flexutils-tensorflow
pip install -e . -v
conda deactivate
echo "Package flexutils-toolkit succesfully installed in flexutils-tensorlfow env"
echo "-------------- Flexutils-toolkit installation finished! --------------"

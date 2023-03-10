"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""


import os
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import sys
import subprocess
import pkg_resources
from packaging import version

# Install pynvml if not present before installing the package
required = {'pynvml', 'packaging'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


scripts = os.listdir(os.path.join("toolkit/tensorflow_toolkit", "scripts"))
scripts.remove("__init__.py")
scripts = [os.path.join("toolkit/tensorflow_toolkit", "scripts", script)
           for script in scripts if ".py" in script]

# Get all GPUs and models
from pynvml import nvmlDeviceGetName, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetCount, nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion
import re
nvmlInit()
number_devices = nvmlDeviceGetCount()
gpu_models = [nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)) for i in
              range(number_devices)]
driver = nvmlSystemGetDriverVersion()
nvmlShutdown()

# Default values compatible with Series 2000 and below
cuda_version = "10"

# If at least one GPU is series 3000 and above, change installation requirements
for gpu_model in gpu_models:
    if re.findall(r"30[0-9]+", gpu_model) or version.parse("450.80.02") <= version.parse(driver):
        cuda_version = "11"
        break

conda_path_command = r"conda info --envs | grep -Po 'flexutils-tensorflow\K.*' | sed 's: ::g'"
if cuda_version == "11" or version.parse("450.80.02") <= version.parse(driver):
    req_file = os.path.join("requirements", "tensorflow_2_11_requirements.txt")
    command = "if ! { conda env list | grep 'flexutils-tensorflow'; } >/dev/null 2>&1; then " \
              "conda create -y -n flexutils-tensorflow " \
              "-c conda-forge python=3.8 cudatoolkit=11.2 cudnn=8.1.0 cudatoolkit-dev -y; fi"
              # "conda create -y -n test -c conda-forge python=3.8 -y; fi"
else:
    req_file = os.path.join("requirements", "tensorflow_2_3_requirements.txt")
    command = "if ! { conda env list | grep 'flexutils-tensorflow'; } >/dev/null 2>&1; then " \
              "conda create -y -n flexutils-tensorflow -c conda-forge python=3.8 cudatoolkit=10.1 cudnn=7 -y; fi" % req_file

print("Installing Tensorflow conda env...")
subprocess.check_call(command, shell=True, stdout=subprocess.PIPE)
print("...done")
print("Getting env pip...")
path = subprocess.check_output(conda_path_command, shell=True).decode("utf-8").replace('\n', '')
install_toolkit_command = "%s/bin/pip install -r %s && %s/bin/pip install -e toolkit" % (path, req_file, path)
print("...done")
print("Installing Flexutils-Tensorflow toolkit in conda env...")
subprocess.check_call(install_toolkit_command, shell=True, stdout=subprocess.PIPE)
print("...done")

setup(name='setup_env',
      version="1.0.0",  # Required
      description='Xmipp tensorflow utilities for flexibility',
      author='David Herreros',
      author_email='dherreros@cnb.csic.es',
      keywords='scipion continuous-heterogeneity imageprocessing xmipp',
      packages=find_packages(),
      package_data={  # Optional
         'requirements': ["*"],
         'toolkit': ["*"]
      }
      )

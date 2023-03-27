"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""


import os
import sys
import subprocess
import pkg_resources
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools.command.install import install
import atexit
# To use a consistent encoding
from codecs import open

from setup_env import __version__


class Installation(install):
    # Installation assumes conda is in the PATH
    def __init__(self, *args, **kwargs):
        super(Installation, self).__init__(*args, **kwargs)
        atexit.register(self.runCondaInstallation)

    def print_flush(self, str):
        print(str)
        sys.stdout.flush()

    def installMissingPackages(self):
        # Install pynvml and packaging if not present before installing the package
        required = {'pynvml', 'packaging'}
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed

        if missing:
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

    def condaInstallationCommands(self):
        # Get all GPUs and models and drivers
        from pynvml import nvmlDeviceGetName, nvmlDeviceGetHandleByIndex, \
            nvmlDeviceGetCount, nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion
        from packaging import version
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

        # Command: Get path to new conda env
        conda_path_command = r"conda info --envs | grep -Po 'flexutils-tensorflow\K.*' | sed 's: ::g'"

        # Command: Get condabin/conda
        condabin_path_command = r"which conda | sed 's: ::g'"

        # Command: Get installation of new conda env with Cuda, Cudnn, and Tensorflow dependencies
        if cuda_version == "11" or version.parse("450.80.02") <= version.parse(driver):
            req_file = os.path.join("requirements", "tensorflow_2_11_requirements.txt")
            command = "if ! { conda env list | grep 'flexutils-tensorflow'; } >/dev/null 2>&1; then " \
                      "conda create -y -n flexutils-tensorflow " \
                      "-c conda-forge python=3.8 cudatoolkit=11.2 cudnn=8.1.0 cudatoolkit-dev -y; fi"
        else:
            req_file = os.path.join("requirements", "tensorflow_2_3_requirements.txt")
            command = "if ! { conda env list | grep 'flexutils-tensorflow'; } >/dev/null 2>&1; then " \
                      "conda create -y -n flexutils-tensorflow -c conda-forge python=3.8 cudatoolkit=10.1 cudnn=7" \
                      "cudatoolkit-dev -y; fi"

        return req_file, conda_path_command, condabin_path_command, command

    def runCondaInstallation(self):
        # Check conda is in PATH
        try:
            subprocess.check_call("conda", shell=True, stdout=subprocess.PIPE)
            self.print_flush("Conda found in PATH")
        except:
            raise Exception("Conda not found in PATH \n "
                            "Installation will be aborted \n"
                            "Install Conda and/or add it to the PATH variable and try to install again "
                            "this package with 'pip install tensorflow-toolkit'")

        # Install Tensorflow-Toolkit conda environment
        self.print_flush("Installing missing packages...")
        self.installMissingPackages()
        self.print_flush("...done")

        req_file, _, condabin_path_command, install_conda_command = self.condaInstallationCommands()

        self.print_flush("Installing Tensorflow conda env...")
        subprocess.check_call(install_conda_command, shell=True)
        self.print_flush("...done")

        self.print_flush("Getting env pip...")
        path = subprocess.check_output(condabin_path_command, shell=True).decode("utf-8").replace('\n', '').replace("*", "")
        install_toolkit_command = 'eval "$(%s shell.bash hook)" && conda activate flexutils-tensorflow && ' \
                                  'pip install -r %s && pip install -e toolkit' % (path, req_file)
        self.print_flush("...done")

        self.print_flush("Installing Flexutils-Tensorflow toolkit in conda env...")
        subprocess.check_call(install_toolkit_command, shell=True)
        self.print_flush("...done")


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


scripts = os.listdir(os.path.join("toolkit/tensorflow_toolkit", "scripts"))
scripts.remove("__init__.py")
scripts = [os.path.join("toolkit/tensorflow_toolkit", "scripts", script)
           for script in scripts if ".py" in script]

setup(name='scipionn-toolkit',
      version=__version__,  # Required
      description='Xmipp neural network utilities for flexibility',
      long_description=long_description,  # Optional
      author='David Herreros',
      author_email='dherreros@cnb.csic.es',
      keywords='scipion continuous-heterogeneity imageprocessing xmipp',
      packages=find_packages(),
      package_data={  # Optional
         'requirements': ["*"],
         'toolkit': ["*"]
      },
      cmdclass={'install': Installation}
      )

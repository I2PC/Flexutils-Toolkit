# **************************************************************************
# *
# * Authors:     David Herreros (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import os
import sys
import subprocess
from pynvml import nvmlDeviceGetName, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetCount, nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion
from packaging import version
import re


class Installation:
    def print_flush(self, str):
        print(str)
        sys.stdout.flush()

    def condaInstallationCommands(self):
        # Get all GPUs and models and drivers
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
            if re.findall(r"40[0-9]+", gpu_model) or re.findall(r"30[0-9]+", gpu_model) and \
                    version.parse("520.56.06") <= version.parse(driver):
                cuda_version = "11.8"
                break
            elif re.findall(r"30[0-9]+", gpu_model) or version.parse("450.80.02") <= version.parse(driver):
                cuda_version = "11.2"
                break

        self.print_flush("Cuda version to be installed: " + cuda_version)

        # Command: Get condabin/conda
        condabin_path = subprocess.run(r"which conda | sed 's: ::g'", shell=True, check=False,
                                       stdout=subprocess.PIPE).stdout
        condabin_path = condabin_path.decode("utf-8").replace('\n', '').replace("*", "")

        # Check if conda env is installed
        env_installed = subprocess.run(
            r"conda env list | grep 'flexutils-tensorflow '",
            shell=True, check=False, stdout=subprocess.PIPE).stdout
        env_installed = bool(env_installed.decode("utf-8").replace('\n', '').replace("*", ""))
        check_cuda = subprocess.run(
            r"conda list -n flexutils-tensorflow | grep 'cudatoolkit ' | grep -Eo '[0-9]+\.[0-9]+'",
            shell=True, check=False, stdout=subprocess.PIPE).stdout
        check_cuda = check_cuda.decode("utf-8").replace('\n', '').replace("*", "")
        if check_cuda != cuda_version:
            env_installed = False

        # Command: Get installation of new conda env with Cuda, Cudnn, and Tensorflow dependencies
        if not env_installed:
            if cuda_version == "11.8":
                tensorflow = "2.12"
                req_file = os.path.join("requirements", "tensorflow_2_12_requirements.txt")
                command = "conda env remove -n flexutils-tensorflow && conda create -y -n flexutils-tensorflow " \
                          "-c conda-forge python=3.9 cudatoolkit=11.8 cudatoolkit-dev pyyaml"
            elif cuda_version == "11.2":
                tensorflow = "2.11"
                req_file = os.path.join("requirements", "tensorflow_2_11_requirements.txt")
                command = "conda env remove -n flexutils-tensorflow && conda create -y -n flexutils-tensorflow " \
                          "-c conda-forge python=3.8 cudatoolkit=11.2 cudnn=8.1.0 cudatoolkit-dev pyyaml"
            else:
                tensorflow = "2.3"
                req_file = os.path.join("requirements", "tensorflow_2_3_requirements.txt")
                command = "conda env remove -n flexutils-tensorflow && conda create -y -n flexutils-tensorflow " \
                          "-c conda-forge python=3.8 cudatoolkit=10.1 cudnn=7 cudatoolkit-dev pyyaml"
        else:
            tensorflow = "None"
            req_file = None
            command = None

        return req_file, condabin_path, command, cuda_version, tensorflow

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

        req_file, condabin_path, install_conda_command, cuda_version, tensorlfow = self.condaInstallationCommands()

        # Install flexutils-tensorflow conda environment
        self.print_flush("Installing Tensorflow conda env...")
        if install_conda_command is not None:
            subprocess.check_call(install_conda_command, shell=True)
            self.print_flush("...done")
        else:
            self.print_flush("Already installed - skipping")

        # Get command to install envrionment and pip dependencies
        self.print_flush("Getting env pip...")
        if install_conda_command is not None:
            if tensorlfow == "2.12":
                install_toolkit_command = 'eval "$(%s shell.bash hook)" && conda activate flexutils-tensorflow && ' \
                                          'conda install -y -c nvidia cuda-nvcc=11.3.58 && ' \
                                          'pip install nvidia-cudnn-cu11==8.6.0.163 && ' \
                                          'pip install -r %s' % (condabin_path, req_file)

            else:
                install_toolkit_command = 'eval "$(%s shell.bash hook)" && conda activate flexutils-tensorflow && ' \
                                          'pip install -r %s' % (condabin_path, req_file)
            self.print_flush("...done")
        else:
            self.print_flush("Already installed - skipping")

        # Install remaining dependencies in env
        if install_conda_command is not None:
            self.print_flush("Installing Flexutils-Tensorflow toolkit in conda env...")
            subprocess.check_call(install_toolkit_command, shell=True)
            self.print_flush("...done")
        else:
            self.print_flush("Already installed - skipping")

        # Set Tensorflow env variables when env is activated
        self.print_flush("Set environment variables in conda env...")
        if tensorlfow == "2.12":
            commands = ['eval "$(%s shell.bash hook) "' % condabin_path,
                        'conda activate flexutils-tensorflow ',
                        'mkdir -p $CONDA_PREFIX/etc/conda/activate.d ',
                        'echo \'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))\''
                        ' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh ',
                        'echo \'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib\' '
                        '>> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh ',
                        'echo \'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n\' >> '
                        '$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh ',
                        'mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice ',
                        'cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/'
                        ]
        else:
            commands = ['eval "$(%s shell.bash hook) "' % condabin_path,
                        'conda activate flexutils-tensorflow ',
                        'mkdir -p $CONDA_PREFIX/etc/conda/activate.d ',
                        'echo \'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/\' '
                        '>> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh ',
                        'echo \'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n\' >> '
                        '$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh ',
                        'mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice ',
                        'cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/'
                        ]
        commands = "&&".join(commands)
        subprocess.check_call(commands, shell=True)
        self.print_flush("...done")


if __name__ == '__main__':
    Installation().runCondaInstallation()

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
import argparse
from pynvml import NVMLError
from packaging import version
import re


def get_nvidia_driver_version_no_nvml():
    try:
        with open("/proc/driver/nvidia/version", "r") as f:
            content = f.read()
    except IOError as e:
        raise RuntimeError("Unable to read /proc/driver/nvidia/version") from e
    # Search for a pattern like '535.183.01'
    match = re.search(r'\b(\d+\.\d+\.\d+)\b', content)
    if match:
        return match.group(1)
    else:
        raise RuntimeError("Driver version not found in the file.")


class Installation:
    def print_flush(self, str):
        print(str)
        sys.stdout.flush()

    def condaInstallationCommands(self, condabin_path=None):
        try:
            from pynvml import nvmlDeviceGetName, nvmlDeviceGetHandleByIndex, \
                nvmlDeviceGetCount, nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion

            # Get all GPUs and models and drivers
            nvmlInit()
            number_devices = nvmlDeviceGetCount()
            gpu_models = [nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)) for i in
                          range(number_devices)]
            driver = nvmlSystemGetDriverVersion()
            nvmlShutdown()
        except NVMLError:
            driver = get_nvidia_driver_version_no_nvml()

        # Default values compatible with Series 2000 and below
        cuda_version = "10"

        # If at least one GPU is series 3000 and above, change installation requirements
        for gpu_model in gpu_models:
            if version.parse("535.54.03") <= version.parse(driver):
                cuda_version = "12.2"
                break
            elif re.findall(r"40[0-9]+", gpu_model) or re.findall(r"30[0-9]+", gpu_model) or \
                    version.parse("520.56.06") <= version.parse(driver):
                cuda_version = "11.8"
                break
            elif re.findall(r"30[0-9]+", gpu_model) or version.parse("450.80.02") <= version.parse(driver):
                cuda_version = "11.2"
                break

        self.print_flush("Cuda version to be installed: " + cuda_version)

        # Command: Get condabin/conda
        if condabin_path is None:
            condabin_path = subprocess.run(r"which conda | sed 's: ::g'", shell=True, check=False,
                                           stdout=subprocess.PIPE).stdout
            condabin_path = condabin_path.decode("utf-8").replace('\n', '').replace("*", "")
        command = f'eval "$({condabin_path} shell.bash hook)" && '

        # Check if conda env is installed
        env_installed = subprocess.run(
            f'eval "$({condabin_path} shell.bash hook)" && ' +  r"conda env list | grep 'flexutils-tensorflow '",
            shell=True, check=False, stdout=subprocess.PIPE).stdout
        env_installed = bool(env_installed.decode("utf-8").replace('\n', '').replace("*", ""))
        check_cuda_conda = subprocess.run(
            f'eval "$({condabin_path} shell.bash hook)" && '
            +  r"conda list -n flexutils-tensorflow | grep 'cudatoolkit ' | grep -Eo '[0-9]+\.[0-9]+'",
            shell=True, check=False, stdout=subprocess.PIPE).stdout
        check_cuda_pip = subprocess.run(
            f'eval "$({condabin_path} shell.bash hook)" && '
            +  r"conda list -n flexutils-tensorflow | grep 'nvidia-cuda-nvcc*' | grep -Eo '[0-9]+\.[0-9]+'",
            shell=True, check=False, stdout=subprocess.PIPE).stdout
        check_cuda = check_cuda_conda or check_cuda_pip
        check_cuda = check_cuda.decode("utf-8").replace('\n', '').replace("*", "")

        if check_cuda != cuda_version:
            if env_installed:
                command += "conda env remove -n flexutils-tensorflow && "
            env_installed = False

        # Command: Get installation of new conda env with Cuda, Cudnn, and Tensorflow dependencies
        if not env_installed:
            if cuda_version == "12.2":
                tensorflow = "2.17"
                req_file = os.path.join("requirements", "tensorflow_2_17_requirements.txt")
                command += ("conda create -y -n flexutils-tensorflow "
                            "-c nvidia/label/cuda-12.2.0 -c conda-forge -c anaconda python=3.9 pyyaml=6.0.1 cuda=12.2.0 "
                            "cmake=3.29.3 make=4.3 mesalib=24.1.0 libglu=9.0.0 xorg-libx11=1.8.9 xorg-libxrandr=1.5.2 "
                            "xorg-libxinerama=1.1.5 xorg-libxcursor=1.2.0 libxcb=1.15 libcxx=17.0.6 libcxxabi=17.0.6 "
                            "sdl2=2.30.2 ninja=1.12.1 xorg-libxi=1.7.10 tbb-devel=2021.12.0 libudev=255 autoconf=2.71 "
                            "libtool=2.4.7 cxx-compiler=1.7.0 gcc=12.3.0 c-compiler=1.7.0 clang=18.1.5 "
                            "mesa-libgl-devel-cos6-x86_64=11.0.7  mesa-libgl-cos6-x86_64=11.0.7 "
                            "mesa-dri-drivers-cos6-x86_64=11.0.7")

            elif cuda_version == "11.8":
                tensorflow = "2.12"
                req_file = os.path.join("requirements", "tensorflow_2_12_requirements.txt")
                command += "conda create -y -n flexutils-tensorflow " \
                           "-c conda-forge python=3.9 cudatoolkit=11.8 cudatoolkit-dev pyyaml"
            elif cuda_version == "11.2":
                tensorflow = "2.11"
                req_file = os.path.join("requirements", "tensorflow_2_11_requirements.txt")
                command += "conda create -y -n flexutils-tensorflow " \
                           "-c conda-forge python=3.8 cudatoolkit=11.2 cudnn=8.1.0 cudatoolkit-dev pyyaml"
            else:
                tensorflow = "2.3"
                req_file = os.path.join("requirements", "tensorflow_2_3_requirements.txt")
                command += "conda create -y -n flexutils-tensorflow " \
                           "-c conda-forge python=3.8 cudatoolkit=10.1 cudnn=7 cudatoolkit-dev pyyaml"
        else:
            tensorflow = "None"
            req_file = None
            command = None

        return req_file, condabin_path, command, cuda_version, tensorflow

    def runCondaInstallation(self, condaBin=None):
        # Check conda is in PATH
        if condaBin is None:
            try:
                subprocess.check_call("conda", shell=True, stdout=subprocess.PIPE)
                self.print_flush("Conda found in PATH")
            except:
                raise Exception("Conda not found in PATH \n "
                                "Installation will be aborted \n"
                                "Install Conda and/or add it to the PATH variable and try to install again "
                                "this package with 'pip install tensorflow-toolkit'")
        else:
            self.print_flush(f"Using provided Conda: {condaBin}")

        req_file, condabin_path, install_conda_command, cuda_version, tensorflow = self.condaInstallationCommands(condabin_path=condaBin)

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
            if tensorflow == "2.12":
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
        if tensorflow >= "2.15" or tensorflow == "None":
            commands = []
        elif tensorflow == "2.12":
            commands = ['eval "$(%s shell.bash hook) "' % condabin_path,
                        'conda activate flexutils-tensorflow ',
                        'mkdir -p $CONDA_PREFIX/etc/conda/activate.d ',
                        'echo \'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))\''
                        ' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh ',
                        'echo \'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib\' '
                        '>> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh ',
                        'echo \'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\' >> '
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
                        'echo \'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\' >> '
                        '$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh ',
                        'mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice ',
                        'cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/'
                        ]
        commands = "&&".join(commands)
        subprocess.check_call(commands, shell=True)
        self.print_flush("...done")


if __name__ == '__main__':

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--condaBin', type=str, required=False)
    args = parser.parse_args()

    Installation().runCondaInstallation(args.condaBin)

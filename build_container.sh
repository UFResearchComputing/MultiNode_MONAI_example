#!/bin/bash

# Script to build a singularity container based on NGC PyTorch docker image
# and install MONAI together with other dependencies in the container on UF HiperGator.
# 
# 2021/09, Huiwen Ju, hju@nvidia.com

#SBATCH -J build_container
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=8
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3gb
#SBATCH --exclusive
#SBATCH -t 08:00:00
date;hostname;pwd

module load singularity

# Build a singularity container based on NGC PyTorch docker image in your blue directory. 
# Alter `/blue/vendor-nvidia/hju/pyt21.07/` to your blue directory path.  
# New versions of NGC PyTorch are released monthly, https://ngc.nvidia.com/catalog/containers/nvidia:pytorch 
singularity build --sandbox /blue/vendor-nvidia/hju/pyt21.07/ docker://nvcr.io/nvidia/pytorch:21.07-py3

# Install MONAI in the container. 
# New versions of MONAI are released frequently https://docs.monai.io/en/latest/whatsnew.html
singularity exec --writable /blue/vendor-nvidia/hju/pyt_monai_tutorial/ pip3 install monai==0.6

# Install all dependencies required by the MONAI tutorial scripts
# MONAI tutorial scripts https://github.com/Project-MONAI/tutorials
singularity exec --writable /blue/vendor-nvidia/hju/pyt_monai_tutorial/ pip3 install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt
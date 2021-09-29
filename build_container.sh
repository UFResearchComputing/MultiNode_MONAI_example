#!/bin/bash

# Script to build a singularity container based on NGC PyTorch docker image
# and install MONAI together with other dependencies in the container on UF HiPerGator.
# 
# September 2021, Huiwen Ju, hju@nvidia.com
# September 2021, updated by Matt Gitzendanner, magitz@ufl.edu

#SBATCH --job-name build_container
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time 01:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user magitz@ufl.edu
#SBATCH --output=container_build.%j.out

date;hostname;pwd

module load singularity

echo "Building singularity container based on NGC PyTorch docker image the current directory."
# New versions of NGC PyTorch are released monthly, https://ngc.nvidia.com/catalog/containers/nvidia:pytorch 
singularity build --sandbox pyt21.07/ docker://nvcr.io/nvidia/pytorch:21.07-py3

echo "Adding UFRC filesystem paths to container"
singularity exec pyt21.07 mkdir -p /blue /orange /scratch/local

echo "Installing MONAI in the container." 
# New versions of MONAI are released frequently https://docs.monai.io/en/latest/whatsnew.html
singularity exec --writable pyt21.07 pip3 install monai==0.6

ehco "Installing dependencies required by the MONAI tutorial scripts"
# MONAI tutorial scripts https://github.com/Project-MONAI/tutorials
singularity exec --writable pyt21.07 pip3 install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt

echo "Finshed building the container."
date
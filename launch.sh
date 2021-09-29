#!/bin/bash

# Script to launch a multi-node pytorch.distributed training run on UF HiPerGator's AI partition,
# a SLURM cluster using Singularity as container runtime.
# 
# This script uses `pt_multinode_helper_funcs.sh` and `run_on_node.sh`.
#
# If launch with torch.distributed.launch, 
#       set #SBATCH --ntasks=--nodes
#       set #SBATCH --ntasks-per-node=1  
#       set #SBATCH --gpus=total number of processes to run on all nodes
#       set #SBATCH --gpus-per-task=--gpus / --ntasks  
#       modify `LAUNCH_CMD` in `run_on_node.sh` to launch with torch.distributed.launch
      
# If launch without torch.distributed.launch,
#       set #SBATCH --ntasks=total number of processes to run on all nodes
#       set #SBATCH --ntasks-per-node=--ntasks/--nodes    
#       set #SBATCH --gpus=--ntasks     
#       set #SBATCH --gpus-per-task=1
#       modify `LAUNCH_CMD` in `run_on_node.sh` to launch without torch.distributed.launch

# (c) 2021, Brian J. Stucky, UF Research Computing
# 2021/09, modified by Huiwen Ju, hju@nvidia.com

# Resource allocation.
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=multinode_pytorch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your email>
#SBATCH --nodes=2               # How many DGX nodes? Each has 8 A100 GPUs
#SBATCH --ntasks=16             # How many tasks? One per GPU
#SBATCH --ntasks-per-node=8     # Split 8 per node for the 8 GPUs
#SBATCH --gpus=16               # Total GPUs
#SBATCH --gpus-per-task=1       # 1 GPU per task
#SBATCH --cpus-per-task=4       # How many CPU cores per task, upto 16 for 8 tasks per node
#SBATCH --mem=96gb              # CPU memory per node--up to 1TB (Not GPU memory--that is 80GB per A100 GPU)
#SBATCH --partition=hpg-ai      # Specify the HPG AI partition
# Enable the following to limit the allocation to a single SU.
## SBATCH --constraint=su7
#SBATCH --time=48:00:00
#SBATCH --output=train.%j.out

export NCCL_DEBUG=INFO

# Training command specification: training_script -args.
# TRAINING_CMD="unet_training_ddp_slurm_torchlaunch.py" # See notes above on other changes needed
# TRAINING_CMD="unet_training_ddp_slurm.py"
TRAINING_CMD="unet_training_workflows_slurm.py"

# Python location (if not provided, system default will be used).
# Here we run within a MONAI Singularity container based on NGC PyTorch container,
# see `build_container.sh` to build a MONAI Singularity container.
PYTHON_PATH="singularity exec --nv pyt21.07 python3"       

# Location of the PyTorch launch utilities, i.e. `pt_multinode_helper_funcs.sh` & `run_on_node.sh`.
PT_LAUNCH_UTILS_PATH="./"
source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"

init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

PT_LAUNCH_SCRIPT="run_on_node.sh"
echo "Running \"$TRAINING_CMD\" on each node..."

srun --unbuffered "$PT_LAUNCH_SCRIPT" "${PT_LAUNCH_UTILS_PATH}" "$TRAINING_CMD" "$PYTHON_PATH"

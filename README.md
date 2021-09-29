# Running multi-GPU, multi-node applications on HiPerGator AI under SLURM

![UF Research Computing logo](images/ufrc_logo.png)

This repository provides examples of setting up muti-GPU, muti-node jobs on HiPerGator AI under SLURM.
 > Before attempting to run applications across a large number of GPUs, it is important to do some testing to make sure this is needed in the first place. The A100 GPUs on HiPerGator AI are very powerful GPUs and your application may not need multiple GPUs.

The scripts were developed with Research Computing and NVIDIA staff and are examples intended to be modified for your own work. This README attempts to outline the process and document what is needed to get the example running. 

## Setting up the Singularity container with the software environment

The example uses a [Singularity](https://sylabs.io/singularity/) container with a customized environment. Building this container is the first step, and once built, the container can be used for further analyses.

The base container image comes from the [NVIDIA NGC](https://ngc.nvidia.com/catalog) repository which provides many optimized ready-to-run containers.

The [build_container.sh](build_container.sh) script is a SLURM submit script to download and build the container for this example. Note that the Singularity build process does not need GPUs.

### Base container

For this tutorial, everything is expected to be in the same folder. On HiPerGator, this folder should be located in your folder on the `/blue` or `/red` filesystems (not `/orange`).

For this example we will use the PyTorch 21.07 base container, building that with the command: `singularity build --sandbox pyt21.07/ docker://nvcr.io/nvidia/pytorch:21.07-py3`

### Add in MONAI and dependancies

In addition to the base container, for this example, we need the [MONAI package](https://monai.io/) and dependancies. These are added to the container we built above (called `pyt21.07`) using the following commands in the script.

```bash
singularity exec --writable pyt21.07/ pip3 install monai==0.6
singularity exec --writable pyt21.07/ pip3 install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt
```

## Launch a training job

We will use the [launch.sh](launch.sh) script to submit the job and launch the multi-GPU multi-node training run. In the example here, we will use 16 GPUs on 2 nodes (each DGX node in HiPerGator AI has 8 A100 GPUs).

See the `#SBATCH` lines in [launch.sh](launch.sh) for explanations of the resource requests, GPUs per server, etc.


## Thanks to:
  * Huiwen Ju, hju@nvidia.com
  * Brian J. Stucky, UF Research Computing (formerly)
  * Matt Gitzendanner, UF Research Computing (some cleanup and documentation)

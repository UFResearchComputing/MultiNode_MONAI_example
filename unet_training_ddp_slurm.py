"""
This script is adpated from a MONAI tutorial script 
https://github.com/Project-MONAI/tutorials/blob/master/acceleration/distributed_training/unet_training_ddp.py
so that can be run mutli-node multi-gpu distributed training (using `torch.distributed.DistributedDataParallel` 
and native PyTorch training loop) on UF HiperGator's AI partition. To use PyTorch.ignite/MONAI workflows 
instead of native PyTorch training loop, see sample script `unet_training_workflows_slurm.py`.


How to launch this script:
- See sample SLURM batch script `launch.sh` (also helper scripts `run_on_node.sh` & `pt_multinode_helper_funcs.sh`), 
which can launch a PyTorch/MONAI script with or without using `torch.distributed.launch` on 
a SLURM cluster like HiperGator using Singularity as container runtime. This script is for launching without 
`torch.distributed.launch`. Sample script `unet_training_ddp_slurm_torchlaunch.py` is for launching with 
`torch.distributed.launch`. 
- See sample output unet_training_ddp_slurm.out


Steps to use `torch.distributed.DistributedDataParallel` in this script:
- Call `init_process_group` to initialize a process group. In this script, each process runs on one GPU.
  Here we use `NVIDIA NCCL` as the backend for optimized multi-GPU training performance and `init_method="env://"`
  to initialize a process group by environment variables.
- Create a `DistributedSampler` and pass it to DataLoader. Disable `shuffle` in DataLoader; instead, 
  shuffle data by turning on `shuffle` in `DistributedSampler` and calling `set_epoch` at the beginning 
  of each epoch before creating the DataLoader iterator.
- Wrap the model with `DistributedDataParallel` after moving to expected GPU.
- Call `destroy_process_group` after training finishes.


References:
torch.distributed: 
- https://pytorch.org/tutorials/beginner/dist_overview.html#
torch.distributed.launch: 
- https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md 
- https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
torch.distributed.DistributedDataParallel:
- https://pytorch.org/tutorials/intermediate/ddp_tutorial.html



There might be more multi-node multi-gpu support at the MONAI github mentioned above in the future, 
so please stay updated.

Huiwen Ju, hju@nvidia.com
2021/09
"""

import argparse
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import DataLoader, Dataset, create_test_image_3d, DistributedSampler
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
)


def train(args):
    # disable logging for processes except local_rank=0 on every node
    # if os.environ["LOCAL_RANK"] != 0:
    #     f = open(os.devnull, "w")
    #     sys.stdout = sys.stderr = f

    # parameters used to initialize the process group
    # env_dict = {
    #     key: os.environ[key]
    #     for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    # }
    # print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    # initialize a process group, every GPU runs in a process
    # (all processes connects to the master, obtain information about the other processes, 
    # and finally handshake with them) 
    dist.init_process_group(backend="nccl", init_method="env://")

    # process rank=0 generates synthetic data
    if dist.get_rank() == 0 and not os.path.exists(args.dir):     
        # create 40 random image, mask pair for training
        print(f"[{dist.get_rank()}] generating synthetic data to {args.dir} (this may take a while)")
        os.makedirs(args.dir)
        # set random seed to generate same random data for every node
        np.random.seed(seed=0)
        for i in range(64):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"seg{i:d}.nii.gz"))

    # wait for process rank=0 to finish            
    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

    images = sorted(glob(os.path.join(args.dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(args.dir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms)
    # create a training data sampler
    train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    # in distributed training, `batch_size` is for each process, not the sum for all processes
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler,
    )

    # create UNet, DiceLoss and Adam optimizer  
    device = torch.device(f'cuda:{os.environ["LOCAL_RANK"]}')     
    torch.cuda.set_device(device)
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)  
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[device])

    # start a typical PyTorch training
    epoch_loss_values = list()
    for epoch in range(5):
        print(f"[{dist.get_rank()}] " + "-" * 10 + f" epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"[{dist.get_rank()}] " + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"[{dist.get_rank()}] " + f"epoch {epoch + 1}, average loss: {epoch_loss:.4f}")
    print(f"[{dist.get_rank()}] " + f"train completed, epoch losses: {epoch_loss_values}")
    if dist.get_rank() == 0:
        # saving it in one process is sufficient, because all processes start from the same random parameters 
        # and are synchronized
        torch.save(model.state_dict(), "final_model.pth")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    args = parser.parse_args()

    # set env variables required by dist.init_process_group(init_method="env://")
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    # "PRIMARY_PORT" & "PRIMARY" are set by launch script
    os.environ["MASTER_PORT"] = os.environ["PRIMARY_PORT"] 
    os.environ["MASTER_ADDR"] = os.environ["PRIMARY"]

    train(args=args)

if __name__ == "__main__":
    main()
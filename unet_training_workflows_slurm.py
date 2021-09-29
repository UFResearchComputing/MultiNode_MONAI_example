"""
This script is adpated from a MONAI tutorial script 
https://github.com/Project-MONAI/tutorials/blob/master/acceleration/distributed_training/unet_training_workflows.py
so that can be run mutli-node multi-gpu distributed training (using `torch.distributed.DistributedDataParallel` 
and PyTorch-Ignite/MONAI training workflows) on UF HiperGator's AI partition. To use native PyTorch training loop 
instead of PyTorch.ignite/MONAI workflows, see sample scripts `unet_training_ddp_slurm.py` & 
`unet_training_ddp_slurm_torchlaunch.py`.


How to launch this script:
- See sample SLURM batch script `launch.sh` (also helper scripts `run_on_node.sh` & `pt_multinode_helper_funcs.sh`), 
which can launch a PyTorch/MONAI script with or without using `torch.distributed.launch` on 
a SLURM cluster like HiperGator using Singularity as container runtime. This script is for launching without 
`torch.distributed.launch`. Currently, this script can't be launched by `torch.distributed.launch` on HiperGator. 
- See sample output unet_training_workflows_slurm.out


Steps to use `torch.distributed.DistributedDataParallel` in this script:
- Call `init_process_group` to initialize a process group. In this script, each process runs on one GPU.
  Here we use `NVIDIA NCCL` as the backend for optimized multi-GPU training performance and `init_method="env://"`
  to initialize a process group by environment variables.
- Create a `DistributedSampler` and pass it to DataLoader.
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
import logging
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from ignite.metrics import Accuracy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import monai
from monai.data import DataLoader, Dataset, create_test_image_3d
from monai.engines import SupervisedTrainer
from monai.handlers import CheckpointSaver, LrScheduleHandler, StatsHandler, from_engine
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
)


def train(args):
    # output SupervisedTrainer's logging during training
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
 
    # parameters used to initialize the process group
    # env_dict = {
    #     key: os.environ[key]
    #     for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
    # }
    # print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    # initialize a process group, every GPU runs in a process
    # (all processes connects to the master, obtain information about the other processes, 
    # and finally handshake with them) 
    dist.init_process_group(backend="nccl", init_method="env://")

    # process rank=0 generates synthetic data
    if int(os.environ["RANK"]) == 0 and not os.path.exists(args.dir):
        # create 40 random image, mask paris for training
        print(f"[{os.environ['RANK']}] generating synthetic data to {args.dir} (this may take a while)")
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
    train_files = [{"image": img, "label": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms)
    # create a training data sampler
    train_sampler = DistributedSampler(train_ds)
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
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    torch.cuda.set_device(device)
    net = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = monai.losses.DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
    # wrap the model with DistributedDataParallel module
    net = DistributedDataParallel(net, device_ids=[device])

    train_post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )
    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
    ]
    if dist.get_rank() == 0:        
        train_handlers.extend(
            [
                StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
                CheckpointSaver(save_dir="./runs/", save_dict={"net": net, "opt": opt}, save_interval=2),
            ]
        )
    
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        amp=True if monai.utils.get_torch_version_tuple() >= (1, 6) else False,
        postprocessing=train_post_transforms,
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]), device=device)},
        train_handlers=train_handlers,
    )

    trainer.run()
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
# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
#
# This version trains ResNet-50 on TinyImageNet using DDP + AMP + StepLR.
# Data directory is taken from the DATA_DIR environment variable.
#
# Run with:
#   export DATA_DIR=/path/to/tinyimagenet
#   torchrun --nproc_per_node=NUM_GPUS ddp_tinyimagenet_resnet50.py --epochs 50

from datetime import datetime
import argparse
import os
import time
import gc

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.cuda import amp
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


# -------------------------
# Timer / memory helpers
# -------------------------

def start_timer():
    """Reset CUDA memory stats and return a start time in seconds."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
    return time.time()


def end_timer_and_print(start_time):
    """Sync CUDA, compute elapsed time, and print time + max memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated()
    else:
        max_mem = 0

    end_time = time.time()
    total_time = end_time - start_time

    print("Finished Training")
    print(f"Total execution time = {total_time:.3f} sec")
    print(f"Max memory used by tensors = {max_mem} bytes")


# -------------------------
# Utility: accuracy function
# -------------------------

def top1_accuracy(outputs, targets):
    """
    Compute top-1 accuracy (number of correct predictions in the batch).
    Returns an integer count (not percentage).
    """
    with torch.no_grad():
        _, pred = outputs.max(1)
        correct = pred.eq(targets).sum().item()
    return correct


# -------------------------
# Training / validation loop
# -------------------------

def train_and_validate(args):
    # Initialize distributed backend
    dist.init_process_group(backend="nccl")

    # Rank / local rank info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Make this process use the correct GPU
    torch.cuda.set_device(local_rank)

    # Basic flags
    is_main_process = rank == 0

    # For performance (not strictly deterministic)
    torch.backends.cudnn.benchmark = True

    # -------------------------
    # TensorBoard setup (rank 0 only)
    # -------------------------
    writer = None
    if is_main_process:
        log_dir = os.path.join(
            "runs",
            datetime.now().strftime("ddp_resnet50_%Y%m%d-%H%M%S"),
        )
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[Rank 0] TensorBoard logging to: {log_dir}")

    # -------------------------
    # Model, criterion, optimizer, scheduler, scaler
    # -------------------------
    num_classes = 200  # TinyImageNet has 200 classes
    model = resnet50(num_classes=num_classes)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # AMP GradScaler
    scaler = amp.GradScaler()

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])

    # -------------------------
    # Data: TinyImageNet train/val via ImageFolder
    # -------------------------

    data_dir = os.environ.get("DATA_DIR", "./tiny-imagenet-200")
    if is_main_process:
        print(f"[Rank 0] Using DATA_DIR={data_dir}")

    # Standard ImageNet-style transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # ImageFolder assumes folder-per-class structure
    train_dataset = ImageFolder(root=os.path.join(data_dir, "train"),
                                transform=train_transform)
    val_dataset = ImageFolder(root=os.path.join(data_dir, "val"),
                              transform=val_transform)

    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    # Dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if is_main_process:
        print(f"[Rank 0] Train size (global): {len(train_dataset)}")
        print(f"[Rank 0] Val size   (global): {len(val_dataset)}")
        print(f"[Rank 0] World size: {world_size}")
        print(f"[Rank 0] Per-GPU batch size: {args.batch_size}")
        print(f"[Rank 0] Global batch size: {args.batch_size * world_size}")

        start_time = start_timer()
    else:
        # still create variable for consistency; not used on non-main ranks
        start_time = None

    device = torch.device("cuda", local_rank)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(args.epochs):
        # Let the sampler shuffle differently each epoch
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()

        # ---- TRAIN ----
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()

            # Forward + loss under autocast (FP16)
            with amp.autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scaled backward + step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            epoch_train_loss += loss.item() * batch_size
            epoch_train_correct += top1_accuracy(outputs, labels)
            epoch_train_samples += batch_size

        # Local (per-rank) averages
        local_train_loss = epoch_train_loss / epoch_train_samples
        local_train_acc = 100.0 * epoch_train_correct / epoch_train_samples

        # ---- VALIDATION ----
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # autocast in eval as well
                with amp.autocast(dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                epoch_val_loss += loss.item() * batch_size
                epoch_val_correct += top1_accuracy(outputs, labels)
                epoch_val_samples += batch_size

        # -------------------------
        # Reduce metrics across all ranks to get GLOBAL stats
        # -------------------------
        t_loss_tensor = torch.tensor([epoch_train_loss], device=device)
        t_corr_tensor = torch.tensor([epoch_train_correct], device=device)
        t_samples_tensor = torch.tensor([epoch_train_samples], device=device)

        v_loss_tensor = torch.tensor([epoch_val_loss], device=device)
        v_corr_tensor = torch.tensor([epoch_val_correct], device=device)
        v_samples_tensor = torch.tensor([epoch_val_samples], device=device)

        dist.all_reduce(t_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_corr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_samples_tensor, op=dist.ReduceOp.SUM)

        dist.all_reduce(v_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(v_corr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(v_samples_tensor, op=dist.ReduceOp.SUM)

        global_train_loss = t_loss_tensor.item() / t_samples_tensor.item()
        global_train_acc = 100.0 * t_corr_tensor.item() / t_samples_tensor.item()
        global_val_loss = v_loss_tensor.item() / v_samples_tensor.item()
        global_val_acc = 100.0 * v_corr_tensor.item() / v_samples_tensor.item()

        epoch_end = time.perf_counter()
        epoch_duration = epoch_end - epoch_start
        global_train_samples = t_samples_tensor.item()
        throughput = global_train_samples / epoch_duration  # images per second (global)

        current_lr = scheduler.get_last_lr()[0]

        if is_main_process:
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Loss (train, val): {global_train_loss:.3f}, {global_val_loss:.3f} | "
                f"Accuracy (train, val): {global_train_acc:.2f}%, {global_val_acc:.2f}%"
                f"Throughput: {throughput:.1f} img/s"
            )

            if writer is not None:
                writer.add_scalar("Loss/train", global_train_loss, epoch)
                writer.add_scalar("Loss/val", global_val_loss, epoch)
                writer.add_scalar("Accuracy/train", global_train_acc, epoch)
                writer.add_scalar("Accuracy/val", global_val_acc, epoch)
                writer.add_scalar("Throughput/images_per_sec", throughput, epoch)
                writer.add_scalar("LR", current_lr, epoch)

        scheduler.step()

    # Final timing & memory print on rank 0
    if is_main_process:
        end_timer_and_print(start_time)
        if writer is not None:
            writer.close()


# -------------------------
# Main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DDP ResNet50 on TinyImageNet with AMP + StepLR")

    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="mini-batch size per GPU",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="number of DataLoader workers per process",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="SGD momentum",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-4,
        type=float,
        help="weight decay (L2 penalty)",
    )
    parser.add_argument(
        "--step-size",
        default=30,
        type=int,
        help="StepLR step_size (in epochs)",
    )
    parser.add_argument(
        "--gamma",
        default=0.1,
        type=float,
        help="StepLR decay factor",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_and_validate(args)


if __name__ == "__main__":
    main()
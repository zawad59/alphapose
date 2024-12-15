"""Script for multi-GPU training with memory optimizations and checkpoint resumption."""
import json
import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord

# Get the number of GPUs
num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

# Log system info
logger.info(f"Number of GPUs: {num_gpu}")
logger.info(f"Available CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(num_gpu):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    """Load model and optimizer states from checkpoint."""
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded.")

        # Load learning rate scheduler state
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            logger.info("Learning rate scheduler state loaded.")

        # Resume epoch
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    else:
        logger.info(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
        return 0


def save_checkpoint(epoch, model, optimizer, lr_scheduler, checkpoint_path):
    """Save model and optimizer states to a checkpoint."""
    # Create the directory if it doesn't exist
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logger.info(f"Created directory: {checkpoint_dir}")

    logger.info(f"Saving checkpoint for epoch {epoch} to '{checkpoint_path}'")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
    }, checkpoint_path)


def train(opt, train_loader, m, criterion, optimizer, writer):
    """Train the model for one epoch."""
    loss_logger = DataLogger()
    acc_logger = DataLogger()

    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')

    m.train()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, label_masks, _, bboxes) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda(non_blocking=True) for inp in inps]
        else:
            inps = inps.cuda(non_blocking=True)
        if isinstance(labels, list):
            labels = [label.cuda(non_blocking=True) for label in labels]
            label_masks = [label_mask.cuda(non_blocking=True) for label_mask in label_masks]
        else:
            labels = labels.cuda(non_blocking=True)
            label_masks = label_masks.cuda(non_blocking=True)

        output = m(inps)

        if cfg.LOSS.get('TYPE') == 'MSELoss':
            loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))
            acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))
        elif cfg.LOSS.get('TYPE') == 'Combined':
            # Combined loss logic (unchanged)
            pass
        else:
            loss = criterion(output, labels, label_masks)
            acc = calc_integral_accuracy(output, labels, label_masks, output_3d=False, norm_type=norm_type)

        batch_size = inps[0].size(0) if isinstance(inps, list) else inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            torch.cuda.empty_cache()

        opt.trainIters += 1
        if opt.board:
            board_writing(writer, loss_logger.avg, acc_logger.avg, opt.trainIters, 'Train')

        if opt.debug and not i % 10:
            debug_writing(writer, output, labels, inps, opt.trainIters)

        train_loader.set_description(
            f'loss: {loss_logger.avg:.8f} | acc: {acc_logger.avg:.4f}'
        )

    train_loader.close()
    return loss_logger.avg, acc_logger.avg


def main():
    """Main training loop."""
    logger.info('Initializing training process...')
    logger.info(cfg)

    # Model Initialization
    m = preset_model(cfg)
    m = nn.DataParallel(m).cuda()

    # Optimizer
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR
    )

    writer = SummaryWriter(f'.tensorboard/{opt.exp_id}-{cfg.FILE_NAME}')

    # Dataset and DataLoader
    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(16, cfg.TRAIN.BATCH_SIZE * num_gpu),  # Adjust batch size
        shuffle=True,
        num_workers=min(4, opt.nThreads),  # Adjust workers
        pin_memory=torch.cuda.is_available(),
    )

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    # Initialize trainIters
    opt.trainIters = 0

    # Checkpoint resumption
    checkpoint_path = './exp/{}/{}_checkpoint.pth'.format(opt.exp_id, cfg.FILE_NAME)
    start_epoch = load_checkpoint(checkpoint_path, m, optimizer, lr_scheduler)

    for epoch in range(start_epoch, cfg.TRAIN.END_EPOCH):
        opt.epoch = epoch
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {epoch} | LR: {current_lr} #############')

        # Training
        loss, miou = train(opt, train_loader, m, builder.build_loss(cfg.LOSS).cuda(), optimizer, writer)
        logger.epochInfo('Train', epoch, loss, miou)

        lr_scheduler.step()

        # Save checkpoint at the end of each epoch
        save_checkpoint(epoch, m, optimizer, lr_scheduler, checkpoint_path)


def preset_model(cfg):
    """Initialize the model."""
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading pretrained model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    else:
        logger.info('Initializing new model...')
        model._initialize()

    return model


if __name__ == "__main__":
    main()
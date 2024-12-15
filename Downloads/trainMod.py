"""Script for multi-gpu training."""
import json
import os
import sys
import csv
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

# List to store memory usage and epoch info
memory_usage_log = []

def train(opt, train_loader, m, criterion, optimizer):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')
    m.train()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, label_masks, _, bboxes) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
        else:
            inps = inps.cuda().requires_grad_()
        if isinstance(labels, list):
            labels = [label.cuda() for label in labels]
            label_masks = [label_mask.cuda() for label_mask in label_masks]
        else:
            labels = labels.cuda()
            label_masks = label_masks.cuda()

        output = m(inps)

        if cfg.LOSS.get('TYPE') == 'MSELoss':
            loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))
            acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))
        elif cfg.LOSS.get('TYPE') == 'Combined':
            if output.size()[1] == 68:
                face_hand_num = 42
            else:
                face_hand_num = 110

            output_body_foot = output[:, :-face_hand_num, :, :]
            output_face_hand = output[:, -face_hand_num:, :, :]
            num_body_foot = output_body_foot.shape[1]
            num_face_hand = output_face_hand.shape[1]

            label_masks_body_foot = label_masks[0]
            label_masks_face_hand = label_masks[1]

            labels_body_foot = labels[0]
            labels_face_hand = labels[1]

            loss_body_foot = 0.5 * criterion[0](output_body_foot.mul(label_masks_body_foot), labels_body_foot.mul(label_masks_body_foot))
            acc_body_foot = calc_accuracy(output_body_foot.mul(label_masks_body_foot), labels_body_foot.mul(label_masks_body_foot))

            loss_face_hand = criterion[1](output_face_hand, labels_face_hand, label_masks_face_hand)
            acc_face_hand = calc_integral_accuracy(output_face_hand, labels_face_hand, label_masks_face_hand, output_3d=False, norm_type=norm_type)

            loss_body_foot *= 100
            loss_face_hand *= 0.01

            loss = loss_body_foot + loss_face_hand
            acc = acc_body_foot * num_body_foot / (num_body_foot + num_face_hand) + acc_face_hand * num_face_hand / (num_body_foot + num_face_hand)
        else:
            loss = criterion(output, labels, label_masks)
            acc = calc_integral_accuracy(output, labels, label_masks, output_3d=False, norm_type=norm_type)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1

        train_loader.set_description(f'loss: {loss_logger.avg:.8f} | acc: {acc_logger.avg:.4f}')

    train_loader.close()

    return loss_logger.avg, acc_logger.avg


def main():
    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    m = preset_model(cfg)
    m = nn.DataParallel(m).cuda()

    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')
    if combined_loss:
        criterion1 = builder.build_loss(cfg.LOSS.LOSS_1).cuda()
        criterion2 = builder.build_loss(cfg.LOSS.LOSS_2).cuda()
        criterion = [criterion1, criterion2]
    else:
        criterion = builder.build_loss(cfg.LOSS).cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=opt.nThreads)

    opt.trainIters = 0

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        loss, miou = train(opt, train_loader, m, criterion, optimizer)
        logger.epochInfo('Train', opt.epoch, loss, miou)

        lr_scheduler.step()

        # Log memory usage
        current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        memory_usage_log.append({"epoch": opt.epoch, "memory_usage": current_memory})

    # Write memory usage to CSV
    memory_log_file = './exp/{}-{}/memory_usage.csv'.format(opt.exp_id, cfg.FILE_NAME)
    os.makedirs(os.path.dirname(memory_log_file), exist_ok=True)
    with open(memory_log_file, mode='w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=["epoch", "memory_usage"])
        csv_writer.writeheader()
        csv_writer.writerows(memory_usage_log)

    torch.save(m.module.state_dict(), './exp/{}-{}/final_DPG.pth'.format(opt.exp_id, cfg.FILE_NAME))


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == "__main__":
    main()

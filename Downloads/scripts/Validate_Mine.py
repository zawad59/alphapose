import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import time
import sys

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP, calc_integral_accuracy
from alphapose.utils.transforms import (flip, flip_heatmap,
                                        get_func_heatmap_to_coord)
from alphapose.utils.pPose_nms import oks_pose_nms

import numpy as np
from tqdm import tqdm
import time


parser = argparse.ArgumentParser(description='AlphaPose Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    required=True,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    default='0',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    default=32,
                    type=int)
parser.add_argument('--num_workers',
                    help='validation dataloader number of workers',
                    default=20,
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--oks-nms',
                    default=False,
                    dest='oks_nms',
                    help='use oks nms',
                    action='store_true')
parser.add_argument('--ppose-nms',
                    default=False,
                    dest='ppose_nms',
                    help='use pPose nms, recommended',
                    action='store_true')

opt = parser.parse_args()
cfg = update_config(opt.cfg)

gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")


def calculate_PCK(pred_keypoints, gt_keypoints, bboxes, threshold=0.5):
    """
    Calculate Percentage of Correct Keypoints (PCK).

    Parameters:
    -----------
    pred_keypoints : np.ndarray
        Predicted keypoints, shape (N, J, 2), where N is the number of samples, J is the number of joints.
    gt_keypoints : np.ndarray
        Ground-truth keypoints, shape (N, J, 2).
    bboxes : np.ndarray
        Bounding boxes, shape (N, 4), in format [x_min, y_min, width, height].
    threshold : float
        Distance threshold for considering a keypoint as correct.

    Returns:
    --------
    float
        PCK score.
    """
    # Convert bounding boxes from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max = x_min + width
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max = y_min + height

    if pred_keypoints.shape != gt_keypoints.shape:
        raise ValueError(f"Shape mismatch: pred_keypoints {pred_keypoints.shape}, gt_keypoints {gt_keypoints.shape}")

    num_samples, num_joints = pred_keypoints.shape[:2]
    correct = 0
    total = 0

    for i in range(num_samples):
        bbox_width = bboxes[i, 2] - bboxes[i, 0]
        bbox_height = bboxes[i, 3] - bboxes[i, 1]
        bbox_size = max(bbox_width, bbox_height)

        for j in range(num_joints):
            pred = pred_keypoints[i, j]
            gt = gt_keypoints[i, j]
            dist = np.linalg.norm(pred - gt)

            if dist < threshold * bbox_size:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0



def validate(model, heatmap_to_coord, batch_size=20, num_workers=4, threshold=0.5):
    """
    Validate the model and compute mAP and PCK metrics.

    Parameters:
    - model: PyTorch model to evaluate.
    - heatmap_to_coord: Function to convert heatmaps to coordinates.
    - batch_size: int, batch size for validation.
    - num_workers: int, number of data loader workers.
    - threshold: float, PCK distance threshold.

    Returns:
    - None
    """
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    eval_joints = det_dataset.EVAL_JOINTS

    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    model.eval()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    all_preds = []
    all_gt_keypoints = []
    all_bboxes = []
    latencies = []

    for batch in tqdm(det_loader, dynamic_ncols=True):
        start_time = time.time()
        inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds, gt_keypoints, label_masks = batch

        inps = inps.cuda()
        output = model(inps)

        # Convert heatmaps to keypoint coordinates
        batch_preds = []
        for i in range(output.size(0)):
            bbox = crop_bboxes[i].tolist()
            preds, _ = heatmap_to_coord(
                output[i][eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type
            )
            batch_preds.append(preds)

        all_preds.extend(batch_preds)
        all_gt_keypoints.extend(gt_keypoints.cpu().numpy())
        all_bboxes.extend(crop_bboxes.cpu().numpy())

        # Record latency
        end_time = time.time()
        latencies.append(end_time - start_time)

    # Compute Average Latency per Batch
    avg_latency = np.mean(latencies) * 1000  # Convert to milliseconds
    print(f"Average Latency per Batch: {avg_latency:.4f} ms")

    # Compute PCK
    try:
        pck_score = calculate_PCK(
            np.array(all_preds),
            np.array(all_gt_keypoints),
            np.array(all_bboxes),
            threshold=threshold
        )
        print(f"PCK@{threshold}: {pck_score:.4f}")
    except Exception as e:
        print(f"Error calculating PCK: {e}")
        pck_score = None

    # Save predictions to JSON file for COCO evaluation
    kpt_json = []
    for preds, bbox, img_id, score in zip(all_preds, all_bboxes, img_ids, scores):
        data = dict()
        data['bbox'] = bbox.tolist()
        data['image_id'] = int(img_id)
        data['score'] = float(score)
        data['keypoints'] = preds.reshape(-1).tolist()
        kpt_json.append(data)

    with open('./exp/json/validate_rcnn_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)

    # Compute mAP
    try:
        mAP = evaluate_mAP(
            './exp/json/validate_rcnn_kpt.json',
            ann_type='keypoints',
            ann_file=os.path.join(cfg.DATASET.TEST.ROOT, cfg.DATASET.TEST.ANN)
        )
        print(f"Validation mAP: {mAP}")
    except Exception as e:
        print(f"Error calculating mAP: {e}")

if __name__ == "__main__":
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint))

    m = torch.nn.DataParallel(m, device_ids=gpus).cuda()
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    with torch.no_grad():
        detbox_AP = validate(m, heatmap_to_coord, opt.batch, opt.num_workers)
    print('Validation mAP:', detbox_AP)

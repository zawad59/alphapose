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
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord

parser = argparse.ArgumentParser(description='AlphaPose Validate')
parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
parser.add_argument('--checkpoint', help='checkpoint file name', required=True, type=str)
parser.add_argument('--gpus', help='gpus', default='0', type=str)
parser.add_argument('--batch', help='validation batch size', default=32, type=int)
parser.add_argument('--num_workers', help='validation dataloader number of workers', default=20, type=int)
parser.add_argument('--flip-test', default=False, dest='flip_test', help='flip test', action='store_true')
parser.add_argument('--detector', dest='detector', help='detector name', default="yolo")
parser.add_argument('--oks-nms', default=False, dest='oks_nms', help='use oks nms', action='store_true')
parser.add_argument('--ppose-nms', default=False, dest='ppose_nms', help='use pPose nms, recommended', action='store_true')

opt = parser.parse_args()
cfg = update_config(opt.cfg)

gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")


def calculate_OKS(pred_keypoints, gt_keypoints, keypoint_visibility, area, sigma=0.5):
    """
    Calculate Object Keypoint Similarity (OKS).

    Parameters:
    - pred_keypoints : np.ndarray
        Predicted keypoints, shape (N, J, 2), where N is the number of samples, J is the number of joints.
    - gt_keypoints : np.ndarray
        Ground-truth keypoints, shape (N, J, 2).
    - keypoint_visibility : np.ndarray
        Visibility flags for keypoints, shape (N, J).
    - area : np.ndarray
        Object area, shape (N,).
    - sigma : float
        Standard deviation for OKS computation.

    Returns:
    - oks_scores : list of float
        OKS scores for all samples.
    """
    if pred_keypoints.shape != gt_keypoints.shape:
        raise ValueError(f"Shape mismatch: pred_keypoints {pred_keypoints.shape}, gt_keypoints {gt_keypoints.shape}")

    oks_scores = []

    for i in range(len(pred_keypoints)):
        pred = pred_keypoints[i]
        gt = gt_keypoints[i]
        vis = keypoint_visibility[i]
        a = area[i]

        if a == 0:  # Avoid division by zero for invalid bounding boxes
            oks_scores.append(0)
            continue

        dists = np.linalg.norm(pred - gt, axis=1)
        exp_term = -(dists ** 2) / (2 * (sigma ** 2) * a)
        oks = np.exp(exp_term)
        oks_scores.append(np.mean(oks[vis > 0]))  # Only include visible keypoints

    return oks_scores


def validate(model, heatmap_to_coord, batch_size=20, num_workers=4, sigma=0.5):
    """
    Validate the model and compute mAP, PCK, and OKS metrics.

    Parameters:
    - model: PyTorch model to evaluate.
    - heatmap_to_coord: Function to convert heatmaps to coordinates.
    - batch_size: int, batch size for validation.
    - num_workers: int, number of data loader workers.
    - sigma: float, standard deviation for OKS calculation.

    Returns:
    - avg_pck: Average PCK over all samples.
    - avg_oks: Average OKS over all samples.
    - mAP: Mean Average Precision for the validation set.
    """
    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    eval_joints = det_dataset.EVAL_JOINTS

    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )

    model.eval()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

    all_preds = []
    all_gt_keypoints = []
    all_visibility = []
    all_areas = []
    latencies = []

    for batch in tqdm(det_loader, dynamic_ncols=True):
        start_time = time.time()
        inps, crop_bboxes, bboxes, img_ids, scores, imghts, imgwds, gt_keypoints, label_masks = batch

        inps = inps.cuda()
        output = model(inps)

        # Convert heatmaps to keypoint coordinates
        batch_preds = []
        batch_visibility = label_masks.cpu().numpy()  # Assuming label_masks represent visibility
        batch_areas = (crop_bboxes[:, 2] - crop_bboxes[:, 0]) * (crop_bboxes[:, 3] - crop_bboxes[:, 1])  # Areas of bboxes

        for i in range(output.size(0)):
            bbox = crop_bboxes[i].tolist()
            preds, _ = heatmap_to_coord(
                output[i][eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type
            )
            batch_preds.append(preds)

        all_preds.extend(batch_preds)
        all_gt_keypoints.extend(gt_keypoints.cpu().numpy())
        all_visibility.extend(batch_visibility)
        all_areas.extend(batch_areas.cpu().numpy())

        # Record latency
        end_time = time.time()
        latencies.append(end_time - start_time)

    # Compute Average Latency per Batch
    avg_latency = np.mean(latencies) * 1000  # Convert to milliseconds
    print(f"Average Latency per Batch: {avg_latency:.4f} ms")

    # Compute OKS for all samples
    oks_scores = calculate_OKS(
        np.array(all_preds),
        np.array(all_gt_keypoints),
        np.array(all_visibility),
        np.array(all_areas),
        sigma=sigma
    )
    avg_oks = np.mean(oks_scores)
    print(f"Average OKS: {avg_oks:.4f}")  # <-- Print OKS

    # Compute PCK
    avg_pck = calculate_PCK(
        np.array(all_preds),
        np.array(all_gt_keypoints),
        np.array(all_bboxes),
        threshold=0.5
    )
    print(f"PCK@0.5: {avg_pck:.4f}")

    # Save predictions to JSON file for COCO evaluation
    kpt_json = []
    for preds, bbox, img_id, score in zip(all_preds, crop_bboxes.cpu().numpy(), img_ids, scores):
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
        mAP = None

    return avg_pck, avg_oks, mAP


def calculate_OKS(pred_keypoints, gt_keypoints, keypoint_visibility, area, sigma=0.5):
    """
    Calculate Object Keypoint Similarity (OKS).

    Parameters:
    - pred_keypoints : np.ndarray
        Predicted keypoints, shape (N, J, 2), where N is the number of samples, J is the number of joints.
    - gt_keypoints : np.ndarray
        Ground-truth keypoints, shape (N, J, 2).
    - keypoint_visibility : np.ndarray
        Visibility flags for keypoints, shape (N, J).
    - area : np.ndarray
        Object area, shape (N,).
    - sigma : float
        Standard deviation for OKS computation.

    Returns:
    - oks_scores : list of float
        OKS scores for all samples.
    """
    if pred_keypoints.shape != gt_keypoints.shape:
        raise ValueError(f"Shape mismatch: pred_keypoints {pred_keypoints.shape}, gt_keypoints {gt_keypoints.shape}")

    oks_scores = []

    for i in range(len(pred_keypoints)):
        pred = pred_keypoints[i]
        gt = gt_keypoints[i]
        vis = keypoint_visibility[i]
        a = area[i]

        if a == 0:  # Avoid division by zero for invalid bounding boxes
            oks_scores.append(0)
            continue

        dists = np.linalg.norm(pred - gt, axis=1)
        exp_term = -(dists ** 2) / (2 * (sigma ** 2) * a)
        oks = np.exp(exp_term)
        oks_scores.append(np.mean(oks[vis > 0]))  # Only include visible keypoints

    return oks_scores

def calculate_pck_for_image(pred_keypoints, gt_keypoints, bbox, threshold=0.5):
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    scale = np.sqrt(bbox_width ** 2 + bbox_height ** 2)  # Diagonal of the bounding box

    if scale <= 0:
        print(f"Invalid bounding box: {bbox}, Scale: {scale}")
        return 0  # Return 0 PCK for invalid bounding boxes

    # Calculate the Euclidean distances between predicted and ground truth keypoints
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)

    # Check if distances are within the threshold (scale * threshold)
    correct_keypoints = distances < (threshold * scale)

    # Calculate the PCK for this image (mean of correct keypoints)
    return np.mean(correct_keypoints)


if __name__ == "__main__":
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint, map_location=opt.device))

    m = torch.nn.DataParallel(m, device_ids=gpus).cuda()
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    with torch.no_grad():
        detbox_AP = validate(m, heatmap_to_coord, opt.batch, opt.num_workers)
    print('Validation mAP:', detbox_AP)

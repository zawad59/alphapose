"""Validation script with latency, PcK, and OKH scores."""
import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import time  # Added for latency measurement
import csv

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP  # Use mAP as provided
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.metrics import calculate_PcK, calculate_OKS

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

# Initialize latency log and summation variable
latency_log_file = './exp/json/inference_latency.csv'
os.makedirs(os.path.dirname(latency_log_file), exist_ok=True)
total_inference_time = 0.0

def validate(m, batch_size=20, num_workers=20):
    global total_inference_time  # To track the cumulative inference time

    det_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    all_pred_keypoints = []
    all_gt_keypoints = []
    all_keypoint_visibility = []
    all_areas = []  # For OKH calculation

    with open(latency_log_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["image_id", "inference_time"])  # CSV header

        for i, (inps, crop_bboxes, gt_keypoints, img_ids, scores, imghts, imgwds) in enumerate(tqdm(det_loader)):
            start_time = time.time()  # Start time

            # Forward pass to get output heatmaps
            output = m(inps.cuda())  # Shape: [B, J, H, W] (Batch, Joints, Heatmap Height, Heatmap Width)

            batch_pred_keypoints = []
            for b in range(output.shape[0]):  # Iterate through batch
                keypoint_heatmaps = output[b].cpu().numpy()  # Shape: [J, H, W]
                pred_coords = []
                for joint_idx in range(keypoint_heatmaps.shape[0]):  # Iterate over joints
                    heatmap = keypoint_heatmaps[joint_idx]  # Shape: [H, W]
                    coord = heatmap_to_coord(
                        heatmap[None], crop_bboxes[b].cpu().numpy(), hm_shape=hm_size, norm_type=norm_type
                    )
                    # Flatten the coordinate to (2,)
                    coord = np.array(coord).squeeze()
                    if coord.shape == (2,):  # Validate that coord has a valid shape
                        pred_coords.append(coord)
                    else:
                        print(f"Skipping invalid coordinate shape: {coord.shape}")

                if len(pred_coords) == keypoint_heatmaps.shape[0]:  # Ensure all joints are processed
                    pred_coords = np.array(pred_coords).reshape(-1, 2)  # Shape: [J, 2]
                    batch_pred_keypoints.append(pred_coords)

            if len(batch_pred_keypoints) != output.shape[0]:
                print(f"Warning: Skipping batch {i} due to inconsistent predictions")
                continue

            # Append batch predictions
            all_pred_keypoints.extend(batch_pred_keypoints)

            # Append ground truth keypoints, ensuring shape compatibility
            gt_keypoints_array = gt_keypoints.cpu().numpy()  # Shape: [B, J, 2] or similar
            if gt_keypoints_array.ndim == 3:  # Ensure [B, J, 2]
                all_gt_keypoints.extend(gt_keypoints_array)
            else:
                print(f"Skipping batch {i}: Invalid GT keypoint shape {gt_keypoints_array.shape}")

            # Append visibility scores and area
            all_keypoint_visibility.extend(scores.cpu().numpy())
            areas = (crop_bboxes[:, 2] - crop_bboxes[:, 0]) * (crop_bboxes[:, 3] - crop_bboxes[:, 1])
            all_areas.extend(areas.cpu().numpy())

            end_time = time.time()  # End time
            inference_time = end_time - start_time
            total_inference_time += inference_time

            # Log latency
            for idx in range(len(img_ids)):
                csv_writer.writerow([int(img_ids[idx].item()), inference_time])

    # Calculate PcK and OKH scores
    PcK_score = calculate_PcK(
        np.array(all_pred_keypoints), np.array(all_gt_keypoints), threshold=0.5
    )

    OKH_score = calculate_OKS(
        np.array(all_pred_keypoints), np.array(all_gt_keypoints),
        np.array(all_keypoint_visibility), np.array(all_areas), sigma=0.5
    )

    return evaluate_mAP('./exp/json/validate_rcnn_kpt.json', ann_type='keypoints',
                        ann_file=os.path.join(cfg.DATASET.TEST.ROOT, cfg.DATASET.TEST.ANN)), PcK_score, OKH_score

if __name__ == "__main__":
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint))
    m = torch.nn.DataParallel(m, device_ids=gpus).cuda()

    with torch.no_grad():
        detbox_AP, PcK_score, OKH_score = validate(m, opt.batch, opt.num_workers)

    print(f'##### det box: {detbox_AP} mAP #####')
    print(f'##### PcK Score: {PcK_score} | OKH Score: {OKH_score} #####')
    print(f'Total Inference Time: {total_inference_time:.4f} seconds')

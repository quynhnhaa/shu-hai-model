import sys
sys.path.append(".")

import os
import numpy as np
import argparse
from tqdm import tqdm
from dataset import BratsDataset
from config import config
from utils import combine_labels_predicting
from scipy.spatial import KDTree

def init_args():
    parser = argparse.ArgumentParser(description="Evaluate model predictions from saved .npy files.")
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing the predicted probability map .npy files.')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt', help='File to save the evaluation results.')
    parser.add_argument('-i', '--image_shape', type=int,  nargs='+', help='The shape of input tensor for dataset config', default=[128, 192, 160])
    return parser.parse_args()

def dice_score(pred, target, smooth=1e-5):
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2. * intersection + smooth) / (union + smooth)

def hausdorff_distance_95(pred, target):
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return np.nan

    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan

    # Use KDTree for memory-efficient nearest neighbor search
    # Directed Hausdorff from pred to target
    tree_target = KDTree(target_points)
    dist_pred_to_target, _ = tree_target.query(pred_points, k=1)
    hd95_pred_to_target = np.percentile(dist_pred_to_target, 95)

    # Directed Hausdorff from target to pred
    tree_pred = KDTree(pred_points)
    dist_target_to_pred, _ = tree_pred.query(target_points, k=1)
    hd95_target_to_pred = np.percentile(dist_target_to_pred, 95)

    return max(hd95_pred_to_target, hd95_target_to_pred)

def get_tumor_regions(segmentation):
    wt = (segmentation == 1) | (segmentation == 2) | (segmentation == 4)
    tc = (segmentation == 1) | (segmentation == 4)
    et = (segmentation == 4)
    return wt, tc, et

def evaluate(args):
    # --- Basic Configs ---
    config["image_shape"] = args.image_shape
    config["input_shape"] = tuple([1] + [config["nb_channels"]] + list(config["image_shape"]))
    config["data_path"] = "/kaggle/input/data-npy2" # This needs to be correct
    config["seg_label"] = None # Ensure dataset provides all 3 labels
    config["VAE_enable"] = False

    # --- Get Patient List ---
    test_list_path = os.path.join(config["base_path"], 'test_list.txt')
    with open(test_list_path, 'r') as f:
        patient_list = f.read().splitlines()
    print(f"Found {len(patient_list)} patients for evaluation in {os.path.basename(test_list_path)}")

    # --- Prepare Ground Truth Label Loader ---
    print("Preparing ground truth label loader...")
    config["validation_patients"] = patient_list
    label_dataset = BratsDataset(phase="validate", config=config)
    patient_name_to_idx = {name: i for i, name in enumerate(label_dataset.patient_names)}
    print("Label loader ready.")

    dice_scores = {"wt": [], "tc": [], "et": []}
    hd95_scores = {"wt": [], "tc": [], "et": []}

    # --- Evaluation Loop ---
    print(f"Evaluating predictions from: {args.pred_dir}")
    for patient_name in tqdm(patient_list):
        # Load prediction
        pred_path = os.path.join(args.pred_dir, patient_name + ".npy")
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file not found for {patient_name}, skipping.")
            continue
        
        probsMap_array = np.load(pred_path)
        preds_binary = (probsMap_array > 0.5).astype(np.uint8)
        pred_combined = combine_labels_predicting(preds_binary)

        # Get ground truth on-the-fly
        patient_idx = patient_name_to_idx.get(patient_name)
        if patient_idx is None:
            print(f"Warning: Label index not found for {patient_name}, skipping.")
            continue
        
        _, label_numpy = label_dataset[patient_idx]
        label_combined = combine_labels_predicting(label_numpy.squeeze())

        # Get regions for pred and label
        pred_wt, pred_tc, pred_et = get_tumor_regions(pred_combined)
        label_wt, label_tc, label_et = get_tumor_regions(label_combined)

        # Calculate and store metrics
        dice_scores["wt"].append(dice_score(pred_wt, label_wt))
        dice_scores["tc"].append(dice_score(pred_tc, label_tc))
        dice_scores["et"].append(dice_score(pred_et, label_et))

        hd95_scores["wt"].append(hausdorff_distance_95(pred_wt, label_wt))
        hd95_scores["tc"].append(hausdorff_distance_95(pred_tc, label_tc))
        hd95_scores["et"].append(hausdorff_distance_95(pred_et, label_et))

    # --- Report and Save Results ---
    with open(args.output_file, 'w') as f:
        f.write("---" + " Evaluation Results ---" + "\n")
        print("\n---" + " Evaluation Results ---")
        for region in ["wt", "tc", "et"]:
            avg_dice = np.nanmean(dice_scores[region])
            avg_hd95 = np.nanmean(hd95_scores[region])
            
            result_line1 = f"Region: {region.upper()}"
            result_line2 = f"  Average Dice Score: {avg_dice:.4f}"
            result_line3 = f"  Average 95% Hausdorff Distance: {avg_hd95:.4f}"
            result_line4 = "------------------------"
            
            print(result_line1)
            print(result_line2)
            print(result_line3)
            print(result_line4)
            
            f.write(result_line1 + '\n')
            f.write(result_line2 + '\n')
            f.write(result_line3 + '\n')
            f.write(result_line4 + '\n')
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    args = init_args()
    evaluate(args)
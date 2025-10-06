"""
@author: Cline
@time: 2025-01-06
Prediction script with Dice score and Hausdorff distance calculation for BraTS dataset
"""
import sys
import os
import numpy as np
import nibabel as nib
import argparse
import torch
from tqdm import tqdm
from dataset import BratsDataset
from config import config
from pandas import read_csv
from utils import combine_labels_predicting, dim_recovery
from scipy.spatial.distance import directed_hausdorff
import scipy.ndimage as ndimage

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--num_gpu', type=int, help='Number of GPUs', default=1)
    parser.add_argument('-s', '--save_folder', type=str, help='Saved model folder', default='saved_pth')
    parser.add_argument('-f', '--checkpoint_file', type=str, help='Model checkpoint file', default='model_best.pth')
    parser.add_argument('--model_path', type=str, default=None, help='Full path to model file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for predictions')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=None, help='End index')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    return parser.parse_args()

def calculate_hausdorff_distance(pred, gt, percentile=95):
    """
    Calculate Hausdorff distance between prediction and ground truth
    """
    pred = pred > 0.5  # Binarize prediction
    gt = gt > 0.5     # Binarize ground truth

    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    elif np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.inf

    # Get coordinates of foreground voxels
    pred_coords = np.argwhere(pred)
    gt_coords = np.argwhere(gt)

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.inf

    # Calculate directed Hausdorff distances
    dist1 = directed_hausdorff(pred_coords, gt_coords)[0]
    dist2 = directed_hausdorff(gt_coords, pred_coords)[0]

    # Return the maximum of the two directed distances (standard Hausdorff)
    hausdorff_dist = max(dist1, dist2)
    return hausdorff_dist

def calculate_metrics_for_patient(pred_array, gt_array):
    """
    Calculate Dice score and Hausdorff distance for WT, TC, ET
    """
    # Extract individual tumor regions from prediction and ground truth
    # pred_array shape: (H, W, D) with values 0, 1, 2, 4
    # gt_array shape: (H, W, D) with values 0, 1, 2, 4

    # Whole Tumor (WT): labels 1, 2, 4
    pred_wt = ((pred_array == 1) | (pred_array == 2) | (pred_array == 4))
    gt_wt = ((gt_array == 1) | (gt_array == 2) | (gt_array == 4))

    # Tumor Core (TC): labels 1, 4
    pred_tc = ((pred_array == 1) | (pred_array == 4))
    gt_tc = ((gt_array == 1) | (gt_array == 4))

    # Enhancing Tumor (ET): label 4
    pred_et = (pred_array == 4)
    gt_et = (gt_array == 4)

    metrics = {}

    # Calculate Dice scores
    metrics['dice_wt'] = dice_coefficient_single_label(pred_wt, gt_wt)
    metrics['dice_tc'] = dice_coefficient_single_label(pred_tc, gt_tc)
    metrics['dice_et'] = dice_coefficient_single_label(pred_et, gt_et)

    # Calculate Hausdorff distances
    metrics['hausdorff_wt'] = calculate_hausdorff_distance(pred_wt, gt_wt)
    metrics['hausdorff_tc'] = calculate_hausdorff_distance(pred_tc, gt_tc)
    metrics['hausdorff_et'] = calculate_hausdorff_distance(pred_et, gt_et)

    return metrics

def dice_coefficient_single_label(y_pred, y_truth, eps=1e-8):
    """
    Calculate Dice coefficient for single label
    """
    intersection = np.sum(y_pred * y_truth) + eps / 2
    union = np.sum(y_pred) + np.sum(y_truth) + eps
    dice = 2 * intersection / union
    return dice

def init_model_from_states(config, args):
    """Initialize model from saved state"""
    print("Loading model...")

    # Import here to avoid circular imports
    from nvnet import NvNet

    model = NvNet(config=config)

    # Set up GPU
    if args.num_gpu > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    # Load checkpoint
    checkpoint = torch.load(config['saved_model_path'], map_location='cpu')
    state_dict = checkpoint["state_dict"]

    # Handle data parallel loading
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module." in k:
            new_state_dict[k] = v
        else:
            new_state_dict["module." + k] = v

    model.load_state_dict(new_state_dict)
    return model

def predict_and_evaluate(name_list, model, config, args):
    """Run prediction and calculate metrics"""
    model.eval()

    # Set up output directory
    if args.output_dir:
        config["prediction_dir"] = args.output_dir
    else:
        config["prediction_dir"] = os.path.join(config["base_path"], "pred", "metrics_evaluation")

    os.makedirs(config["prediction_dir"], exist_ok=True)

    # Initialize metrics collection
    all_metrics = []

    for patient_filename in tqdm(name_list, desc="Processing patients"):
        try:
            # Load test data
            file_path = os.path.join('/kaggle/input/data-npy2', patient_filename + ".npy")
            if not os.path.exists(file_path):
                print(f"Warning: Data file not found for {patient_filename}")
                continue

            imgs_npy = np.load(file_path)

            # Prepare input data (remove segmentation mask if present)
            if imgs_npy.shape[0] == 5:  # 4 modalities + segmentation
                input_data = imgs_npy[:4]  # Take only first 4 channels
            else:
                input_data = imgs_npy

            # Add batch dimension
            input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)

            if args.num_gpu > 0:
                input_tensor = input_tensor.cuda()

            # Run prediction
            with torch.no_grad():
                outputs = model(input_tensor)

            # Process outputs
            output_array = outputs.cpu().numpy()
            pred_probs = output_array[0]  # Remove batch dimension

            # Apply threshold and get predictions
            pred_binary = (pred_probs > 0.5).astype(float)

            # Load ground truth
            seg_path = os.path.join('/kaggle/input/data-npy', patient_filename, patient_filename + '_seg.nii')
            if not os.path.exists(seg_path):
                print(f"Warning: Ground truth not found for {patient_filename}")
                continue

            gt_nii = nib.load(seg_path)
            gt_array = np.array(gt_nii.dataobj)

            # Calculate metrics
            patient_metrics = calculate_metrics_for_patient(pred_binary, gt_array)
            all_metrics.append(patient_metrics)

            # Print individual results
            print(f"\nPatient {patient_filename}:")
            print(f"  Dice WT: {patient_metrics['dice_wt']:.4f}, TC: {patient_metrics['dice_tc']:.4f}, ET: {patient_metrics['dice_et']:.4f}")
            print(f"  Hausdorff WT: {patient_metrics['hausdorff_wt']:.4f}, TC: {patient_metrics['hausdorff_tc']:.4f}, ET: {patient_metrics['hausdorff_et']:.4f}")

        except Exception as e:
            print(f"Error processing patient {patient_filename}: {str(e)}")
            continue

    # Calculate and print average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [patient[key] for patient in all_metrics if patient[key] != np.inf]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = np.inf

        print("\n" + "="*70)
        print("AVERAGE METRICS ACROSS ALL PATIENTS:")
        print("="*70)
        print(f"Dice WT:  {avg_metrics['dice_wt']:.4f}")
        print(f"Dice TC:  {avg_metrics['dice_tc']:.4f}")
        print(f"Dice ET:  {avg_metrics['dice_et']:.4f}")
        print(f"Hausdorff WT (95%):  {avg_metrics['hausdorff_wt']:.4f}")
        print(f"Hausdorff TC (95%):  {avg_metrics['hausdorff_tc']:.4f}")
        print(f"Hausdorff ET (95%):  {avg_metrics['hausdorff_et']:.4f}")
        print("="*70)

        # Save metrics to file
        metrics_file = os.path.join(config["prediction_dir"], "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write("Evaluation Metrics for BraTS Dataset\n")
            f.write("="*50 + "\n")
            f.write(f"Number of patients evaluated: {len(all_metrics)}\n")
            f.write(f"Average Dice WT: {avg_metrics['dice_wt']:.4f}\n")
            f.write(f"Average Dice TC: {avg_metrics['dice_tc']:.4f}\n")
            f.write(f"Average Dice ET: {avg_metrics['dice_et']:.4f}\n")
            f.write(f"Average Hausdorff WT: {avg_metrics['hausdorff_wt']:.4f}\n")
            f.write(f"Average Hausdorff TC: {avg_metrics['hausdorff_tc']:.4f}\n")
            f.write(f"Average Hausdorff ET: {avg_metrics['hausdorff_et']:.4f}\n")

        print(f"\nDetailed metrics saved to: {metrics_file}")
    else:
        print("No patients were successfully evaluated.")

def main():
    args = init_args()

    # Set up configuration
    config["cuda_devices"] = args.num_gpu > 0
    config["batch_size"] = args.batch_size
    config["image_shape"] = [128, 192, 160]  # Default shape
    config["input_shape"] = (args.batch_size, 4, 128, 192, 160)  # 4 channels for MRI modalities
    config["activation"] = "relu"  # Default activation function

    # Model path logic
    if args.model_path:
        config['saved_model_path'] = args.model_path
    else:
        config["checkpoint_path"] = os.path.join(config["base_path"], "models", args.save_folder)
        config['saved_model_path'] = os.path.join(config["checkpoint_path"], args.checkpoint_file)

    # Load model
    model = init_model_from_states(config, args)

    # Load test patient list
    test_list_path = '../test_list.txt'
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r') as f:
            val_list = f.read().splitlines()
    else:
        print(f"Error: {test_list_path} not found!")
        return

    # Apply start/end indices
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(val_list)
    start = max(0, min(start, len(val_list)))
    end = max(start, min(end, len(val_list)))
    val_list = val_list[start:end]

    print(f"Evaluating {len(val_list)} patients...")

    # Run prediction and evaluation
    predict_and_evaluate(val_list, model, config, args)

if __name__ == "__main__":
    main()

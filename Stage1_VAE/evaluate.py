"""
Evaluation script for BraTS2020 Stage 1 VAE model.
Calculates Dice score and 95% Hausdorff distance for WT, TC, ET.
"""
import sys
sys.path.append(".")

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
from scipy.spatial.distance import cdist

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--num_gpu', type=int, help='Can be 0, 1, 2, 4', default=1)
    parser.add_argument('-s', '--save_folder', type=str, help='The folder of the saved model', default='saved_pth')
    parser.add_argument('-f', '--checkpoint_file', type=str, help='name of the saved pth file', default='best_model.pth')
    parser.add_argument('-i', '--image_shape', type=int,  nargs='+', help='The shape of input tensor', default=[128, 192, 160])
    parser.add_argument('--model_path', type=str, default=None, help='Full path to the saved pth file')
    return parser.parse_args()

def dice_score(pred, target, smooth=1e-5):
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2. * intersection + smooth) / (union + smooth)

def hausdorff_distance_95(pred, target):
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan

    # Directed Hausdorff distance from pred to target
    dist_pred_to_target = cdist(pred_points, target_points).min(axis=1)
    hd95_pred_to_target = np.percentile(dist_pred_to_target, 95)

    # Directed Hausdorff distance from target to pred
    dist_target_to_pred = cdist(target_points, pred_points).min(axis=1)
    hd95_target_to_pred = np.percentile(dist_target_to_pred, 95)

    return max(hd95_pred_to_target, hd95_target_to_pred)


def get_tumor_regions(segmentation):
    wt = (segmentation == 1) | (segmentation == 2) | (segmentation == 4)
    tc = (segmentation == 1) | (segmentation == 4)
    et = (segmentation == 4)
    return wt, tc, et

def init_model_from_states(config):
    print("Init model...")
    # This assumes NvNet is the model used. If not, this needs to be adjusted.
    from nvnet import NvNet
    model = NvNet(config=config)
    if config["cuda_devices"] is not None:
        if torch.cuda.device_count() > 1 and args.num_gpu > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()

    if not os.path.exists(config['saved_model_path']):
        raise Exception(f"Invalid model path: {config['saved_model_path']}")

    checkpoint = torch.load(config['saved_model_path'], map_location='cpu')
    
    try:
        # Try loading from a 'state_dict' key, common in training checkpoints
        state_dict = checkpoint["state_dict"]
    except KeyError:
        # If no 'state_dict', assume the checkpoint is the model itself
        state_dict = checkpoint

    if config.get("load_from_data_parallel", True):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module." not in k:
                name = "module." + k
            else:
                name = k
            if "vae" not in name: # Exclude VAE path for pure segmentation
                 new_state_dict[name] = v
        # When loading a DataParallel model, the keys are prefixed with 'module.'
        # If the current setup is not DataParallel, we might need to remove this prefix.
        if not (torch.cuda.device_count() > 1 and args.num_gpu > 1):
             new_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}

        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    return model


def evaluate(name_list, model):
    model.eval()
    config["validation_patients"] = name_list
    
    data_set = BratsDataset(phase="validate", config=config)
    evaluation_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=config["batch_size"], shuffle=False)

    dice_scores = {"wt": [], "tc": [], "et": []}
    hd95_scores = {"wt": [], "tc": [], "et": []}

    predict_process = tqdm(evaluation_loader)
    for inputs, labels in predict_process:
        if config["cuda_devices"] is not None:
            inputs = inputs.type(torch.FloatTensor).cuda()
        
        with torch.no_grad():
            outputs = model(inputs)

        # Process predictions
        output_array = outputs.cpu().numpy()[:, :3, :, :, :]
        preds_binary = (output_array > 0.5).astype(np.uint8)
        
        # Process labels
        labels_array = labels.numpy()

        for i in range(inputs.size(0)):
            pred_combined = combine_labels_predicting(preds_binary[i].squeeze())
            label_combined = combine_labels_predicting(labels_array[i].squeeze())

            pred_wt, pred_tc, pred_et = get_tumor_regions(pred_combined)
            label_wt, label_tc, label_et = get_tumor_regions(label_combined)

            # Calculate metrics
            dice_scores["wt"].append(dice_score(pred_wt, label_wt))
            dice_scores["tc"].append(dice_score(pred_tc, label_tc))
            dice_scores["et"].append(dice_score(pred_et, label_et))

            hd95_scores["wt"].append(hausdorff_distance_95(pred_wt, label_wt))
            hd95_scores["tc"].append(hausdorff_distance_95(pred_tc, label_tc))
            hd95_scores["et"].append(hausdorff_distance_95(pred_et, label_et))

    # Calculate and print average scores
    print("\n--- Evaluation Results ---")
    for region in ["wt", "tc", "et"]:
        avg_dice = np.nanmean(dice_scores[region])
        avg_hd95 = np.nanmean(hd95_scores[region])
        print(f"Region: {region.upper()}")
        print(f"  Average Dice Score: {avg_dice:.4f}")
        print(f"  Average 95% Hausdorff Distance: {avg_hd95:.4f}")
        print("--------------------")


if __name__ == "__main__":
    args = init_args()
    num_gpu = args.num_gpu

    config["cuda_devices"] = True
    if num_gpu == 0:
        config["cuda_devices"] = None
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpu))

    config["batch_size"] = args.num_gpu if args.num_gpu > 0 else 1
    config["image_shape"] = args.image_shape
    
    if args.model_path:
        config['saved_model_path'] = args.model_path
    else:
        config["checkpoint_path"] = os.path.join(config["base_path"], "models", args.save_folder)
        config['saved_model_path'] = os.path.join(config["checkpoint_path"], args.checkpoint_file)

    config["VAE_enable"] = False
    config["num_labels"] = 3
    config["attention"] = "att" in config.get("checkpoint_file", os.path.basename(config['saved_model_path']))
    config["concat"] = "cat" in config.get("checkpoint_file", os.path.basename(config['saved_model_path']))
    config["activation"] = "sin" if "sin" in config.get("checkpoint_file", os.path.basename(config['saved_model_path'])) else "relu"
    
    config["data_path"] = "/kaggle/input/data-npy2"
    
    model = init_model_from_states(config)

    with open(os.path.join(config["base_path"], 'test_list.txt'), 'r') as f:
        test_list = f.read().splitlines()
    
    


    evaluate(test_list, model)

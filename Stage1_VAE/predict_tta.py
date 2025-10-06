"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
Modified to perform evaluation with TTA.
"""
import sys
sys.path.append(".")

import os
import numpy as np
import argparse
import torch
from tqdm import tqdm
from dataset import BratsDataset
from config import config
from utils import combine_labels_predicting
from scipy.spatial.distance import cdist

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--num_gpu', type=int, help='Can be 0, 1, 2, 4', default=1)
    parser.add_argument('-s', '--save_folder', type=str, help='The folder of the saved model', default='saved_pth')
    parser.add_argument('-f', '--checkpoint_file', type=str, help='name of the saved pth file', default='best_model.pth')
    parser.add_argument('-i', '--image_shape', type=int,  nargs='+', help='The shape of input tensor', default=[128, 192, 160])
    parser.add_argument('-t', '--tta', type=bool, help='Whether to implement test-time augmentation;', default=False)
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
    dist_pred_to_target = cdist(pred_points, target_points).min(axis=1)
    hd95_pred_to_target = np.percentile(dist_pred_to_target, 95)
    dist_target_to_pred = cdist(target_points, pred_points).min(axis=1)
    hd95_target_to_pred = np.percentile(dist_target_to_pred, 95)
    return max(hd95_pred_to_target, hd95_target_to_pred)

def get_tumor_regions(segmentation):
    wt = (segmentation == 1) | (segmentation == 2) | (segmentation == 4)
    tc = (segmentation == 1) | (segmentation == 4)
    et = (segmentation == 4)
    return wt, tc, et

args = init_args()
num_gpu = args.num_gpu
tta = args.tta
config["cuda_devices"] = True
if num_gpu == 0:
    config["cuda_devices"] = None
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpu))

config["batch_size"] = args.num_gpu if args.num_gpu > 0 else 1
config["image_shape"] = args.image_shape

if args.model_path:
    config['saved_model_path'] = args.model_path
    config["checkpoint_file"] = os.path.basename(args.model_path)
else:
    config["checkpoint_file"] = args.checkpoint_file
    config["checkpoint_path"] = os.path.join(config["base_path"], "models", args.save_folder)
    config['saved_model_path'] = os.path.join(config["checkpoint_path"], config["checkpoint_file"])

config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["VAE_enable"] = False
config["num_labels"] = 3
config["load_from_data_parallel"] = True

config["activation"] = "relu"
if "sin" in config["checkpoint_file"]:
    config["activation"] = "sin"
config["concat"] = "cat" in config["checkpoint_file"]
config["attention"] = "att" in config["checkpoint_file"]

# Hardcoded data path, ensure this is correct for your environment
config["data_path"] = "/kaggle/input/data-npy2"

from nvnet import NvNet

def init_model_from_states(config):
    print("Init model...")
    model = NvNet(config=config)
    if config["cuda_devices"] is not None:
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
    
    if not os.path.exists(config['saved_model_path']):
        raise Exception(f"Invalid model path: {config['saved_model_path']}")

    checkpoint = torch.load(config['saved_model_path'], map_location='cpu')
    try:
        state_dict = checkpoint["state_dict"]
    except KeyError:
        state_dict = checkpoint

    if config.get("load_from_data_parallel", True):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module." not in k:
                name = "module." + k
            else:
                name = k
            if "vae" not in name:
                 new_state_dict[name] = v
        if not (torch.cuda.device_count() > 1 and args.num_gpu > 1):
             new_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    return model

def test_time_flip_recovery(imgs_array, tta_idx):
    if tta_idx == 0: return imgs_array
    if tta_idx == 1: return imgs_array[:, :, ::-1, :, :]
    if tta_idx == 2: return imgs_array[:, :, :, ::-1, :]
    if tta_idx == 3: return imgs_array[:, :, :, :, ::-1]
    if tta_idx == 4: return imgs_array[:, :, ::-1, ::-1, :]
    if tta_idx == 5: return imgs_array[:, :, ::-1, :, ::-1]
    if tta_idx == 6: return imgs_array[:, :, :, ::-1, ::-1]
    if tta_idx == 7: return imgs_array[:, :, ::-1, ::-1, ::-1]
    return imgs_array

def evaluate(name_list, model):
    model.eval()

    dice_scores = {"wt": [], "tc": [], "et": []}
    hd95_scores = {"wt": [], "tc": [], "et": []}

    # Create a dataset for validation to get the ground truth labels
    config["validation_patients"] = name_list
    label_dataset = BratsDataset(phase="validate", config=config)
    labels_dict = {label_dataset.patient_list[i]: label_dataset[i][1] for i in range(len(label_dataset))}

    tmp_dir = "../tmp_result_{}".format(config["checkpoint_file"][:-4])
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # TTA Prediction Phase
    tta_idx_limit = 8 if tta else 1
    for tta_idx in range(tta_idx_limit):
        config["tta_idx"] = tta_idx
        if tta:
            print(f"-- Starting TTA flip index: {tta_idx} --")
        # Use phase='test' for prediction dataset as it doesn't load labels, making it faster
        pred_dataset = BratsDataset(phase="test", config=config)
        pred_loader = torch.utils.data.DataLoader(dataset=pred_dataset, batch_size=config["batch_size"], shuffle=False)
        
        predict_process = tqdm(pred_loader)
        for idx, inputs in enumerate(predict_process):
            inputs = inputs.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                outputs = model(inputs)
            
            output_array = outputs.cpu().numpy()[:, :3, :, :, :]
            output_array = test_time_flip_recovery(output_array, config["tta_idx"])

            for i in range(inputs.size(0)):
                file_idx = idx * config["batch_size"] + i
                if file_idx < len(name_list):
                    patient_filename = name_list[file_idx]
                    np.save(os.path.join(tmp_dir, f"flip_{tta_idx}_{patient_filename}.npy"), output_array[i])

    # Evaluation Phase
    print("\n-- Aggregating TTA results and evaluating --")
    for patient_filename in tqdm(name_list):
        flip_arrays = []
        for tta_idx in range(tta_idx_limit):
            flip_path = os.path.join(tmp_dir, f"flip_{tta_idx}_{patient_filename}.npy")
            if os.path.exists(flip_path):
                flip_arrays.append(np.load(flip_path))
        
        if not flip_arrays:
            continue

        probsMap_array = np.array(flip_arrays).mean(axis=0)
        preds_binary = (probsMap_array > 0.5).astype(np.uint8)
        pred_combined = combine_labels_predicting(preds_binary)

        label_tensor = labels_dict[patient_filename]
        label_array = label_tensor.numpy().squeeze()
        label_combined = combine_labels_predicting(label_array)

        pred_wt, pred_tc, pred_et = get_tumor_regions(pred_combined)
        label_wt, label_tc, label_et = get_tumor_regions(label_combined)

        dice_scores["wt"].append(dice_score(pred_wt, label_wt))
        dice_scores["tc"].append(dice_score(pred_tc, label_tc))
        dice_scores["et"].append(dice_score(pred_et, label_et))

        hd95_scores["wt"].append(hausdorff_distance_95(pred_wt, label_wt))
        hd95_scores["tc"].append(hausdorff_distance_95(pred_tc, label_tc))
        hd95_scores["et"].append(hausdorff_distance_95(pred_et, label_et))

    os.system("rm -r " + tmp_dir)

    # Print Results
    print("\n--- Evaluation Results ---")
    for region in ["wt", "tc", "et"]:
        avg_dice = np.nanmean(dice_scores[region])
        avg_hd95 = np.nanmean(hd95_scores[region])
        print(f"Region: {region.upper()}")
        print(f"  Average Dice Score: {avg_dice:.4f}")
        print(f"  Average 95% Hausdorff Distance: {avg_hd95:.4f}")
        print("------------------------")

if __name__ == "__main__":
    model = init_model_from_states(config)

    # Use test_list.txt for evaluation
    test_list_path = os.path.join(config["base_path"], 'test_list.txt')
    with open(test_list_path, 'r') as f:
        test_list = f.read().splitlines()
    
    print(f"Found {len(test_list)} patients for evaluation in {os.path.basename(test_list_path)}")

    evaluate(test_list, model)

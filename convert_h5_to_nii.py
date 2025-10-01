#!/usr/bin/env python
import os
import glob
import re
import subprocess
import sys
from collections import defaultdict
from tqdm import tqdm

# Function to install a package using pip
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Please install it manually using 'pip install {package}'")
        sys.exit(1)

# Check for and install required libraries
try:
    import pandas as pd
except ImportError:
    print("pandas not found. Installing...")
    install("pandas")
    import pandas as pd

try:
    import h5py
except ImportError:
    print("h5py not found. Installing...")
    install("h5py")
    import h5py

try:
    import nibabel as nib
except ImportError:
    print("nibabel not found. Installing...")
    install("nibabel")
    import nibabel

import numpy as np

# --- CONFIGURATION ---
# Path to the directory containing the H5 files
H5_DATA_PATH = "/kaggle/input/brats2020-training-data/BraTS2020_training_data/content/data/"
# Path to the name mapping CSV file
NAME_MAPPING_PATH = os.path.join(H5_DATA_PATH, "name_mapping.csv")
# Path to the final output directory required by the project
OUTPUT_DATA_PATH = "data/MICCAI_BraTS2020_TrainingData/"
# Names for the output modalities, assuming this order in the H5 file's 'image' dataset
MODALITY_NAMES = ['t1', 't1ce', 't2', 'flair']

def main():
    """
    Main function to convert H5 slice data to NIfTI volumes.
    """
    print("Starting data conversion process from .h5 slices to .nii.gz volumes.")

    # 1. Create the main output directory
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    print(f"Output directory created at: {OUTPUT_DATA_PATH}")

    # 2. Load the official BraTS name mappings
    try:
        name_mapping_df = pd.read_csv(NAME_MAPPING_PATH)
        brats_ids = name_mapping_df['BraTS_2020_subject_ID'].tolist()
    except FileNotFoundError:
        print(f"ERROR: Name mapping file not found at {NAME_MAPPING_PATH}")
        sys.exit(1)

    # 3. Find and group all H5 slice files
    h5_files = glob.glob(os.path.join(H5_DATA_PATH, "volume_*_slice_*.h5"))
    if not h5_files:
        print(f"ERROR: No .h5 files found in {H5_DATA_PATH}. Please check the path.")
        sys.exit(1)

    volumes = defaultdict(list)
    print("Grouping H5 slices by volume...")
    for f in h5_files:
        match = re.search(r"volume_(\d+)_slice_(\d+)\.h5", os.path.basename(f))
        if match:
            volume_id = int(match.group(1))
            slice_id = int(match.group(2))
            volumes[volume_id].append({"slice_id": slice_id, "path": f})

    print(f"Found {len(volumes)} volumes to process.")

    # 4. Loop through each volume, reconstruct, and save as NIfTI
    # Sort by volume_id to ensure order matches the CSV file
    sorted_volume_ids = sorted(volumes.keys())
    
    # We make an assumption that the volume IDs in the files correspond to the order in the CSV
    # Let's find the starting index of the volumes if it's not 0 or 1
    start_index_offset = 0
    if sorted_volume_ids and min(sorted_volume_ids) != 0 and min(sorted_volume_ids) != 1:
         # This dataset seems to start from volume_100, which maps to BraTS20_Training_001
         # The CSV has 369 entries. Let's assume the file IDs map to these.
         # This is a heuristic. If volume IDs are not continuous, this will fail.
         print(f"Warning: Volume IDs seem to start at {min(sorted_volume_ids)}. Assuming they map sequentially to BraTS IDs.")


    for i, volume_id in enumerate(tqdm(sorted_volume_ids, desc="Converting Volumes")):
        
        # Get the corresponding BraTS ID from the mapping file (assuming order matches)
        if i < len(brats_ids):
            brats_id = brats_ids[i]
        else:
            print(f"Warning: More volumes found ({len(sorted_volume_ids)}) than BraTS IDs in CSV ({len(brats_ids)}). Skipping volume {volume_id}.")
            continue

        # --- Reconstruct 3D volumes from slices ---
        slices_info = sorted(volumes[volume_id], key=lambda x: x['slice_id'])
        
        images_list, masks_list = [], []
        for slice_info in slices_info:
            with h5py.File(slice_info['path'], 'r') as f:
                # Transpose to (Channels, H, W)
                image = np.transpose(f['image'][:], (2, 0, 1))
                mask = np.transpose(f['mask'][:], (2, 0, 1))
            images_list.append(image)
            masks_list.append(mask)

        # Stack along the depth axis (D) to get (C, D, H, W)
        images_3d = np.stack(images_list, axis=1)
        masks_3d = np.stack(masks_list, axis=1)

        # --- Create patient-specific output directory ---
        patient_output_dir = os.path.join(OUTPUT_DATA_PATH, brats_id)
        os.makedirs(patient_output_dir, exist_ok=True)

        # --- Define a standard 1mm isotropic affine matrix ---
        # This assumes data is in radiological LPS orientation after processing
        affine = np.diag([-1, -1, 1, 1])

        # --- Save the 4 modality images ---
        for modality_idx, modality_name in enumerate(MODALITY_NAMES):
            # Get data for the current modality: (D, H, W)
            modality_data = images_3d[modality_idx]
            
            # Create NIfTI image object
            nifti_img = nib.Nifti1Image(modality_data, affine)
            
            # Save the file
            output_filename = os.path.join(patient_output_dir, f"{brats_id}_{modality_name}.nii.gz")
            nib.save(nifti_img, output_filename)

        # --- Convert 3-channel mask to single-channel segmentation file ---
        # Create an empty array for the single-channel mask
        seg_data = np.zeros(masks_3d.shape[1:], dtype=np.uint8)
        
        # Assign labels based on the BraTS annotation protocol and the project's expected order
        # masks_3d channel 0 -> ED (label 2)
        # masks_3d channel 1 -> NCR/NET (label 1)
        # masks_3d channel 2 -> ET (label 4)
        seg_data[masks_3d[0] == 1] = 2  # Peritumoral Edema
        seg_data[masks_3d[1] == 1] = 1  # Necrotic and Non-Enhancing Tumor
        seg_data[masks_3d[2] == 1] = 4  # GD-enhancing Tumor
        
        # Create NIfTI image for the segmentation
        nifti_seg = nib.Nifti1Image(seg_data, affine)
        
        # Save the file
        output_seg_filename = os.path.join(patient_output_dir, f"{brats_id}_seg.nii.gz")
        nib.save(nifti_seg, output_seg_filename)

    print("\nConversion complete!")
    print(f"All data has been converted and saved in the correct structure inside '{OUTPUT_DATA_PATH}'.")
    print("You can now run the 'normalization.py' script.")

if __name__ == "__main__":
    main()

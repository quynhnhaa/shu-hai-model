import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
# Define the base path for the data, which contains the mapping file.
# This makes the script independent of the config.py files.
DATA_ROOT = "data/MICCAI_BraTS2020_TrainingData"
MAPPING_FILE_PATH = os.path.join(DATA_ROOT, "name_mapping.csv")

# Define the split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15 # The rest will go to test

# Set a random seed for reproducibility
RANDOM_SEED = 42

def main():
    """
    Reads a list of patients, shuffles it, and splits it into
    train, validation, and test sets, saving the results to .txt files.
    """
    print("Starting patient list splitting process...")

    # 1. Read patient IDs from the name_mapping.csv file
    try:
        name_mapping_df = pd.read_csv(MAPPING_FILE_PATH)
        # Assuming the column with patient IDs for BraTS 2020 is named 'BraTS_2020_subject_ID'
        patient_ids = name_mapping_df['BraTS_2020_subject_ID'].dropna().tolist()
    except FileNotFoundError:
        print(f"ERROR: Mapping file not found at '{MAPPING_FILE_PATH}'.")
        print("Please make sure you have run the conversion script or placed the data correctly.")
        return
    except KeyError:
        print(f"ERROR: Column 'BraTS_2020_subject_ID' not found in '{MAPPING_FILE_PATH}'.")
        return

    if not patient_ids:
        print("No patient IDs found. Exiting.")
        return

    total_patients = len(patient_ids)
    print(f"Found {total_patients} total patients.")

    # 2. Shuffle the patient list
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(patient_ids)

    # 3. Calculate split indices
    train_end_idx = int(total_patients * TRAIN_RATIO)
    valid_end_idx = train_end_idx + int(total_patients * VALID_RATIO)

    # 4. Create the three lists
    train_list = patient_ids[:train_end_idx]
    valid_list = patient_ids[train_end_idx:valid_end_idx]
    test_list = patient_ids[valid_end_idx:]

    # 5. Write the lists to .txt files
    def write_list_to_file(filename, patient_list):
        with open(filename, 'w') as f:
            for item in patient_list:
                f.write(f"{item}\n")
        print(f"Created {filename} with {len(patient_list)} patients.")

    write_list_to_file('train_list.txt', train_list)
    write_list_to_file('valid_list.txt', valid_list)
    write_list_to_file('test_list.txt', test_list)

    print("\nSplitting process complete!")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_ROOT = "data/MICCAI_BraTS2020_TrainingData"
MAPPING_FILE_PATH = os.path.join(DATA_ROOT, "name_mapping.csv")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# The rest (0.15) will be for the test set

RANDOM_SEED = 42

def split_list(patient_list, train_ratio, valid_ratio):
    """Helper function to split a list into three parts."""
    total_patients = len(patient_list)
    train_end_idx = int(total_patients * train_ratio)
    valid_end_idx = train_end_idx + int(total_patients * valid_ratio)
    
    train = patient_list[:train_end_idx]
    valid = patient_list[train_end_idx:valid_end_idx]
    test = patient_list[valid_end_idx:]
    return train, valid, test

def write_list_to_file(filename, patient_list):
    """Helper function to write a list to a file."""
    with open(filename, 'w') as f:
        for item in patient_list:
            f.write(f"{item}\n")
    print(f"Created {filename} with {len(patient_list)} patients.")

def main():
    """
    Performs a stratified split of patients based on their grade (HGG/LGG)
    into train, validation, and test sets.
    """
    print("Starting stratified patient list splitting process...")

    # 1. Read patient IDs and their grades
    try:
        name_mapping_df = pd.read_csv(MAPPING_FILE_PATH)
        hgg_patients = name_mapping_df[name_mapping_df['Grade'] == 'HGG']['BraTS_2020_subject_ID'].dropna().tolist()
        lgg_patients = name_mapping_df[name_mapping_df['Grade'] == 'LGG']['BraTS_2020_subject_ID'].dropna().tolist()
    except FileNotFoundError:
        print(f"ERROR: Mapping file not found at '{MAPPING_FILE_PATH}'.")
        return
    except KeyError as e:
        print(f"ERROR: Column {e} not found in '{MAPPING_FILE_PATH}'.")
        return

    print(f"Found {len(hgg_patients)} HGG patients and {len(lgg_patients)} LGG patients.")

    # 2. Shuffle each group independently for randomness within strata
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(hgg_patients)
    np.random.shuffle(lgg_patients)

    # 3. Split each group into train, validation, and test sets
    hgg_train, hgg_valid, hgg_test = split_list(hgg_patients, TRAIN_RATIO, VALID_RATIO)
    lgg_train, lgg_valid, lgg_test = split_list(lgg_patients, TRAIN_RATIO, VALID_RATIO)

    # 4. Combine the stratified lists
    final_train_list = hgg_train + lgg_train
    final_valid_list = hgg_valid + lgg_valid
    final_test_list = hgg_test + lgg_test

    # 5. Shuffle the final combined lists to mix HGG and LGG cases during training
    np.random.shuffle(final_train_list)
    np.random.shuffle(final_valid_list)
    np.random.shuffle(final_test_list)

    # 6. Write the final lists to .txt files
    write_list_to_file('train_list.txt', final_train_list)
    write_list_to_file('valid_list.txt', final_valid_list)
    write_list_to_file('test_list.txt', final_test_list)

    print("\nStratified splitting process complete!")
    print(f"--- Train Set: {len(hgg_train)} HGG, {len(lgg_train)} LGG")
    print(f"--- Valid Set: {len(hgg_valid)} HGG, {len(lgg_valid)} LGG")
    print(f"--- Test Set : {len(hgg_test)} HGG, {len(lgg_test)} LGG")

if __name__ == "__main__":
    main()
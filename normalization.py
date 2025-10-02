import numpy as np
import SimpleITK as sitk
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import os
import time
import pandas as pd
import argparse
from Stage2_AttVAE.config import config


def init_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str, help='The data path')
    parser.add_argument('-y', '--year', type=int, help='s', default=2020)
    parser.add_argument('--start', type=int, default=0, help='Start index of patients to process')
    parser.add_argument('--end', type=int, default=None, help='End index of patients to process')

    return parser.parse_args()


def get_list_of_files(base_dir):
    list_of_lists = []
    for glioma_type in ['HGG', 'LGG']:
        current_directory = join(base_dir, glioma_type)
        patients = subfolders(current_directory, join=False)
        for p in patients:
            patient_directory = join(current_directory, p)
            t1_file = join(patient_directory, p + "_t1.nii")
            t1c_file = join(patient_directory, p + "_t1ce.nii")
            t2_file = join(patient_directory, p + "_t2.nii")
            flair_file = join(patient_directory, p + "_flair.nii")
            seg_file = join(patient_directory, p + "_seg.nii")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
            assert all((isfile(i) for i in this_case)), "some file is missing for patient %s; make sure the following " \
                                                        "files are there: %s" % (p, str(this_case))

            list_of_lists.append(this_case)

    print("Found {} patients".format(len(list_of_lists)))
    return list_of_lists


def get_list_of_files_2020(current_directory, patients, mode="training"):
    list_of_lists = []
    # patients = subfolders(current_directory, join=False)
    for p in patients:
        patient_directory = join(current_directory, p)
        t1_file = join(patient_directory, p + "_t1.nii")
        t1c_file = join(patient_directory, p + "_t1ce.nii")
        t2_file = join(patient_directory, p + "_t2.nii")
        flair_file = join(patient_directory, p + "_flair.nii")
        if mode == "training":
            seg_file = join(patient_directory, p + "_seg.nii")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
        else:
            this_case = [t1_file, t1c_file, t2_file, flair_file]
        assert all((isfile(i) for i in this_case)), "some file is missing for patient %s; make sure the following " \
                                                    "files are there: %s" % (p, str(this_case))

        list_of_lists.append(this_case)

    print("Found {} patients".format(len(list_of_lists)))
    return list_of_lists


def load_and_preprocess(case, patient_name, output_folder):
    # load SimpleITK Images
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from SimpleITK images
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)     #(5, 155, 240, 240)

    nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]  # 1488885;  # 1490852;  # 1492561;  #1495212

    # now normalize each modality with its mean and standard deviation (computed within the brain mask)
    for i in range(len(imgs_npy) - 1):   # 158
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0
        print(imgs_npy[i].mean())

    # now save as npy
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)

    print(patient_name)

def load_and_preprocess_val(case, patient_name, output_folder):

    # load images using nibabel
    imgs_nib = [nib.load(i) for i in case]
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from nib object
    # imgs_npy = [i.get_fdata() for i in imgs_nib]
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    # get affine information
    affines = [i.affine for i in imgs_nib]

    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)
    # (4, 155, 240, 240) for STik; (4, 240, 240, 155) for nil
    # get affine information
    affines = np.concatenate([i[None] for i in affines]).astype(np.float32)

    # now we create a brain mask that we use for normalization
    nonzero_masks = [i != 0 for i in imgs_npy]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]  # 1488885;  # 1490852;  # 1492561;  #1495212

    # now normalize each modality with its mean and standard deviation (computed within the brain mask)
    for i in range(len(imgs_npy)):
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0

    # now save as npy
    affine_output_folder = output_folder[:-4] + '/affine'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(affine_output_folder):
        os.mkdir(affine_output_folder)
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)
    # np.save(join(affine_output_folder, patient_name + ".npy"), affines)

    print(patient_name)

if __name__ == "__main__":

    args = init_args()

    # Set data path based on year
    if args.year == 2020:
        data_file_path = "/kaggle/input/data-npy"
    elif args.year == 2018:
        data_file_path = "data/MICCAI_BraTS_2018_Data_Training"
    elif args.year == 202001:
        data_file_path = "data/MICCAI_BraTS2020_ValidationData"
    elif args.year == 202002:
        data_file_path = "data/MICCAI_BraTS2020_TestingData"
    else:
        # Fallback to allow custom path
        data_file_path = args.data_path if args.data_path else "."

    npy_normalized_folder = join('/kaggle/working/shu-hai-model/data/MICCAI_BraTS2020_TrainingData', "npy")
    
    # Determine patient list
    patients = []
    if args.year == 2018:
        # For 2018, patient names are subfolders in HGG/LGG
        hgg_patients = subfolders(join(data_file_path, 'HGG'), join=False)
        lgg_patients = subfolders(join(data_file_path, 'LGG'), join=False)
        patients = hgg_patients + lgg_patients
    elif args.year == 2020:
        mapping_file_path = join(data_file_path, "name_mapping.csv")
        name_mapping = pd.read_csv(mapping_file_path)
        HGG = name_mapping.loc[name_mapping.Grade == "HGG", "BraTS_2020_subject_ID"].tolist()
        LGG = name_mapping.loc[name_mapping.Grade == "LGG", "BraTS_2020_subject_ID"].tolist()
        patients = HGG + LGG
    elif args.year == 202001:
        mapping_file_path = join(data_file_path, "name_mapping_validation_data.csv")
        name_mapping = pd.read_csv(mapping_file_path)
        patients = name_mapping["BraTS20ID"].tolist()
    elif args.year == 202002:
        mapping_file_path = join(data_file_path, "survival_evaluation.csv")
        name_mapping = pd.read_csv(mapping_file_path)
        patients = name_mapping["BraTS20ID"].tolist()

    # Slice patient list
    if args.end is not None:
        print(f"Processing patients from index {args.start} to {args.end}.")
        patients_to_process = patients[args.start:args.end]
    else:
        print(f"Processing patients from index {args.start} to the end.")
        patients_to_process = patients[args.start:]
    
    print(f"Found {len(patients)} total patients, processing {len(patients_to_process)} patients.")

    # Get full file paths for the selected patients
    list_of_lists = []
    if args.year == 2018:
        # The get_list_of_files function gets all files, so we need to re-implement the logic here
        # to only get the files for the selected patients.
        for p in patients_to_process:
            glioma_type = 'HGG' if p in hgg_patients else 'LGG'
            patient_directory = join(data_file_path, glioma_type, p)
            t1_file = join(patient_directory, p + "_t1.nii")
            t1c_file = join(patient_directory, p + "_t1ce.nii")
            t2_file = join(patient_directory, p + "_t2.nii")
            flair_file = join(patient_directory, p + "_flair.nii")
            seg_file = join(patient_directory, p + "_seg.nii")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
            assert all((isfile(i) for i in this_case)), f"Missing files for patient {p}"
            list_of_lists.append(this_case)
    elif args.year in [2020, 202001, 202002]:
        mode = "validation" if args.year in [202001, 202002] else "training"
        list_of_lists = get_list_of_files_2020(data_file_path, patients_to_process, mode=mode)

    # Start multiprocessing
    if patients_to_process:
        p = Pool(processes=8)
        t0 = time.time()
        print("Job starts for preprocessing...")
        
        if args.year in [202001, 202002]:
             p.starmap(load_and_preprocess_val, zip(list_of_lists, patients_to_process, [npy_normalized_folder] * len(patients_to_process)))
        else:
             p.starmap(load_and_preprocess, zip(list_of_lists, patients_to_process, [npy_normalized_folder] * len(patients_to_process)))

        print("Finished; costs {}s".format(time.time() - t0))
        p.close()
        p.join()
    else:
        print("No patients to process.")
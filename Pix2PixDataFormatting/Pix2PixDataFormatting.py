import os
import shutil
from pathlib import Path
import pydicom
from PIL import Image
import numpy as np
from image_combiner import combine_A_and_B

def convert_dcm_to_jpg(dcm_path, jpg_path, crop_top, crop_bottom):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array

    # Normalize the pixel intensity values
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_img = (img - min_val) / (max_val - min_val) * 255

    # Convert the normalized image to PIL Image
    pil_img = Image.fromarray(normalized_img.astype(np.uint8))

    # Convert the image mode to a supported format if it's not RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Initial crop based on the given crop boundaries
    original_width, original_height = pil_img.size
    desired_width, desired_height = 576, 404

    # Calculate center to crop around
    vertical_center = crop_top + (crop_bottom - crop_top) / 2
    horizontal_center = original_width / 2

    # Define initial crop box
    left = max(horizontal_center - desired_width / 2, 0)
    top = max(vertical_center - desired_height / 2, 0)
    right = left + desired_width
    bottom = top + desired_height

    # Adjust crop box if it exceeds original dimensions
    if right > original_width:
        right = original_width
        left = max(right - desired_width, 0)
    if bottom > original_height:
        bottom = original_height
        top = max(bottom - desired_height, 0)

    # Crop the image
    pil_img = pil_img.crop((left, top, right, bottom))

    # Pad the image if it's smaller than the desired dimensions
    padded_img = Image.new("RGB", (desired_width, desired_height), (0, 0, 0))
    padded_img.paste(pil_img, ((desired_width - pil_img.width) // 2, 
                               (desired_height - pil_img.height) // 2))

    padded_img.save(jpg_path)


def create_pix2pix_structure(ct_dir, mr_dir, dest_dir, split, patient_name):
    ct_folders = [folder for folder in ct_dir.glob("*") if folder.is_dir()]
    mr_folders = [folder for folder in mr_dir.glob("*") if folder.is_dir()]
    
    ct_folders.sort(key=lambda x: x.name)
    mr_folders.sort(key=lambda x: x.name)

    dest_split_dir_A = dest_dir / "A" / split
    dest_split_dir_B = dest_dir / "B" / split
    dest_split_dir_A.mkdir(parents=True, exist_ok=True)
    dest_split_dir_B.mkdir(parents=True, exist_ok=True)

    for folder_index, (ct_folder, mr_folder) in enumerate(zip(ct_folders, mr_folders)):
        ct_files = list(ct_folder.glob("*.dcm"))
        mr_files = list(mr_folder.glob("*.dcm"))

        # Extract the position values and sort them
        mr_positions = [(pydicom.dcmread(mr_file).ImagePositionPatient[2], mr_file) for mr_file in mr_files]
        mr_positions.sort(key=lambda x: x[0])

        # Reverse the list after sorting to ensure MR images are processed in reverse order
        mr_positions.reverse()

        # Map from original decimal positions to new integer image numbers in reversed order
        position_to_int_map = {position: index + 1 for index, (position, _) in enumerate(mr_positions)}
        
        # Sort and find the middle DICOM file for boundary calculation
        mr_positions.sort(key=lambda x: x[0])
        middle_mr_file = mr_files[len(mr_files) // 2]
        crop_top, crop_bottom = find_crop_boundaries(middle_mr_file)

        for ct_file, (position, mr_file) in zip(ct_files, mr_positions):
            ct_dcm = pydicom.dcmread(ct_file)
            
            ct_image_number = ct_dcm[0x0020, 0x0013].value
            # Use the reversed mapping for MR image numbers
            mr_image_number = position_to_int_map[position]
            
            ct_jpg_filename = f"{patient_name}_{folder_index}_{ct_image_number}.jpg"
            mr_jpg_filename = f"{patient_name}_{folder_index}_{mr_image_number}.jpg"
            ct_jpg_path = dest_split_dir_B / ct_jpg_filename
            mr_jpg_path = dest_split_dir_A / mr_jpg_filename
            convert_dcm_to_jpg(ct_file, ct_jpg_path, crop_top, crop_bottom)
            convert_dcm_to_jpg(mr_file, mr_jpg_path, crop_top, crop_bottom)
            
def find_crop_boundaries(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array

    # Apply threshold
    thresholded_img = img > 50

    # Find rows with pixels above the threshold
    rows = np.where(np.max(thresholded_img, axis=1))[0]

    if len(rows) == 0:  # If no rows found, return None
        return None, None

    top, bottom = rows[0], rows[-1]

    return top, bottom

def main():
    # Set the base directory path
    base_dir = Path("C:/Brendan/Data")

    # Get the list of patient directories
    patient_dirs = [dir for dir in base_dir.glob("Patient *") if dir.is_dir()]
    patient_dirs.sort(key=lambda x: x.name)

    # Define the split ratios for train, val, and test sets
    split_ratios = {"train": 4, "val": 1, "test": 1}

    # Initialize variables to keep track of the current split and patient index
    current_split = "train"
    current_patient_index = 0

    # Process each patient directory
    for patient_dir in patient_dirs:
        patient_name = patient_dir.name  # Extract the patient name
        ct_src_dir = patient_dir / "CT" / "CT"
        mr_src_dir = patient_dir / "MR" / "MR"
        dest_dir = base_dir / "Formatted"
        dest_dir_final = base_dir / "FormattedFinal"

        # Create the pix2pix data structure for CT and MR
        create_pix2pix_structure(ct_src_dir, mr_src_dir, dest_dir, current_split, patient_name)
        
        # Combine A and B folders
        combine_A_and_B(f'{dest_dir}\\A', f'{dest_dir}\\B', dest_dir_final, use_AB=False, no_multiprocessing=True)

        # Update the patient index and check if we need to switch to the next split
        current_patient_index += 1
        if current_patient_index == split_ratios[current_split]:
            current_patient_index = 0
            if current_split == "train":
                current_split = "val"
            elif current_split == "val":
                current_split = "test"

if __name__ == '__main__':
    main()
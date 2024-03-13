import os
import shutil
from pathlib import Path
import pydicom
from PIL import Image
import numpy as np
from image_combiner import combine_A_and_B

def convert_dcm_to_jpg(dcm_path, jpg_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array
    
    # Normalize the pixel intensity values
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_img = (img - min_val) / (max_val - min_val) * 255
    
    # Convert the normalized image to PIL Image
    pil_img = Image.fromarray(normalized_img.astype(np.uint8))
    
    # Convert the image mode to a supported format
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Check if the path is within a directory named "A" and apply initial cropping
    if 'A' in Path(jpg_path).parts:
        width, height = pil_img.size
        pil_img = pil_img.crop((0, 132, width, height - 40))
    
    final_width, final_height = 576, 404
    width, height = pil_img.size

    # Determine the operation: crop or pad
    if width > final_width or height > final_height:
        # Crop the image if it's too large
        # Calculate the new top left corner for cropping to keep the image centered
        left = (width - final_width) / 2
        top = (height - final_height) / 2
        pil_img = pil_img.crop((left, top, left + final_width, top + final_height))
    elif width < final_width or height < final_height:
        # Pad the image with black pixels if it's too small
        # Create a new image with the desired dimensions and black background
        new_img = Image.new('RGB', (final_width, final_height), (0, 0, 0))
        # Calculate the position to paste the original image to keep it centered
        left = (final_width - width) / 2
        top = (final_height - height) / 2
        new_img.paste(pil_img, (int(left), int(top)))
        pil_img = new_img
    
    pil_img.save(jpg_path)


def create_pix2pix_structure(ct_dir, mr_dir, dest_dir, split_ratios):
    ct_folders = [folder for folder in ct_dir.glob("*") if folder.is_dir()]
    mr_folders = [folder for folder in mr_dir.glob("*") if folder.is_dir()]
    
    ct_folders.sort(key=lambda x: x.name)
    mr_folders.sort(key=lambda x: x.name)

    split_keys = list(split_ratios.keys())
    split_values = list(split_ratios.values())

    start_idx = 0
    for split, ratio in zip(split_keys, split_values):
        dest_split_dir_A = dest_dir / "A" / split
        dest_split_dir_B = dest_dir / "B" / split
        dest_split_dir_A.mkdir(parents=True, exist_ok=True)
        dest_split_dir_B.mkdir(parents=True, exist_ok=True)

        end_idx = start_idx + ratio

        for folder_index, (ct_folder, mr_folder) in enumerate(zip(ct_folders[start_idx:end_idx], mr_folders[start_idx:end_idx]), start=start_idx):
            ct_files = list(ct_folder.glob("*.dcm"))
            mr_files = list(mr_folder.glob("*.dcm"))

            # Extract the position values and sort them
            mr_positions = [(pydicom.dcmread(mr_file).ImagePositionPatient[2], mr_file) for mr_file in mr_files]
            mr_positions.sort(key=lambda x: x[0])

            # Reverse the list after sorting to ensure MR images are processed in reverse order
            mr_positions.reverse()

            # Map from original decimal positions to new integer image numbers in reversed order
            position_to_int_map = {position: index + 1 for index, (position, _) in enumerate(mr_positions)}

            for ct_file, (position, mr_file) in zip(ct_files, mr_positions):
                ct_dcm = pydicom.dcmread(ct_file)
                mr_dcm = pydicom.dcmread(mr_file)
                
                ct_image_number = ct_dcm[0x0020, 0x0013].value
                # Use the reversed mapping for MR image numbers
                mr_image_number = position_to_int_map[position]
                
                ct_jpg_filename = f"{folder_index}_{ct_image_number}.jpg"
                mr_jpg_filename = f"{folder_index}_{mr_image_number}.jpg"
                ct_jpg_path = dest_split_dir_B / ct_jpg_filename
                mr_jpg_path = dest_split_dir_A / mr_jpg_filename
                convert_dcm_to_jpg(ct_file, ct_jpg_path)
                convert_dcm_to_jpg(mr_file, mr_jpg_path)

        start_idx = end_idx

def main():
    # Set the paths to your source and destination directories
    ct_src_dir = Path("C:/Patient 5/CT/CT")
    mr_src_dir = Path("C:/Patient 5/MR/MR")
    dest_dir = Path("C:/Patient 5/Formatted")
    dest_dir_final = Path("C:/Patient 5/FormattedFinal")

    # Define the split ratios for train, val, and test sets
    split_ratios = {"train": 3, "val": 1, "test": 1}

    # Create the pix2pix data structure for CT and MR
    create_pix2pix_structure(ct_src_dir, mr_src_dir, dest_dir, split_ratios)

    # Assuming dest_dir is the destination directory and the structure required by fold_A, fold_B, and fold_AB is already set up as needed.
    combine_A_and_B(f'{dest_dir}\\A', f'{dest_dir}\\B', dest_dir_final, use_AB=False, no_multiprocessing=True)

if __name__ == '__main__':
    main()
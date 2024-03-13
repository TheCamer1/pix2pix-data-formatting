import os
import numpy as np
import cv2
from multiprocessing import Pool

def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
    im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)

def combine_A_and_B(fold_A, fold_B, fold_AB, use_AB=False, no_multiprocessing=False):
    # Ensure fold_A exists
    if not os.path.isdir(fold_A):
        os.makedirs(fold_A, exist_ok=True)
    # Ensure fold_B exists
    if not os.path.isdir(fold_B):
        os.makedirs(fold_B, exist_ok=True)
    
    splits = os.listdir(fold_A)

    if not no_multiprocessing:
        pool = Pool()

    for sp in splits:
        img_fold_A = os.path.join(fold_A, sp)
        img_fold_B = os.path.join(fold_B, sp)

        # Create img_fold_A if it doesn't exist
        if not os.path.isdir(img_fold_A):
            os.makedirs(img_fold_A, exist_ok=True)
        # Create img_fold_B if it doesn't exist
        if not os.path.isdir(img_fold_B):
            os.makedirs(img_fold_B, exist_ok=True)

        img_list = os.listdir(img_fold_A)
        if use_AB:
            img_list = [img_path for img_path in img_list if '_A.' in img_path]

        img_fold_AB = os.path.join(fold_AB, sp)
        if not os.path.isdir(img_fold_AB):
            os.makedirs(img_fold_AB)

        for name_A in img_list:
            path_A = os.path.join(img_fold_A, name_A)
            if use_AB:
                name_B = name_A.replace('A_', 'B_')
            else:
                name_B = name_A
            path_B = os.path.join(img_fold_B, name_B)
            if os.path.isfile(path_A) and os.path.isfile(path_B):
                name_AB = name_A.replace('_A.', '.') if use_AB else name_A  # remove _A for AB
                path_AB = os.path.join(img_fold_AB, name_AB)
                if not no_multiprocessing:
                    pool.apply_async(image_write, args=(path_A, path_B, path_AB))
                else:
                    image_write(path_A, path_B, path_AB)

    if not no_multiprocessing:
        pool.close()
        pool.join()

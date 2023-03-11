import os
import shutil
import multiprocessing
from functools import partial
import numpy as np
from PIL import Image
import imageio


Image.MAX_IMAGE_PIXELS = 933120000

def copy_images_to_single_directory(all_dir, resized_images_dir):

    resized_tape_type_dirs = [f"{resized_images_dir}{os.sep}HQHT", 
                                f"{resized_images_dir}{os.sep}HQC",
                                f"{resized_images_dir}{os.sep}HQHTS", 
                                f"{resized_images_dir}{os.sep}HQSC", 
                                f"{resized_images_dir}{os.sep}LQHT", 
                                f"{resized_images_dir}{os.sep}LQHTS", 
                                f"{resized_images_dir}{os.sep}LQSC", 
                                f"{resized_images_dir}{os.sep}MQHT", 
                                f"{resized_images_dir}{os.sep}MQSC"]


    if os.path.exists(all_dir):
        shutil.rmtree(all_dir)
    os.makedirs(all_dir)

    for resized_tape_type_dir in resized_tape_type_dirs:
        for file in os.listdir(resized_tape_type_dir):
            shutil.copy2(f"{resized_tape_type_dir}{os.sep}{file}", all_dir )




if __name__ == '__main__':
    parent_dir = f"{os.sep}users{os.sep}lllang{os.sep}SCRATCH{os.sep}forensics"
    # parent_dir = f"Z:{os.sep}Research{os.sep}forensic"
    dataset_dir = f"{parent_dir}{os.sep}datasets{os.sep}"

    # Directory to copy all images to 
    all_dir=f"{dataset_dir}{os.sep}original_images"

    # Originals image directories to copy from
    resized_images_dir = f"{dataset_dir}{os.sep}resized_images"
    
    copy_images_to_single_directory(all_dir=all_dir,resized_images_dir=resized_images_dir)
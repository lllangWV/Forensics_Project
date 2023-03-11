import os
import shutil
import multiprocessing
from functools import partial
import numpy as np
from PIL import Image
import imageio



def copy_from_shared_directory(copy_dir, original_scan_dir):
    

    if os.path.exists(copy_dir):
        shutil.rmtree(copy_dir)
    os.makedirs(copy_dir)


    dirs_to_copy = [f"{original_scan_dir}{os.sep}High Quality Hand Torn{os.sep}Set 1", 
                            f"{original_scan_dir}{os.sep}High Quality Hand Torn{os.sep}Set C", 
                            f"{original_scan_dir}{os.sep}High Quality Hand Torn Stretched", 
                            f"{original_scan_dir}{os.sep}High Quality Scissor Cut",
                            f"{original_scan_dir}{os.sep}Low Quality Hand Torn", 
                            f"{original_scan_dir}{os.sep}Low Quality Hand Torn Stretched", 
                            f"{original_scan_dir}{os.sep}Low Quality Scissor Cut",
                            f"{original_scan_dir}{os.sep}Medium Quality Hand Torn", 
                            f"{original_scan_dir}{os.sep}Medium Quality Scissor Cut",
                            f"{original_scan_dir}{os.sep}Corrected Scans", 
                            f"{original_scan_dir}{os.sep}Master Inventory"]

    dirs_to_copy_to = [f"{copy_dir}{os.sep}High Quality Hand Torn", 
                            f"{copy_dir}{os.sep}High Quality Cut", 
                            f"{copy_dir}{os.sep}High Quality Hand Torn Stretched", 
                            f"{copy_dir}{os.sep}High Quality Scissor Cut",
                            f"{copy_dir}{os.sep}Low Quality Hand Torn", 
                            f"{copy_dir}{os.sep}Low Quality Hand Torn Stretched", 
                            f"{copy_dir}{os.sep}Low Quality Scissor Cut",
                            f"{copy_dir}{os.sep}Medium Quality Hand Torn", 
                            f"{copy_dir}{os.sep}Medium Quality Scissor Cut",
                            f"{copy_dir}{os.sep}Corrected Scans", 
                            f"{copy_dir}{os.sep}Master Inventory"]


    for dir_to_copy, dir_to_copy_to in zip(dirs_to_copy,dirs_to_copy_to):
        shutil.copytree(dir_to_copy, dir_to_copy_to, ignore =shutil.ignore_patterns("Old{slash}Replaced Scans") )


if __name__ == '__main__':

    scratch_dir = f"{os.sep}users{os.sep}lllang{os.sep}SCRATCH"
    parent_dir = f"{scratch_dir}{os.sep}forensics"
    dataset_dir = f"{parent_dir}{os.sep}datasets"

    # Directory to copy to 
    copy_dir= f"{dataset_dir}{os.sep}copied_images"
    
    # Directory of original scans
    original_scan_dir = f"{scratch_dir}{os.sep}shared_forensics{os.sep}Tape{os.sep}Automated_Algorithm_Photos{os.sep}Scans"
    copy_from_shared_directory(copy_dir=copy_dir, original_scan_dir=original_scan_dir)
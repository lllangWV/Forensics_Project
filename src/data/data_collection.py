import os
import shutil
import multiprocessing
from functools import partial
import numpy as np
from PIL import Image
import imageio


def copy_mp(directory_pair):

    dir_to_copy, dir_to_copy_to =  directory_pair[0],directory_pair[1]
    print(f'Copying {dir_to_copy} to {dir_to_copy_to}')
    shutil.copytree(dir_to_copy, dir_to_copy_to, ignore =shutil.ignore_patterns("Old{slash}Replaced Scans") )
    return None

def collect_data(copy_dir, original_scan_dir,ncores=6):
    
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

    if ncores==1:
        print("___________________")
        print("Processing Serially")
        print("___________________")
        
        for dir_to_copy, dir_to_copy_to in zip(dirs_to_copy,dirs_to_copy_to):
            print(f'Copying {dir_to_copy} to {dir_to_copy_to}')
            shutil.copytree(dir_to_copy, dir_to_copy_to, ignore =shutil.ignore_patterns("Old{slash}Replaced Scans") )
    else:
        print("___________________")
        print("Processing in Parallel")
        print("___________________")
        with multiprocessing.Pool(ncores) as pool:
            pool.map(copy_mp, list(zip(dirs_to_copy,dirs_to_copy_to)))

        




if __name__ == '__main__':

    scratch_dir = f"{os.sep}users{os.sep}lllang{os.sep}SCRATCH"
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    dataset_dir = f"{project_dir}{os.sep}datasets"

    # Directory to copy to 
    copy_dir= f"{dataset_dir}{os.sep}raw{os.sep}shared"
    if os.path.exists(copy_dir):
        shutil.rmtree(copy_dir)
    os.makedirs(copy_dir)
    # Directory of original scans
    original_scan_dir = f"{scratch_dir}{os.sep}shared_forensics{os.sep}Tape{os.sep}Automated_Algorithm_Photos{os.sep}Scans"

    collect_data(copy_dir=copy_dir, original_scan_dir=original_scan_dir, ncores = 11)
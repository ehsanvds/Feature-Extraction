"""
Image Preprocessing
"""
# importing modules
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

#%% functions
def filelist(path, ext):
    """creating a list of specific files in a folder"""
    # path is the folder path and ext is the file extension.
    from os import listdir
    file = [f for f in listdir(path) if f.endswith(ext)]
    file.sort()
    return file

def folderlist(path):
    """creating a list of folders in a directory"""
    # path is the path of a directory.
    from os import walk
    for (_,folder,_) in walk(path):
        break
    folder.sort()
    return folder

def correction(array_t, array_r, averaging=True, scale_max=255):
    """Image correction based on a reference image"""
    if averaging:
        # the average of all pixels
        ref = np.mean(array_r)
        array_t = array_t / ref * scale_max
    else:
        # element wise correction assuming the dimensions of the images are the same
        array_t = array_t / array_r * scale_max
    return array_t



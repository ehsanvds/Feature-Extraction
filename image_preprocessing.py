"""
Image Analysis

@author: Ehsan Vaziri
October 21, 2019
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

def normalize(array_t, array_w, array_b, averaging=True, scale_max=255):
    """Normalizing an image based on white and black images"""
    if averaging:
        # the average of all pixels
        w = np.mean(array_w)
        b = np.mean(array_b)
        array_t = (array_t - b) / (w - b) * scale_max
    else:
        # element wise normalization assuming the dimensions of the images are the same
        array_t = (array_t - array_b) / (array_w - array_b) * scale_max
    return array_t



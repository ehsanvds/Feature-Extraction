"""
Image Preprocessing
"""
# importing modules
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

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

def correction_all(input_path, output_path, ref_img, ext):
    """correcting and saving all images"""
    # reading the files
    array_r = np.array(Image.open(ref_img))
    # list of folders
    folders = folderlist(input_path)
    for i in tqdm(folders):
        # output directory
        outdir = os.path.join(output_path,i)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        # input directory
        indir = os.path.join(input_path,i)
        files = filelist(indir, ext)
        for k in files:
            array_t = np.mean(np.array(Image.open(os.path.join(indir,k))), axis=2)
            array_t = correction(array_t, array_r, averaging=False)
            img_out = Image.fromarray(array_t.astype('uint8'))
            img_out = img_out.convert('RGB')
            img_out.save(os.path.join(outdir,k))

 def combine_files(input_path, ext):
    """Combining mulitple files and setting the header as file names"""
    # the first two columns in each file are used for matching rows
    # list of folders
    folders = folderlist(input_path)
    for i in tqdm(folders):
        # input directory
        indir = os.path.join(input_path,i)
        files = filelist(indir, ext)
        df_all = pd.DataFrame()
        for j in files:
            # reading the file. Check for the settings.
            df = pd.read_csv(os.path.join(indir,j), comment=';', header=None)
            # setting fist two columns as indices (x and y)
            df.set_index([df.columns[0], df.columns[1]], inplace=True)
            # averaging columns as in RGB file
            df = df.mean(axis=1)
            # concatenating 
            df_all = pd.concat([df_all,df], ignore_index=False, axis=1, join='outer')
        # changing indices to columns
        df_all = df_all.reindex(pd.MultiIndex.from_tuples(df_all.index))
        df_all.reset_index(inplace=True)
        # assigning the header. Check for the first two columns.
        df_all.columns = ['x_pixel','y_pixel']+[i.replace(ext,'') for i in files]
        # saving
        df_all.to_csv(os.path.join(input_path,i+'_all.csv'), index=False)

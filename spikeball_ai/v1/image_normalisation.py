

# - just match mean and std
# - histogram matching
# - comprehensive colour normalization


#%% inputs

import numpy as np

import colour
from skimage.exposure import match_histograms

import os
from pathlib import Path
import general_utils as utils


#%%  


# from img's mean and std to a reference one (per channel)
def standardise(img, new_mean, new_std, rgb = True):
    #TODO check if all shapes correct
    if rgb:
        new_img = np.copy(img)
    else:
        new_img = utils.convert_RGB_to_Lab(img)
    # 
    channels = []
    for i_ch in range(new_img.shape[2]):
        temp_slice = new_img[:, :, i_ch] - new_img[:, :, i_ch].mean(axis = (0, 1))
        temp_slice = temp_slice/new_img[:, :, i_ch].std(axis = (0, 1))
        # 
        temp_slice = temp_slice * new_std[i_ch]
        temp_slice = temp_slice + new_mean[i_ch]
        # 
        channels.append(temp_slice)
    # 
    temp_img = np.dstack(channels)
    if not rgb:
        temp_img = utils.convert_Lab_to_RGB(temp_img)
    temp_img = np.clip(temp_img, 0, 1)
    return temp_img


# scikit-image's Histogram Matching function
def match_histo(img, reference):
    
    temp_img = match_histograms(img, reference, channel_axis = -1)
    temp_img = np.clip(temp_img, 0, 1)
    
    return temp_img


#%%

# img_path = 'E:\Python_Data\spikeball\spikeball_v1\img_data\internet_grass/'
# images = os.listdir(Path(img_path))


# #%%

# ref_img = colour.read_image(img_path + images[3])[:, :, :3]

# ref_lab = utils.convert_RGB_to_Lab(ref_img)

# ref_mean, ref_std = ref_img.mean(axis = (0, 1)), ref_img.std(axis = (0, 1))
# lab_mean, lab_std = ref_lab.mean(axis = (0, 1)), ref_lab.std(axis = (0, 1))

# for img_name in images:
#     img = colour.read_image(img_path + img_name)[:, :, :3]
#     utils.show_img(img, size_scaler = 0.5, title = 'img')
    
#     temp_img = standardise(img, ref_mean, ref_std)
#     utils.show_img(temp_img, size_scaler = 0.5, title = 'standardised on RGB')
    
#     temp_img = standardise(img, lab_mean, lab_std, rgb = False)
#     utils.show_img(temp_img, size_scaler = 0.5, title = 'standardised on Lab')
    
#     # temp_img = match_histo(img, ref_img)
#     # utils.show_img(temp_img, size_scaler = 0.5, title = 'match histograms RGB')
    
#     # temp_img = match_histo(utils.convert_RGB_to_Lab(img), ref_lab)
#     # temp_img = utils.convert_Lab_to_RGB(temp_img)
#     # utils.show_img(temp_img, size_scaler = 0.5, title = 'match histograms Lab')
# # 































































































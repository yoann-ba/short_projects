



#%% imports 

import numpy as np
import matplotlib.pyplot as plt

import colour

import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

# -- own scripts
import general_utils as utils
import image_normalisation as norm
import build_run_models

#%% funcs

# 
def build_5d_img(img):
    
    [rows, cols] = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rows = rows/img.shape[0]
    cols = cols/img.shape[1]
    temp = np.dstack((rows, cols))
    temp = np.dstack((img, temp))
    
    return temp.astype(np.float16)



#%% sandbox

img_path = r"E:\Python_Data\spikeball\spikeball_v1\img_data\alex_raphi_grass/"
shaped_datasets = "E:\Python_Data\spikeball\spikeball_v1\shaped_datasets/3_label_unique_4reduc/"

ref_path = r"E:\Python_Data\spikeball\spikeball_v1\img_data\internet_grass/"
ref_name = "chrome_fsGkZjAQZT.jpg"

img_names = os.listdir(Path(img_path))

#%%

model = build_run_models.run_all_models(shaped_datasets, 
                                        ["RFC"], [RandomForestClassifier(class_weight="balanced", n_jobs = -1)])


#%%

for img_name in img_names:
    print(f' -- {img_name}')
    img = colour.read_image(Path(img_path + img_name))
    utils.show_img(img, title = f"original {img_name}", size_scaler = 1)
    
    ref_img = colour.read_image(Path(ref_path+ref_name))
    ref_lab = utils.convert_RGB_to_Lab(ref_img)
    lab_mean, lab_std = ref_lab.mean(axis = (0, 1)), ref_lab.std(axis = (0, 1))
    
    img = colour.read_image(Path(img_path+img_name))[:, :, :3]
    
    # Standardise (in Lab)
    img = norm.standardise(img, lab_mean, lab_std, rgb = False)
    utils.show_img(img, title = 'standard Lab', size_scaler = 1)
    
    # # rgb
    # for var in utils.colors_to_labels:
    #     prediction = img.reshape((img.shape[0]*img.shape[1], 3)).astype(np.float16)
    #     prediction = model[f"{var}_rgb"].predict_proba(prediction)[:, 1]
    #     prediction = prediction.reshape((img.shape[0], img.shape[1]))
    #     utils.show_img(prediction, title = f"RFC {var} rgb", size_scaler = 1, cmap = 'gray')
    # 5d
    for var in utils.colors_to_labels:
        temp_5d = build_5d_img(img)
        prediction = temp_5d.reshape((img.shape[0]*img.shape[1], 5)).astype(np.float16)
        prediction = model[f"{var}_5d"].predict_proba(prediction)[:, 1]
        prediction = prediction.reshape((img.shape[0], img.shape[1]))
        utils.show_img(prediction, title = f"RFC {var} 5d", cmap = 'gray')
# 












































































































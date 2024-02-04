

#%% imports

import numpy as np
import pandas as pd

from pathlib import Path

import colour


#%% util funcs

# 
def convert_RGB_to_Lab(rgb_img):
    
    # alternatively colour.XYZ_to_Lab for CIELab
    return colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(rgb_img))

# 
def convert_Lab_to_RGB(lab_img):
    
    # alternatively
    return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))



#%% sandbox

img_path = 'E:\Python_Data\spikeball\spikeball_quick/'
img_name = "frame_"+str(0)+".png"

img_path = Path(img_path + img_name)

rgb_img = colour.read_image(img_path)[:, :, :3]

# lab_img = convert_RGB_to_Lab(rgb_img)


img_path = 'E:\Python_Data\spikeball\spikeball_quick/'
img_name = "label_frame_"+str(0)+".png"

img_path = Path(img_path + img_name)

label_rgb = colour.read_image(img_path)[:, :, :3]

# label_lab = convert_RGB_to_Lab(label_rgb)


#%%

colors_to_labels = {
    'isBall': [1, 1, 1], 
    'isGrass': [0, 0, 0], 
    'isGoalPost': [1, 0, 0], 
    'isSky': [0, 0, 1], 
    'isSkin': [0, 1, 0], 
    }

dataset = {}

unique = True

# one dataset per label, with boolean 1/0
for label in colors_to_labels:
    label_color = colors_to_labels[label]
    
    indices = np.where(np.all(label_rgb == label_color, axis=-1))
    coords = zip(indices[0], indices[1])
    coords = np.array(list(coords))
    
    rgb_colors = rgb_img[tuple(coords.T)]
    # lab_colors = lab_img[tuple(coords.T)]
    
    if unique:
        rgb_colors = (255*rgb_colors).astype(np.uint8)
        rgb_colors = np.unique(rgb_colors, axis = 0)/255
        rgb_colors = rgb_colors.astype(np.float16)
        
        # lab_colors = (255*lab_colors).astype(np.uint8)
        # lab_colors = np.unique(lab_colors, axis = 0)/255
        # lab_colors = lab_colors.astype(np.float16)
    
    # y is fine even though the unique shuffles, since it's always a full block of 1s or 0s
    y = np.full(rgb_colors.shape[0], 1)
    
    # for all other labels, add their data with a 0
    for other_label in colors_to_labels:
        if label == other_label:
            continue
        label_color = colors_to_labels[other_label]
        
        indices = np.where(np.all(label_rgb == label_color, axis=-1))
        coords = zip(indices[0], indices[1])
        coords = np.array(list(coords))
        
        temp_rgb_colors = rgb_img[tuple(coords.T)]
        #temp_lab_colors = lab_img[tuple(coords.T)]
        
        if unique:
            temp_rgb_colors = (255*temp_rgb_colors).astype(np.uint8)
            temp_rgb_colors = np.unique(temp_rgb_colors, axis = 0)/255
            temp_rgb_colors = temp_rgb_colors.astype(np.float16)
            
            # temp_lab_colors = (255*temp_lab_colors).astype(np.uint8)
            # temp_lab_colors = np.unique(temp_lab_colors, axis = 0)/255
            # temp_lab_colors = temp_lab_colors.astype(np.float16)
        
        temp_y = np.full(temp_rgb_colors.shape[0], 0)
        
        rgb_colors = np.vstack((rgb_colors, temp_rgb_colors))
        # lab_colors = np.vstack((lab_colors, temp_lab_colors))
        y = np.concatenate((y, temp_y))
    # 
    
    np.save(f"{label}_rgb", np.array(rgb_colors))
    np.save(f"{label}_y", np.array(y))
# 

































































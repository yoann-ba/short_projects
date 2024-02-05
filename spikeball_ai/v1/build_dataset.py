

#%% imports

import numpy as np

import os
from pathlib import Path

import colour

# -- own scripts
import general_utils as utils
import image_normalisation as norm


#%% labels

# in floats, *255 for uint8
colors_to_labels = {
    'isBall': [1, 1, 1], 
    'isGrass': [0, 0, 0], 
    'isGoalPost': [1, 0, 0], 
    'isSky': [0, 0, 1], 
    'isSkin': [0, 1, 0], 
    }


#%% funcs

# 
def get_indices_of_color(img, color_triple):
        
    indices = np.where(np.all(img == color_triple, axis=-1))
    coords = zip(indices[0], indices[1])
    coords = np.array(list(coords))
    
    return coords

# take a list of coords and get a list of pixels i nan image at those coords
# unique makes the list unique
# sample_reduction makes the 'unique' erasure width harsher
# also returns the new coords if unique was applied, to use for 5D input
def get_colors_of_coords(img, coords, unique = True, sample_reduction = 1):
    
    colors = img[tuple(coords.T)]
    
    if unique:
        colors = (int(255/sample_reduction)*colors).astype(np.uint8)
        colors, idx = np.unique(colors, axis = 0, return_index = True)
        colors = colors/int(255/sample_reduction)
        colors = colors.astype(np.float16)
        
        new_coords = coords[idx]
    # 
    
    return colors, new_coords

# 
def build_5d(colors, pixel_coords, img_shape):
    
    temp1 = pixel_coords[:, 0]/img_shape[0]
    temp2 = pixel_coords[:, 1]/img_shape[1]
    temp1 = np.expand_dims(temp1, axis = -1)
    temp2 = np.expand_dims(temp2, axis = -1)
    
    return np.hstack((colors, temp1, temp2)).astype(np.float16)


#%% paths

img_path = "E:\Python_Data\spikeball\spikeball_v1/img_data/internet_grass/"
label_path = "E:\Python_Data\spikeball\spikeball_v1/img_label/internet_grass/"

image_names = os.listdir(Path(img_path))

#%% ref pre process

ref_name = "chrome_fsGkZjAQZT.jpg"
ref_img = colour.read_image(Path(img_path+ref_name))

ref_lab = utils.convert_RGB_to_Lab(ref_img)
lab_mean, lab_std = ref_lab.mean(axis = (0, 1)), ref_lab.std(axis = (0, 1))


#%% loop all label images

dataset = {}
for img_name in image_names:
    print(f' -- {img_name}')
    img = colour.read_image(Path(img_path+img_name))[:, :, :3]
    label_img = colour.read_image(Path(label_path+img_name.split('.')[0]+'.png'))[:, :, :3]
    
    # Standardise (in Lab)
    img = norm.standardise(img, lab_mean, lab_std, rgb = False)
    
    for label in colors_to_labels:
        print(label)
        label_color = colors_to_labels[label]
        
        coords = get_indices_of_color(label_img, label_color)
        if len(coords) == 0:
            print('no pixel found, skip')
            continue
        rgb_colors, coords = get_colors_of_coords(img, coords)
        # 
        if f"{label}_rgb" not in dataset:
            dataset[f"{label}_rgb"] = rgb_colors
            dataset[f"{label}_5d"] = build_5d(rgb_colors, coords, img.shape)
        else: # so we can np unique every time instead of doing a massive one at the end
            dataset[f"{label}_rgb"] = np.vstack((dataset[f"{label}_rgb"], rgb_colors))
            dataset[f"{label}_rgb"] = np.unique(dataset[f"{label}_rgb"], axis = 0)
            temp = build_5d(rgb_colors, coords, img.shape)
            dataset[f"{label}_5d"] = np.vstack((dataset[f"{label}_5d"], temp))
            dataset[f"{label}_5d"] = np.unique(dataset[f"{label}_5d"], axis = 0)
            print(dataset[f"{label}_rgb"].shape)
            print(dataset[f"{label}_5d"].shape)
    # 
# 


#%% reshape dataset into binary

shaped_dataset = {}
for label in colors_to_labels:
    rgb_colors = dataset[f"{label}_rgb"]
    data_5d = dataset[f"{label}_5d"]
    y_rgb = np.full(rgb_colors.shape[0], 1)
    y_5d = np.full(data_5d.shape[0], 0)
    
    for other_label in colors_to_labels:
        if label == other_label:
            continue
        temp_rgb = dataset[f"{other_label}_rgb"]
        temp_5d = dataset[f"{other_label}_5d"]
        
        temp_y_rgb = np.full(temp_rgb.shape[0], 0)
        temp_y_5d = np.full(temp_5d.shape[0], 0)
        
        rgb_colors = np.vstack((rgb_colors, temp_rgb))
        data_5d = np.vstack((data_5d, temp_5d))
        y_rgb = np.concatenate((y_rgb, temp_y_rgb))
        y_5d = np.concatenate((y_5d, temp_y_5d))
    # 
    shaped_dataset[f"{label}_rgb_full"] = rgb_colors
    shaped_dataset[f"{label}_5d_full"] = data_5d
    shaped_dataset[f"{label}_y_rgb_full"] = y_rgb
    shaped_dataset[f"{label}_y_5d_full"] = y_5d
# 


#%% save

for file_name in shaped_dataset:
    np.save(file_name, shaped_dataset[file_name])
# 

#%%





























































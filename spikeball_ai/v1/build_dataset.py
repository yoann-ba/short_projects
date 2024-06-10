

#%% imports

import numpy as np

import os
from pathlib import Path

import colour

# -- own scripts
import general_utils as utils
import image_normalisation as norm


#%% labels




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
        return_coords = new_coords
    else:
        return_coords = coords
    
    return colors, return_coords

# 
# RGB, rows%, cols%
def build_5d(colors, pixel_coords, img_shape):
    
    temp1 = pixel_coords[:, 0]/img_shape[0]
    temp2 = pixel_coords[:, 1]/img_shape[1]
    temp1 = np.expand_dims(temp1, axis = -1)
    temp2 = np.expand_dims(temp2, axis = -1)
    
    return np.hstack((colors, temp1, temp2)).astype(np.float16)


#%% main 

def build_positive_datasets(img_path, label_path, ref_name, 
                            unique = True, sample_reduction = 1):
    
    image_names = os.listdir(Path(img_path))
    
    ref_img = colour.read_image(Path(img_path+ref_name))
    ref_lab = utils.convert_RGB_to_Lab(ref_img)
    lab_mean, lab_std = ref_lab.mean(axis = (0, 1)), ref_lab.std(axis = (0, 1))
    
    dataset = {}
    for img_name in image_names:
        print(f' -- {img_name}')
        img = colour.read_image(Path(img_path+img_name))[:, :, :3]
        label_img = colour.read_image(Path(label_path+img_name.split('.')[0]+'.png'))[:, :, :3]
        
        # Standardise (in Lab)
        img = norm.standardise(img, lab_mean, lab_std, rgb = False)
        
        for label in utils.colors_to_labels:
            print(label)
            label_color = utils.colors_to_labels[label]
            
            coords = get_indices_of_color(label_img, label_color)
            if len(coords) == 0:
                print('no pixel found, skip')
                continue
            rgb_colors, coords = get_colors_of_coords(img, coords, unique=unique, 
                                                      sample_reduction=sample_reduction)
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
    
    return dataset


# takes the positive dataset (x_true, y=1) for each variable
# and shuffle them into n (x_true,y=1),(x_false, y=0) datasets
def build_shaped_binary_datasets(dataset):
    
    shaped_dataset = {}
    for label in utils.colors_to_labels:
        rgb_colors = dataset[f"{label}_rgb"]
        data_5d = dataset[f"{label}_5d"]
        y_rgb = np.full(rgb_colors.shape[0], 1)
        y_5d = np.full(data_5d.shape[0], 1)
        
        for other_label in utils.colors_to_labels:
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
    
    return shaped_dataset

# 
def save_shaped(shaped_dataset):
    
    for file_name in shaped_dataset:
        np.save(file_name, shaped_dataset[file_name])
    
    return


#%% paths

img_path = "E:\Python_Data\spikeball\spikeball_v1/img_data/internet_grass/"
label_path = "E:\Python_Data\spikeball\spikeball_v1/img_label/internet_grass/ball_spike_other/"

ref_name = "chrome_fsGkZjAQZT.jpg"

#%%

# dataset = build_positive_datasets(img_path, label_path, ref_name)


#%%

# shaped_datasets = build_shaped_binary_datasets(dataset)


#%%

# save_shaped(shaped_datasets)



























































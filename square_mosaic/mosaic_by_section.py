


#%% imports


import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

import cv2



#%% util funcs


# 
def show_img(img, size_scaler = 1, title = '', cmap = ''):
    
    screen_dpi = 96
    temp_size = (int(img.shape[0]*size_scaler/plt.rcParams['figure.dpi']), 
                 int(img.shape[1]*size_scaler/plt.rcParams['figure.dpi']))
    # for some reason the dpi is neither the screen dpi or plt params dpi?
    # 55 isnt either but its a close approximation
    temp_size = (int(img.shape[0]*size_scaler/55), 
                 int(img.shape[1]*size_scaler/55))
    plt.figure(figsize = temp_size, dpi = plt.rcParams['figure.dpi'])
    
    if cmap:
        plt.imshow(img, cmap = cmap)
    else:
        plt.imshow(img)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
    return

# 
def convert_RGB_to_Lab(rgb_img):
    
    # alternatively colour.XYZ_to_Lab for CIELab
    # return colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(rgb_img))
    return cv2.cvtColor(rgb_img.astype(np.float32), cv2.COLOR_RGB2LAB)

# 
def convert_Lab_to_RGB(lab_img):
    
    # alternatively
    # return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))
    return cv2.cvtColor(lab_img.astype(np.float32), cv2.COLOR_LAB2RGB)


#%% funcs


def cv2_open_img(path, resize = None):
    
    temp_img = cv2.imread(path)
    if resize != None:
        temp_shape = (temp_img.shape[1]//resize, temp_img.shape[0]//resize)
        temp_img = cv2.resize(temp_img, temp_shape)
        # print(temp_img.shape)
    temp_img = temp_img.astype(np.float32)/255
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    
    return temp_img


def get_mean_std(lab):
    
    sub_mean = np.mean(lab, axis = (0, 1))
    sub_std = np.std(lab, axis = (0, 1))
    
    return sub_mean, sub_std


def main_splits(img, lab, n_splits = 5):
    
    row_grid = img.shape[0]//n_splits
    col_grid = img.shape[1]//n_splits
    # print(f"squares of shape {row_grid}*{col_grid}")
    
    all_mu = []
    all_sig = []
    for i in range(n_splits):
        for j in range(n_splits):
            #sub_img = img[row_grid*i:row_grid*(i+1), col_grid*j:col_grid*(j+1), :]
            sub_lab = lab[row_grid*i:row_grid*(i+1), col_grid*j:col_grid*(j+1), :]
            mu, sig = get_mean_std(sub_lab)
            all_mu.append(mu)
            all_sig.append(sig)
            # show_img(sub_img)
        # 
    # 
    
    return all_mu, all_sig


def build_img_dict(ref_path, n_splits = 5):
    
    list_img = os.listdir(ref_path)
    img_dict = {}
    for img_name in list_img:
        print(img_name)
        temp_img = cv2_open_img(ref_path + img_name, resize = 4)
        # 
        # show_img(temp_img)
        temp_lab = convert_RGB_to_Lab(temp_img)
        all_mu, all_sig = main_splits(temp_img, temp_lab, n_splits=n_splits)
        # 
        img_dict[img_name] = all_mu
    # 
    
    return img_dict


#%% sandbox


base_path = "E:/Python_Data/general_img_db/"
crop_folder_path = "E:/Python_Data/general_img_db/anime_comics_fantasy_game/lol_skins/"

img = Path(base_path + "alex_dog.jpg")

img = cv2.imread(img)

img = img.astype(np.float32)/255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lab = convert_RGB_to_Lab(img)


#%% 

n_splits_refs = 3
img_dict = build_img_dict(crop_folder_path, n_splits = n_splits_refs)


#%%

n_splits = 50
row_grid = img.shape[0]//n_splits
col_grid = img.shape[1]//n_splits
print(f"squares of shape {row_grid}*{col_grid}")

show_img(img)
canvas = np.zeros_like(img)
for i in range(n_splits):
    for j in range(n_splits):
        # sub_img = img[row_grid*i:row_grid*(i+1), col_grid*j:col_grid*(j+1), :]
        # show_img(sub_img)
        sub_lab = lab[row_grid*i:row_grid*(i+1), col_grid*j:col_grid*(j+1), :]
        mu, sig = get_mean_std(sub_lab)
        closest = 0
        closest_name = ''
        closest_distance = 100000000000
        for key in img_dict:
            all_mu = img_dict[key]
            for i_mu in range(len(all_mu)):
                temp_dist = np.linalg.norm(all_mu[i_mu] - mu)
                if temp_dist < closest_distance:
                    closest_distance = temp_dist
                    closest = i_mu
                    closest_name = key
        # 
        temp_img = cv2_open_img(crop_folder_path + closest_name)
        t_row_grid = temp_img.shape[0]//n_splits_refs
        t_col_grid = temp_img.shape[1]//n_splits_refs
        t_i = closest//n_splits_refs
        t_j = closest%n_splits_refs
        temp_img = temp_img[t_row_grid*t_i:t_row_grid*(t_i+1), t_col_grid*t_j:t_col_grid*(t_j+1), :]
        temp_img = cv2.resize(temp_img, (sub_lab.shape[1], sub_lab.shape[0]))
        canvas[row_grid*i:row_grid*(i+1), col_grid*j:col_grid*(j+1), :] = temp_img
        # show_img(canvas)
        # 
    # 
# 
show_img(canvas)



#%% 

# list_img = os.listdir(crop_folder_path)
# img_dict = {}
# for img_name in list_img:
#     print(img_name)
#     temp_img = cv2.imread(crop_folder_path + img_name)
#     temp_shape = (temp_img.shape[1]//4, temp_img.shape[0]//4)
#     temp_img = cv2.resize(temp_img, temp_shape)
#     print(temp_img.shape)
#     temp_img = temp_img.astype(np.float32)/255
#     temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
#     # 
#     show_img(temp_img)
#     temp_lab = convert_RGB_to_Lab(temp_img)
#     all_mu, all_sig = main_splits(temp_img, temp_lab)
#     # 
#     img_dict[img_name] = all_mu
# # 


#%% 

# n_splits = 5
# row_grid = img.shape[0]//n_splits
# col_grid = img.shape[1]//n_splits
# print(f"squares of shape {row_grid}*{col_grid}")

# for i in range(n_splits):
#     for j in range(n_splits):
#         sub_img = img[row_grid*i:row_grid*(i+1), col_grid*j:col_grid*(j+1), :]
#         show_img(sub_img)
#         sub_lab = lab[row_grid*i:row_grid*(i+1), col_grid*j:col_grid*(j+1), :]
        
#     # 
# # 















































































































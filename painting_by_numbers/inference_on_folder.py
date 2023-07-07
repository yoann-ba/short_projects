# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:51:54 2023

@author: Yoann
"""


#%% imports


import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

import cv2

from sklearn.cluster import KMeans


#%% paths and globals


screen_dpi = 96


#%% util functions

# 
# no frame, no white border
# should be exact size as input, with one pixel per pixel preserved
# > at least without titles, w/ title might be different shape
def show_img(img, title = ''):
    
    size = (img.shape[0]/screen_dpi, img.shape[1]/screen_dpi)
    fig = plt.figure(figsize = size, dpi = screen_dpi, frameon = False)
    
    plt.imshow(img)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
    return


# 
# see https://stackoverflow.com/a/56900830 for colorbar size
def show_img_gray(img, title = '', cmap = 'viridis'):
    
    size = (img.shape[0]/screen_dpi, img.shape[1]/screen_dpi)
    fig = plt.figure(figsize = size, dpi = screen_dpi)
    ax = plt.axes()
    
    imshow_ref = ax.imshow(img, cmap = cmap)
    
    if title:
        plt.title(title)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(imshow_ref, cax = cax)
    
    plt.show()
    
    return


# 
def show_color_palette(color_list, title = ""):
    
    plt.figure(figsize = (8, 6))
    plt.imshow(np.array([color_list]))
    plt.axis('off')
    plt.title("chosen colors\n"+title)
    plt.show()
    
    return


# 
def convert_to_Lab(rgb_img):
    
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    
    return lab_img
# 
def convert_to_RGB(lab_img):
    
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    
    return rgb_img


# 
# https://stackoverflow.com/a/55590133
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


#%% methods


# 
# naive color counting
def count_colors(lab_img, N_trim = 500, plot_title = '', plot_frequency = False):
    
    # (N pixels, pixel color value)
    color_list = lab_img.reshape((lab_img.shape[0]*lab_img.shape[1], lab_img.shape[2]))
    
    # get the unique pixels, their indices, and their frequencies
    unique, inverse, counts = np.unique(color_list, axis = 0, 
                                        return_counts = True, return_inverse = True)
    inv_counts = -counts
    # debug
    if plot_frequency:
        show_img_gray(counts[inverse].reshape((lab_img.shape[0], lab_img.shape[1])), 
                      title = "pixel value frequency\n"+plot_title)
    
    # get the N_trim top pixel frequencies, save their spot in the freq list
    counts_argsort = inv_counts.argsort()
    trimmed_list = counts_argsort[:N_trim]
    
    new_img = np.zeros(color_list.shape, dtype = color_list.dtype)
    for arg_index in trimmed_list:
        temp_color = unique[arg_index] # color value
        temp_indices = np.argwhere(inverse == arg_index) # where in color_lit was it
        for ind in temp_indices:
            new_img[ind] = temp_color
    # 
    new_img = new_img.reshape(lab_img.shape)
    temp = cv2.cvtColor(new_img, cv2.COLOR_LAB2RGB)
    show_img(temp, title = f"only the top {N_trim} colors\nblue is empty\n"+plot_title)
    
    return new_img


# 
def re_round(lab_0_255_img, discrete_range = 100):
    
    temp_64 = lab_0_255_img.astype(np.float64) # get rid of uint8
    
    # -> [0.0 - 1.0] -> [0.0 - value] -> [0 - val]
    original_min, original_max = temp_64.min(), temp_64.max()
    temp_01_64 = (temp_64 - original_min)/(original_max - original_min)
    
    temp_newrange_64 = temp_01_64 * discrete_range
    temp_newrange = temp_newrange_64.astype(int)
    
    # -> [0.0 - value] -> [0.0 - 1.0] -> [0.0 - 255.0] -> [0 - 255]
    new_img = temp_newrange.astype(np.float64)
    new_img = (new_img - new_img.min())/(new_img.max() - new_img.min())
    new_img = new_img * (original_max - original_min) + original_min
    
    new_img = new_img.astype(np.uint8) # back to open cv format
    
    return new_img


# 
# K Means clustering in Lab space
# specify the nb of clusters
# returns img of same size with cluster Lab pixel values
def kmeans_clustering(lab_img_color, lab_img_predict = None, n_clusters = 20, 
                      plot_title = '', unique_colors = False):
    
    # (N pixels, pixel color value)
    temp_shape = (lab_img_color.shape[0]*lab_img_color.shape[1], lab_img_color.shape[2])
    full_color_list = lab_img_color.reshape(temp_shape)
    
    if unique_colors:
        # get the unique pixels, their indices, and their frequencies
        unique = np.unique(full_color_list, axis = 0)
        color_list = unique
    else:
        color_list = full_color_list
    color_list = color_list.astype(np.float64)
    
    kmeans = KMeans(n_clusters = n_clusters, n_init = 'auto')
    kmeans.fit(color_list)
    # for every cluster, Lab space coords
    cluster_centers = kmeans.cluster_centers_
    
    # ---- Predict
    
    if type(lab_img_predict) != type(None):
        temp_shape = (lab_img_predict.shape[0]*lab_img_predict.shape[1], lab_img_predict.shape[2])
        full_color_list = lab_img_predict.reshape(temp_shape)
    # for every pixel, cluster id
    cluster_indexes = kmeans.predict(full_color_list)
    # for every pixel, cluster Lab coords
    clustered_values = cluster_centers[cluster_indexes]
    
    if type(lab_img_predict) != type(None):
        new_img = clustered_values.reshape(lab_img_predict.shape)
    else:
        new_img = clustered_values.reshape(lab_img_color.shape)
    
    new_img = new_img.astype(np.uint8)
    temp = convert_to_RGB(new_img)
    colors = convert_to_RGB(np.array([cluster_centers], dtype = np.uint8))
    
    show_color_palette(colors[0], title = plot_title)
    #show_img(temp)
    
    return new_img, cluster_centers


#%% macro functions

# 
def run_inference(rgb_img, lab_img, blur_sigma = 150, n_clusters = 8, sharpen_amount = 1, 
                  plot_title = ""):
    
    rgb_blur = cv2.bilateralFilter(rgb_img, 9, blur_sigma, blur_sigma)
    #show_img(rgb_blur)
    lab_blur = convert_to_Lab(rgb_blur)
    
    result, colors = kmeans_clustering(lab_blur, lab_blur, n_clusters = n_clusters, unique_colors = True, 
                               plot_title = plot_title)
    
    sharpened_result = unsharp_mask(result, amount = sharpen_amount)
    sharp_rgb = cv2.cvtColor(sharpened_result, cv2.COLOR_LAB2RGB)
    
    show_img(sharp_rgb)
    
    return sharpened_result, colors


# 
def run_all_options(rgb_img, img_name):
    
    lab_img = convert_to_Lab(rgb_img)
    
    n_clusters = 5
    blur_sigma = 2 
    sharpen_amount = 0.1
    title = img_name+" | " + f"{n_clusters} colors, blur sigma {blur_sigma}, sharpen amount {sharpen_amount}"
    
    method_1, colors = run_inference(rgb_img, lab_img, blur_sigma, n_clusters, sharpen_amount, 
                                     plot_title = title)
    method_1_2 = run_inference(rgb_img, lab_img, blur_sigma, n_clusters, sharpen_amount, 
                                     plot_title = title)
    
    n_clusters = 8
    blur_sigma = 200
    sharpen_amount = 1
    title = img_name+" | " + f"{n_clusters} colors, blur sigma {blur_sigma}, sharpen amount {sharpen_amount}"
    
    method_2 = run_inference(rgb_img, lab_img, blur_sigma, n_clusters, sharpen_amount, 
                                     plot_title = title)
    method_2_2 = run_inference(rgb_img, lab_img, blur_sigma, n_clusters, sharpen_amount, 
                                     plot_title = title)
    
    n_clusters = 10
    blur_sigma = 300
    sharpen_amount = 2
    title = img_name+" | "+ f"{n_clusters} colors, blur sigma {blur_sigma}, sharpen amount {sharpen_amount}"
    
    method_3 = run_inference(rgb_img, lab_img, blur_sigma, n_clusters, sharpen_amount, 
                                     plot_title = title)
    method_3_2 = run_inference(rgb_img, lab_img, blur_sigma, n_clusters, sharpen_amount, 
                                     plot_title = title)
    
    return [method_1, method_2, method_3, method_1_2, method_2_2, method_3_2]


# 
def run_grid(rgb_img, img_name):
    
    lab_img = convert_to_Lab(rgb_img)
    
    for n_clusters in [2, 3, 4, 5, 8, 10, 15, 25]:
        for blur_sigma in [2, 100, 300, 700]:
            sharpen_amount = blur_sigma/150
            title = img_name+" | " + f"{n_clusters} colors, blur sigma {blur_sigma}, sharpen amount {sharpen_amount:0.2f}"
            method_1, colors = run_inference(rgb_img, lab_img, blur_sigma, n_clusters, sharpen_amount, 
                                             plot_title = title)
    
    return 


# 
def run_folder(folder_path):
    
    list_files = os.listdir(folder_path)
    print(list_files)
    
    for img_name in list_files:
        if '.' not in img_name:
            continue
        if img_name.split('.')[1] not in ['jpg', 'jpeg', 'png']:
            continue
        
        print(img_name)
        img_path = str(img_db_path) + "/" + img_name
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        
        #TODO reduce img shape if too big
        if img.shape[0] > 2000 or img.shape[1] > 2000:
            scale_percent = 50 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            img = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS4)
        # 
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = run_all_options(img, img_name)
    # 
    
    return


def run_all_one_img(img_path):
    
    img_name = img_path.split('.')[0].split('/')[-1]
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    
    # reduce img shape if too big
    if img.shape[0] > 2000 or img.shape[1] > 2000:
        scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        img = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS4)
    # 
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = run_grid(img, img_name)
    
    return


#%% main run


img_db_path = "E:\Python_Data\general_img_db/animal/"
img_db_path = "E:\Python_Data\general_img_db/anime_comics_fantasy_game/"
img_db_path = "E:\Python_Data\general_img_db/average_real_photo/"
img_db_path = "E:\Python_Data\general_img_db/classic_art/"
img_db_path = "E:\Python_Data\general_img_db/landscape_pro_photos/"
img_db_path = "E:\Python_Data\general_img_db/memes/"


img_db_path = "E:\Python_Data\general_img_db/anime_comics_fantasy_game/"

img_db_path = Path(img_db_path)

run_folder(img_db_path)


#%% 

solo_img_path = "E:\Python_Data\general_img_db/anime_comics_fantasy_game/" + "blood_moon_aatrox_2.jpg"

run_all_one_img(solo_img_path)




































































# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:45:29 2023

@author: Yoann
"""


#%% imports


import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import cv2

from sklearn.cluster import KMeans



#%% paths and globals

img_db_path = "E:\Python_Data\general_img_db/"

test_img = "test_bear.jpg"

test_path = img_db_path + test_img
test_path = Path(test_path)

img = cv2.imread(str(test_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    temp = cv2.cvtColor(new_img, cv2.COLOR_LAB2RGB)
    show_img(temp, title = f"KMeans clustering in Lab space with {n_clusters} clusters\n"+plot_title)
    
    return new_img


#%% main

lab_img = convert_to_Lab(img)


#%%

# just rounding Lab values to lower discrete ranges
for d_range in [255, 200, 150, 100, 50, 10, 5, 3, 2, 1]:
    reduced_lab = re_round(lab_img, d_range)
    reduced_rgb = convert_to_RGB(reduced_lab)
    show_img(reduced_rgb, f"img reduced from 255 to {d_range}")
# 

#%%

# just truncated the top frequencies colors
for N_trim in [5000, 1000, 500]:
    count_colors(lab_img, N_trim)
# 

#%%

# doing both
for d_range in [50, 10, 5]:
    reduced_lab = re_round(lab_img, d_range)
    for N_trim in [5000, 1000, 500, 250, 100, 20]:
        count_colors(reduced_lab, N_trim, 
                     plot_title = f"discrete range of {d_range} truncated to {N_trim} colors", 
                     plot_frequency = N_trim == 5000)
# 

#%% 

# just K Means on Lab
for n_clusters in [2, 3, 4, 5, 7, 10, 20, 30, 50]:
    kmeans_clustering(lab_img, n_clusters = n_clusters)
    kmeans_clustering(lab_img, n_clusters = n_clusters, 
                      unique_colors = True, plot_title = "unique colors only")
# 
# -> 5 - 10 clusters is fine
# -> unique colors to avoid the background corrupting the focus color
# but less perception range overall

#%%

# rounding values -> K Means
for d_range in [50, 30, 10]:
    reduced_lab = re_round(lab_img, d_range)
    for n_clusters in [2, 3, 4, 5, 7, 10, 20, 30, 50]:
        kmeans_clustering(reduced_lab, n_clusters = n_clusters, 
                          plot_title = f"discrete range of {d_range}")
# 
# -> eh

#%% 

# testing img resize
for n_clusters in [2, 3, 4, 5, 7, 10, 20, 30, 50]:
    for resize_factor in [10, 25, 50, 75, 100]:
        scale_percent = 60 # percent of original size
        width = int(lab_img.shape[1] * scale_percent / 100)
        height = int(lab_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        resized = cv2.resize(lab_img, dim, interpolation = cv2.INTER_LANCZOS4)
    
        #kmeans_clustering(resized, lab_img, n_clusters = n_clusters)
        kmeans_clustering(resized, lab_img, n_clusters = n_clusters, unique_colors = True, 
                          plot_title = f"unique colors only and using f{resize_factor}% of img")
# 
# -> does nothing much in color choice
    

#%%


# testing blur, sharpen, all combinations
for n_clusters in [2, 4, 5, 8, 10, 25, 50]:
    for blur_sigma in [2, 50, 100, 150, 300]:
        blur = cv2.bilateralFilter(img, 9, blur_sigma, blur_sigma)
        test_lab_img = convert_to_Lab(blur)
        
        result = kmeans_clustering(test_lab_img, test_lab_img, n_clusters = n_clusters, unique_colors = True, 
                                   plot_title = f"unique colors only and blur of {blur_sigma}")
        temp = unsharp_mask(result, amount = 1)
        #show_img(convert_to_RGB(temp), title = f"unique colors only and blur of {blur_sigma} sharp")
        
# 
# -> hell yeah that actually looks decent
# at about n_clusters 5-10 and blur_sigma 100-150-300 : 8 x 150

# rgb -> filter -> lab -> fit on lab [unique] predict on blur is ok
# rgb -> filter -> lab -> fit on blur [unique] predict on blur is ok (same?)

# rgb -> filter -> lab -> fit on blur [NOT unique] predict on blur has some color bleeding
# rgb -> filter -> lab -> fit on lab [NOT unique] predict on blur has some color bleeding (same?)

# rgb -> filter -> lab -> fit on blur predict on lab has no blur obv
# though lab on lab > blur on lab slightly

# 150 unsharp or 300 sharp -> actually both sharp?

# -> not useful to kmeans -> sharpen -> kmeans again

# -> sharpen of amount 2 for 150 blur sigma, 1 for 300?

# -> didnt like sharpening before kmeans


#%% 

























































































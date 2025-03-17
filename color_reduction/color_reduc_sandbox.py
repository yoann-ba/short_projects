






#%% imports

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import cv2
import colour

from sklearn.cluster import KMeans


#%% util functions

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
    return colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(rgb_img))

# 
def convert_Lab_to_RGB(lab_img):
    
    # alternatively
    return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))



#%% core functions


# 
# K Means clustering in Lab space
# specify the nb of clusters
# returns img of same size with cluster Lab pixel values
def kmeans_clustering(lab_img_color, n_clusters = 20, 
                      plot_title = '', unique_colors = False):
    
    # ---- Data prep
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
    
    # ---- Fit
    kmeans = KMeans(n_clusters = n_clusters, n_init = 'auto')
    kmeans.fit(color_list)
    
    # for every cluster, Lab space coords
    cluster_centers = kmeans.cluster_centers_
    
    # ---- Predict
    # for every pixel, cluster id
    cluster_indexes = kmeans.predict(full_color_list)
    # for every pixel, cluster Lab coords
    clustered_values = cluster_centers[cluster_indexes]
    
    new_img = clustered_values.reshape(lab_img_color.shape)
    
    return new_img, cluster_centers


# 
def crop_img(img, levels):
    
    row_step = img.shape[0] // (2**levels)
    col_step = img.shape[1] // (2**levels)
    
    crop_list = []
    for i in range(2**levels):
        for j in range(2**levels):
            crop_list.append(img[i*row_step:(i+1)*row_step, j*col_step:(j+1)*col_step])
            #show_img(crop_list[-1])
    
    return crop_list

# 
def merge_crops(crop_list):
    
    levels = int(np.log(len(crop_list))/np.log(4))
    rows = []
    for i in range(2**levels):
        row = crop_list[i*(2**levels):i*(2**levels)+2**levels]
        row = np.hstack(row)
        rows.append(row)
    img = np.vstack(rows)
    #show_img(img)
    
    return img



#%% main run


def run(rgb_img, lab_img, 
        n_clusters = 20, title = '', unique_colors = False):
    
    reduced, lab_clusters = kmeans_clustering(lab_img, 
                                              n_clusters = n_clusters, 
                                              plot_title = title, 
                                              unique_colors = unique_colors)
    # 
    rgb_reduced = convert_Lab_to_RGB(reduced)
    
    #show_img(rgb_img, title=title)
    #show_img(rgb_reduced, title=title)
    
    return rgb_reduced


# 
def run_pyramid(rgb_img, lab_img, 
                n_clusters = 5, title = '', unique_colors = True, 
                levels = 2):
    
    rgb_img_list = crop_img(rgb_img, levels=levels)
    lab_img_list = crop_img(lab_img, levels=levels)
    # 
    result_list = []
    for i_img in range(len(rgb_img_list)):
        rgb, lab = rgb_img_list[i_img], lab_img_list[i_img]
        result_list.append(run(rgb, lab, n_clusters, title, unique_colors))
    # 
    merged_rgb = merge_crops(result_list)
    
    show_img(rgb_img)
    show_img(merged_rgb)
    
    return merged_rgb



#%% sandbox


img_path = "E:\Python_Data\general_img_db/"
img_path = "E:\Python_Data\general_img_db/anime_comics_fantasy_game/lol_skins/"

img_name = "Leona_MechaKingdomsSkin.jpg"

img_path = Path(img_path + img_name)

rgb_img = colour.read_image(img_path)

oklab_img = convert_RGB_to_Lab(rgb_img)



#TODO vary the nb of colors according to how important it is (eg saurons ring)






























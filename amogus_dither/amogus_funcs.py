


#%% imports

import numpy as np
import matplotlib.pyplot as plt

from time import time

from sortedcontainers import SortedList

import cv2

from pathlib import Path

from sklearn.cluster import KMeans


#%% globals

screen_dpi = 96

# amogus indices
# in a (5, 4) matrix
bg_ind = np.array([[0, 0], 
                   [3, 0], 
                   [4, 0], 
                   [4, 2]])
visor_ind = np.array([[1, 2], 
                      [1, 3]])
body_ind = np.array([[0, 1], 
                     [0, 2], 
                     [0, 3], 
                     [1, 0], 
                     [1, 1], 
                     [2, 0], 
                     [2, 1], 
                     [2, 2], 
                     [2, 3], 
                     [3, 1], 
                     [3, 2], 
                     [3, 3], 
                     [4, 1], 
                     [4, 3]])


#%% funcs

# 
def show_img(img, title = '', size_scaler = 1):
    
    size = (size_scaler*img.shape[0]/screen_dpi, size_scaler*img.shape[1]/screen_dpi)
    fig = plt.figure(figsize = size, dpi = screen_dpi, frameon = True)
    
    plt.imshow(img)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
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



#%% core functions

# gets point, can use memory or not, adds square to memory
def get_random_point(target_area, hit_memory = []):
    
    ind = [np.random.randint(0, target_area[0]), np.random.randint(0, target_area[1])]
    if ind in hit_memory:
        return -1
    index_list = [[ind[0] +k, ind[1] +j] for k in range(-4, 5) for j in range(-3, 4)]
    hit_memory += index_list
    return ind


# use a KMeans to make 2 clusters
# use its inertia for the transform proba
# use its color counts to give closest color to each cluster
# > to body and visor
def apply_amogus_prob_kmeans(img_crop):
    
    new_crop = np.copy(img_crop)
    debug_crop = np.copy(img_crop)
    crop_lab = convert_to_Lab(new_crop)
    
    color_std = np.std(crop_lab, axis = (0, 1))
    L_std = color_std[0]
    # tune for visibility : 
    # - strict L_std < 5, random.choice([-2, -1, 1, 2])
    # - visible pattern L_std < high, random.choice(high vals)
    if L_std < 5:
        val_list = [-5, -4, -3, 3, 4, 5] # for high visibility
        # val_list = [-2, -1, 1, 2] # for sneaky
        noise_vals = np.random.choice(val_list, (5, 4, 3))
        color_list = crop_lab.astype(int) + noise_vals
        color_list = np.clip(color_list, 0, 255)
        color_list = color_list.astype(np.uint8)
        color_list = color_list.reshape((5*4, 3))
    else:
        color_list = crop_lab.reshape((5*4, 3))
    
    # fit a KMeans of 2 clusters on the Lab colors of the crop
    kmeans = KMeans(n_clusters = 2, n_init = 'auto')
    kmeans.fit(color_list)
    
    # use kmeans inertia to get how well the crop fits to 2 clusters
    # 0 inertia = 100%, 20k = 50%?
    amogus_proba = (50 - 100)/20000*kmeans.inertia_ + 100
    amogus_proba = np.clip(amogus_proba, 0, 100)/100
    
    if np.random.random() >= amogus_proba:
        return new_crop, 0, debug_crop
    
    # get the Lab colors and distance to its cluster, per cluster
    list_c0, diff_c0 = [], []
    list_c1, diff_c1 = [], []
    for i in range(len(kmeans.labels_)):
        l = kmeans.labels_[i]
        if l == 1:
            list_c1.append(color_list[i])
            diff_c1.append(np.average((color_list[i] - kmeans.cluster_centers_[1])**2))
        if l == 0:
            list_c0.append(color_list[i])
            diff_c0.append(np.average((color_list[i] - kmeans.cluster_centers_[0])**2))
    # 
    
    # get the color closest to each cluster for selection
    # then RGB it
    if len(list_c0) == 0:
        return new_crop, 0, debug_crop
    if len(list_c1) == 0:
        return new_crop, 0, debug_crop
    closest_c0 = list_c0[np.argmin(diff_c0)]
    closest_c1 = list_c1[np.argmin(diff_c1)]
    
    closest_c0 = convert_to_RGB(np.array([[closest_c0]], dtype = np.uint8))[0][0]
    closest_c1 = convert_to_RGB(np.array([[closest_c1]], dtype = np.uint8))[0][0]
    closest_c0 = list(closest_c0)
    closest_c1 = list(closest_c1)
    
    # give the cluster with more colors to the body
    if len(list_c0) >= len(list_c1):
        body_color = closest_c0
        visor_color = closest_c1
    else:
        body_color = closest_c1
        visor_color = closest_c0
    
    # apply
    new_crop[tuple(visor_ind.T)] = visor_color
    new_crop[tuple(body_ind.T)] = body_color
    
    debug_crop[tuple(visor_ind.T)] = [255, 255, 255]
    debug_crop[tuple(body_ind.T)] = [255, 0, 0]
    
    return new_crop, 1, debug_crop


#%% macro func


# 
def main_amogus(img_crop, method = "kmeans"):
    
    if method == "kmeans":
        new_crop, nb, debug_crop = apply_amogus_prob_kmeans(img_crop)
    elif method == "baseline":
        print("method {method} not supported in this file, running KMeans version")
        new_crop, nb, debug_crop = apply_amogus_prob_kmeans(img_crop)
    elif method == "lab_proba":
        print("method {method} not supported in this file, running KMeans version")
        new_crop, nb, debug_crop = apply_amogus_prob_kmeans(img_crop)
    # 
    
    return new_crop, nb, debug_crop


# orchestrates the main loop of finding a spot, processing crop, applying
def main_run(img, method = "kmeans", nb_iter = 5000, size_scaler = 1):
    
    print("image shape", img.shape)
    target_area = (img.shape[0] - 5, img.shape[1] - 4)
    print("target area", target_area)
    hit_memory = SortedList()

    new_img = np.copy(img)
    debug_img = np.copy(img)
    #show_img(new_img, title = "original img", size_scaler = size_scaler)

    nb_amogus = 0
    for i in range(nb_iter):
        
        ind = get_random_point(target_area, hit_memory)
        if ind == -1:
            # print("index in hit memory, skip")
            continue
        img_crop = np.copy(new_img[ind[0]:ind[0]+5, ind[1]:ind[1]+4])
        
        new_crop, temp_nb, debug_crop = main_amogus(img_crop, method=method)
        nb_amogus += temp_nb
        
        new_img[ind[0]:ind[0]+5, ind[1]:ind[1]+4] = new_crop
        debug_img[ind[0]:ind[0]+5, ind[1]:ind[1]+4] = debug_crop
        
        # if i%10 == 0:
        #     show_img(new_img, size_scaler = 5)
    # 
    #show_img(new_img, title = f"{nb_amogus} amoguses", size_scaler = size_scaler)   
    print("nb amogus made:", nb_amogus)
    #show_img(debug_img, title = f"{nb_amogus} amoguses (debug)", size_scaler = size_scaler)
    
    return new_img, nb_amogus, debug_img


# cuts img in 4 patches to run faster
# leaves a somewhat visible 4 pixel seam
# could be made into a recursive loop but it's fine for pictures <4k
def run_as_patches(img, method="kmeans", nb_iter=10000, size_scaler=2):
    
    half_row = img.shape[0]//2 + 1
    half_col = img.shape[1]//2 + 1
    
    TL = np.copy(img[0:half_row, 0:half_col])
    TR = np.copy(img[0:half_row, half_col:])
    BL = np.copy(img[half_row:, 0:half_col])
    BR = np.copy(img[half_row:, half_col:])
    
    total_nb = 0
    TL, temp_nb, tl_d = main_run(TL, method, nb_iter, size_scaler*2)
    total_nb += temp_nb
    TR, temp_nb, tr_d = main_run(TR, method, nb_iter, size_scaler*2)
    total_nb += temp_nb
    BL, temp_nb, bl_d = main_run(BL, method, nb_iter, size_scaler*2)
    total_nb += temp_nb
    BR, temp_nb, br_d = main_run(BR, method, nb_iter, size_scaler*2)
    total_nb += temp_nb
    
    # for some reason np.block doesnt like it?
    # new_img = np.block([
    #     [TL, TR], 
    #     [BL, BR]])
    
    # but this works even tho it should be the same?
    new_img = np.vstack((np.hstack((TL, TR)), np.hstack((BL, BR))))
    debug_img = np.vstack((np.hstack((tl_d, tr_d)), np.hstack((bl_d, br_d))))
    
    show_img(img, "original img", size_scaler)
    show_img(new_img, f"{total_nb} amoguses", size_scaler)
    show_img(debug_img, f"{total_nb} amoguses (debug)", size_scaler)
    
    return new_img, total_nb, debug_img


#%% main


path = "E:\Python_Data/general_img_db/anime_comics_fantasy_game/"
path = "E:\Python_Data/real_birthday/"
path += "IMG-20240503-WA0003_crop_face_x0_5.jpg"

path = Path(path)

img = cv2.imread(str(path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#%%

time_st = time()

test, nb, dbg = run_as_patches(img, method = "kmeans", nb_iter = 50000, size_scaler = 4)
print(f"time taken: {time() - time_st:.2f}")



































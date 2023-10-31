




#%% imports

import numpy as np
import matplotlib.pyplot as plt

import cv2
import colour

from pathlib import Path

from scipy.spatial import KDTree
import scipy.signal as sig



#%% util functions

# 
def show_img(img, size_scaler = 1, title = '', cmap = ''):
    
    screen_dpi = 96
    temp_size = (int(img.shape[0]*size_scaler/plt.rcParams['figure.dpi']), 
                 int(img.shape[1]*size_scaler/plt.rcParams['figure.dpi']))
    
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
def make_01(img):
    
    return (img - np.min(img))/(np.max(img) - np.min(img))


# 
# put to 01 then in proba format
def make_hole_kernel(radius_size):
    
    diameter = 2*radius_size + 1
    hole_kernel = np.zeros((diameter, diameter))
    for i in range(len(hole_kernel)):
        for j in range(len(hole_kernel[i])):
            hole_kernel[i][j] = np.sqrt((radius_size-i)**2 + (radius_size-j)**2)
    hole_kernel = make_01(hole_kernel)
    hole_kernel = hole_kernel/np.sum(hole_kernel)
    
    return hole_kernel


#%% core funcs


# 
def point_iteration(lab_img, storage_pt_coords, diff_threshold, size_scaler):
    
    new_pt_coords = []
    deletion_list = []
    # temp_store_diffs = []
    
    tree = KDTree(storage_pt_coords)
    nbr_dist, nbr_ind = tree.query(storage_pt_coords, k = [2])
    nbr_coords = storage_pt_coords[nbr_ind]
    
    for i in range(len(storage_pt_coords)):
        first_coord = storage_pt_coords[i]
        first_lab = lab_img[first_coord[0], first_coord[1]]
        
        nbr_coord = nbr_coords[i][0]
        nbr_lab = lab_img[nbr_coord[0], nbr_coord[1]]
        
        diff = np.linalg.norm(first_lab - nbr_lab)
        
        if diff > diff_threshold:
            if np.linalg.norm(first_coord - nbr_coord) > 2:
                # if first_coord not in new_pt_coords:
                new_pt_coords.append(list(first_coord))
                # if nbr_coord not in new_pt_coords:
                new_pt_coords.append(list(nbr_coord))
            else:
                mid = ((nbr_coord + first_coord)*0.5).astype(int)
                # if mid not in new_pt_coords:
                new_pt_coords.append(list(mid))
        else:
            mid = ((nbr_coord + first_coord)*0.5).astype(int)
            # if mid not in new_pt_coords:
            new_pt_coords.append(list(mid))
        # 
    # 
    new_pt_coords = np.unique(new_pt_coords, axis = 0)
    first_return = new_pt_coords
    print(f"reduced to {len(new_pt_coords)} pts")
    
    # test_img = np.copy(rgb_img)
    # for pt in new_pt_coords:
    #     test_img[pt[0], pt[1]] = [1, 0, 0]
    # show_img(test_img, size_scaler = size_scaler+1, 
    #           title = f"iter {iter}, {len(new_pt_coords)} pts")
    
    # canvas_img = [np.linspace([row_i, 0], [row_i, rgb_img.shape[1]-1], rgb_img.shape[1]).astype(int)
    #               for row_i in range(0, rgb_img.shape[0])]
    # canvas_img = np.array(canvas_img)

    # dist_map, ind_map = tree.query(canvas_img)
    # show_img(ind_map, size_scaler = size_scaler+1)
    
    
    # --- Deletion of close points
    
    storage_pt_coords = new_pt_coords
    
    tree = KDTree(storage_pt_coords)
    nbr_dist, nbr_ind = tree.query(storage_pt_coords, k = [2])
    nbr_coords = storage_pt_coords[nbr_ind]
    
    replacement_dict = {}
    for i in range(len(storage_pt_coords)):
        first_coord = storage_pt_coords[i]
        nbr_coord = nbr_coords[i][0]
        
        if np.linalg.norm(first_coord - nbr_coord) < 2:
            mid = ((nbr_coord + first_coord)*0.5).astype(int)
            replacement_dict[f"{list(first_coord)}"] = list(mid)
            replacement_dict[f"{list(nbr_coord)}"] = list(mid)
    # 
    for i in range(len(new_pt_coords)):
        if f"{list(new_pt_coords[i])}" in replacement_dict:
            new_pt_coords[i] = replacement_dict[f"{list(new_pt_coords[i])}"]
    #
    new_pt_coords = np.unique(new_pt_coords, axis = 0)
    print(f"reduced to {len(new_pt_coords)} pts")
    storage_pt_coords = new_pt_coords
    
    # test_img = np.copy(rgb_img)
    # for pt in new_pt_coords:
    #     test_img[pt[0], pt[1]] = [1, 0, 0]
    # show_img(test_img, size_scaler = size_scaler+1, 
    #           title = f"iter {iter}, {len(new_pt_coords)} pts")
    
    # canvas_img = [np.linspace([row_i, 0], [row_i, rgb_img.shape[1]-1], rgb_img.shape[1]).astype(int)
    #               for row_i in range(0, rgb_img.shape[0])]
    # canvas_img = np.array(canvas_img)

    # dist_map, ind_map = tree.query(canvas_img)
    # show_img(ind_map, size_scaler = size_scaler+1)
    
    return first_return, storage_pt_coords


# 
def point_generation(img, lab_img, nb_pts, nb_iter = 40, diff_threshold = 0.05):
    
    rand_ind_0 = np.random.randint(0, img.shape[0], size = nb_pts)
    rand_ind_1 = np.random.randint(0, img.shape[1], size = nb_pts)
    rand_pt_coords = np.c_[rand_ind_0, rand_ind_1]
    
    storage_pt_coords = np.copy(rand_pt_coords)
    for iter in range(nb_iter):
        _, storage_pt_coords = point_iteration(lab_img, storage_pt_coords, 
                                            diff_threshold = diff_threshold)
    # 
    tree = KDTree(storage_pt_coords)
    return storage_pt_coords, tree

# 
# gives a map of same size as img with the distance to closest pt
# and another with the indice of that pt
def distance_indices_maps(img, tree):
    
    canvas_img = [np.linspace([row_i, 0], [row_i, img.shape[1]-1], img.shape[1]).astype(int)
                  for row_i in range(0, img.shape[0])]
    canvas_img = np.array(canvas_img)

    dist_map, ind_map = tree.query(canvas_img)
    
    return dist_map, ind_map


# 
# builds {pt indice: float RGB value of Lab avg of pt's region}
def build_colours_per_pt(ind_map, lab_img):
    
    colours_per_pt = {}
    for i in range(0, ind_map.shape[0]):
        for j in range(0, ind_map.shape[1]):
            key = ind_map[i, j] # region id of the pixel
            lab_col = lab_img[i, j] # color of the pixel
            if key not in colours_per_pt:
                colours_per_pt[key] = []
            colours_per_pt[key].append(lab_col)
    # 
    avg_col_per_pt = {key:np.average(colours_per_pt[key], axis = 0) for key in colours_per_pt}
    avg_rgb_per_pt = {key:colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(avg_col_per_pt[key])) 
                      for key in avg_col_per_pt}
    
    return avg_rgb_per_pt


# 
def build_mosaic(img, ind_map, avg_rgb_per_pt):
    
    mosaic_img = np.copy(img)
    for i in range(0, mosaic_img.shape[0]):
        for j in range(0, mosaic_img.shape[1]):
            mosaic_img[i][j] = avg_rgb_per_pt[ind_map[i, j]]
    # 
    
    return mosaic_img


#%% main


# 
def main_run(rgb_img, oklab_img, title = '', nb_pts = 10000):
    
    rand_pt_coords, tree = point_generation(rgb_img, oklab_img, 
                                            nb_pts = nb_pts)
    dist_map, ind_map = distance_indices_maps(rgb_img, tree)
    avg_rgb_per_pt = build_colours_per_pt(ind_map, oklab_img)
    mosaic_img = build_mosaic(rgb_img, ind_map, avg_rgb_per_pt)
    
    show_img(rgb_img, title = title, size_scaler = 1)
    new_title = title
    show_img(mosaic_img, title = new_title, size_scaler = 1)
    
    return mosaic_img


# 
def run_details_iters(rgb_img, oklab_img, title = '', 
                      nb_pts = 10_000, nb_iters = 30, 
                      diff_threshold = 0.1, nb_plot = 15, 
                      size_scaler = 1):
    # 
    show_img(rgb_img, title = title, size_scaler = size_scaler)
    rand_ind_0 = np.random.randint(0, rgb_img.shape[0], size = nb_pts)
    rand_ind_1 = np.random.randint(0, rgb_img.shape[1], size = nb_pts)
    rand_pt_coords = np.c_[rand_ind_0, rand_ind_1]
    
    storage_pt_coords = np.copy(rand_pt_coords)
    for i_iter in range(nb_iters):
        print("step", i_iter)
        first_ret, storage_pt_coords = point_iteration(oklab_img, storage_pt_coords, 
                                            diff_threshold = diff_threshold, 
                                            size_scaler = size_scaler)
        
        if i_iter > nb_plot:
            # tree = KDTree(first_ret)
            # dist_map, ind_map = distance_indices_maps(rgb_img, tree)
            # avg_rgb_per_pt = build_colours_per_pt(ind_map, oklab_img)
            # mosaic_img = build_mosaic(rgb_img, ind_map, avg_rgb_per_pt)
            # show_img(mosaic_img, title = title+'first return', size_scaler = size_scaler)
            
            tree = KDTree(storage_pt_coords)
            dist_map, ind_map = distance_indices_maps(rgb_img, tree)
            avg_rgb_per_pt = build_colours_per_pt(ind_map, oklab_img)
            mosaic_img = build_mosaic(rgb_img, ind_map, avg_rgb_per_pt)
            show_img(mosaic_img, title = title+f'\nstep {i_iter}, {len(storage_pt_coords)} pts, no close pts', size_scaler = size_scaler)
    # 
    
    return mosaic_img


# 
def run_details_resize(rgb_img, oklab_img, title = '', 
                      nb_pts = 10_000, nb_iters = 30, 
                      diff_threshold = 0.1, nb_plot = 15, 
                      size_scaler = 1, 
                      resize_factor = 2):
    
    temp_rgb = cv2.resize(rgb_img, 
                          dsize = (int(rgb_img.shape[1]*resize_factor), 
                                   int(rgb_img.shape[0]*resize_factor)), 
                          interpolation = cv2.INTER_CUBIC)
    temp_lab = cv2.resize(oklab_img, 
                          dsize = (int(rgb_img.shape[1]*resize_factor), 
                                   int(rgb_img.shape[0]*resize_factor)), 
                          interpolation = cv2.INTER_CUBIC)
    mosaic_img = run_details_iters(rgb_img = temp_rgb, 
                                   oklab_img = temp_lab, 
                                   title = title, 
                                   nb_pts = nb_pts, 
                                   nb_iters = nb_iters, 
                                   diff_threshold = diff_threshold, 
                                   nb_plot = nb_plot, 
                                   size_scaler = size_scaler/resize_factor)
    mosaic_img = cv2.resize(mosaic_img, 
                            dsize = (rgb_img.shape[1], rgb_img.shape[0]), 
                            interpolation = cv2.INTER_CUBIC)
    show_img(mosaic_img)
    
    return mosaic_img


#TODO make diff threshold into a proba?
#TODO make gif of mosaic process and zone process
#TODO try taking the delaunay triang of the points
# fix the touching cluster pb for real -> done but oof the solution

# (could keep the other color info as a shade, dither, gradient?)
# impact the shade by external info (2 colors external swatch?)


#%% sandbox


img_path = "E:\Python_Data\general_img_db" + "/alex_dog.jpg"

img_path = "E:\Python_Data\general_img_db/anime_comics_fantasy_game/"
img_name = "the_wrath_gate_by_jasper_zhao.jpg"
img_path = Path(img_path + img_name)

rgb_img = colour.read_image(img_path)

xyz_img = colour.sRGB_to_XYZ(rgb_img)
oklab_img = colour.XYZ_to_Oklab(xyz_img)
cielab_img = colour.XYZ_to_Lab(xyz_img)


#%%

nb_points = 4000
rand_ind_0 = np.random.randint(0, rgb_img.shape[0], size = nb_points)
rand_ind_1 = np.random.randint(0, rgb_img.shape[1], size = nb_points)
rand_pt_coords = np.dstack((rand_ind_0, rand_ind_1))
rand_pt_coords = np.squeeze(rand_pt_coords)

test_img = np.copy(rgb_img)
for pt in rand_pt_coords:
    test_img[pt[0], pt[1]] = [1, 0, 0]
show_img(rgb_img, size_scaler = 2, title = 'original')
show_img(test_img, size_scaler = 2, title = 'dots')


#%%

tree = KDTree(rand_pt_coords)
# tree.query(coords, k = nb of closest pts), tree.data[pt indice]


#%%


from sortedcontainers import SortedList

storage_pt_coords = np.copy(rand_pt_coords)

diff_threshold = 0.1
for iter in range(10):
    new_pt_coords = []
    # temp_store_diffs = []
    
    tree = KDTree(storage_pt_coords)
    nbr_dist, nbr_ind = tree.query(storage_pt_coords, k = [2])
    nbr_coords = storage_pt_coords[nbr_ind]
    
    for i in range(len(storage_pt_coords)):
        first_coord = storage_pt_coords[i]
        first_lab = oklab_img[first_coord[0], first_coord[1]]
        
        nbr_coord = nbr_coords[i][0]
        nbr_lab = oklab_img[nbr_coord[0], nbr_coord[1]]
        
        diff = np.linalg.norm(first_lab - nbr_lab)
        
        # debug/calibration ----------
        # temp_store_diffs.append(diff)
        
        # swatch = np.full((10, 5, 3), first_lab)
        # swatch2 = np.full((10, 5, 3), nbr_lab)
        # swatch = np.hstack((swatch, swatch2))
        
        # swatch = colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(swatch))
        # plt.imshow(swatch)
        # plt.title(f'{diff:.2f}')
        # plt.show()
        # -------------------
        
        if diff > diff_threshold:
            if np.linalg.norm(first_coord - nbr_coord) > 2:
                # if first_coord not in new_pt_coords:
                new_pt_coords.append(list(first_coord))
                # if nbr_coord not in new_pt_coords:
                new_pt_coords.append(list(nbr_coord))
            else:
                mid = ((nbr_coord + first_coord)*0.5).astype(int)
                # if mid not in new_pt_coords:
                new_pt_coords.append(list(mid))
        else:
            mid = ((nbr_coord + first_coord)*0.5).astype(int)
            # if mid not in new_pt_coords:
            new_pt_coords.append(list(mid))
        # 
    # 
    new_pt_coords = np.unique(new_pt_coords, axis = 0)
    
    test_img = np.copy(rgb_img)
    for pt in new_pt_coords:
        test_img[pt[0], pt[1]] = [1, 0, 0]
    show_img(test_img, size_scaler = 3, 
              title = f"iter {iter}, {len(new_pt_coords)} pts")
    
    print(f"reduced to {len(new_pt_coords)} pts")
    
    # --- Deletion of close points
    
    storage_pt_coords = new_pt_coords
    
    tree = KDTree(storage_pt_coords)
    nbr_dist, nbr_ind = tree.query(storage_pt_coords, k = [2])
    nbr_coords = storage_pt_coords[nbr_ind]
    
    replacement_dict = {}
    for i in range(len(storage_pt_coords)):
        first_coord = storage_pt_coords[i]
        nbr_coord = nbr_coords[i][0]
        
        if np.linalg.norm(first_coord - nbr_coord) < 2:
            mid = ((nbr_coord + first_coord)*0.5).astype(int)
            replacement_dict[f"{list(first_coord)}"] = list(mid)
            replacement_dict[f"{list(nbr_coord)}"] = list(mid)
    # 
    for i in range(len(new_pt_coords)):
        if f"{list(new_pt_coords[i])}" in replacement_dict:
            new_pt_coords[i] = replacement_dict[f"{list(new_pt_coords[i])}"]
    #
    new_pt_coords = np.unique(new_pt_coords, axis = 0)
    
    test_img = np.copy(rgb_img)
    for pt in new_pt_coords:
        test_img[pt[0], pt[1]] = [1, 0, 0]
    show_img(test_img, size_scaler = 3, 
              title = f"iter {iter}, {len(new_pt_coords)} pts")
    
    print(f"-reduced to {len(new_pt_coords)} pts")
    
    # canvas_img = [np.linspace([row_i, 0], [row_i, rgb_img.shape[1]-1], rgb_img.shape[1]).astype(int)
    #               for row_i in range(0, rgb_img.shape[0])]
    # canvas_img = np.array(canvas_img)

    # dist_map, ind_map = tree.query(canvas_img)
    # show_img(ind_map, size_scaler = 2)
    
    storage_pt_coords = new_pt_coords
# 

















#%% OLD STUFF


# # 
# # version where new random points are re-sent in multiple times
# # was not found to end with a particularly better source grid
# def run_details_iters(rgb_img, oklab_img, title = '', 
#                       nb_pts = 10_000, nb_iters = 10, diff_threshold = 0.1, 
#                       macro_iters = 3, 
#                       size_scaler = 1):
#     # 
#     show_img(rgb_img, title = title, size_scaler = size_scaler)
#     for m_iter in range(macro_iters):
#         rand_ind_0 = np.random.randint(0, rgb_img.shape[0], size = nb_pts)
#         rand_ind_1 = np.random.randint(0, rgb_img.shape[1], size = nb_pts)
#         rand_pt_coords = np.c_[rand_ind_0, rand_ind_1]
        
#         if m_iter == 0:
#             storage_pt_coords = np.copy(rand_pt_coords)
#         else:
#             storage_pt_coords = np.concatenate((storage_pt_coords, rand_pt_coords))
        
#         temp_nb_iters = nb_iters if m_iter < macro_iters-1 else nb_iters*2
#         for i_iter in range(temp_nb_iters):
#             storage_pt_coords = point_iteration(oklab_img, storage_pt_coords, 
#                                                 diff_threshold = diff_threshold, 
#                                                 size_scaler = size_scaler)
#             tree = KDTree(storage_pt_coords)
            
#             if m_iter == macro_iters -1:
#                 if i_iter > 15:
#                     dist_map, ind_map = distance_indices_maps(rgb_img, tree)
#                     avg_rgb_per_pt = build_colours_per_pt(ind_map, oklab_img)
#                     mosaic_img = build_mosaic(rgb_img, ind_map, avg_rgb_per_pt)
#                     show_img(mosaic_img, title = title, size_scaler = size_scaler)
#             #---
#         # 
#     # 
    
#     return mosaic_img









































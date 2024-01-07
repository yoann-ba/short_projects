

# cleaner code
# + subpixel
# + bilinear interpolate the subpixel values
# + make the threshold probabilistic
# + point weight?
# + make gif?



#%% imports

import numpy as np
import matplotlib.pyplot as plt

import cv2
import colour

from pathlib import Path

from scipy.spatial import KDTree
#import scipy.signal as sig


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



#%% core funcs


# 
# 10 coords means we have coords in [row][subpixel_ind] format
# > where the subpixel index is 0 to 9
def point_generation(lab_img, nb_pts, seed = 123456):
    
    rng = np.random.default_rng(seed = seed)
    rand_pt_10coords_row = rng.integers(0, (lab_img.shape[0]-1)*10, size = nb_pts)
    rand_pt_10coords_col = rng.integers(0, (lab_img.shape[1]-1)*10, size = nb_pts)
    rand_pt_10coords = np.c_[rand_pt_10coords_row, rand_pt_10coords_col]
    
    return rand_pt_10coords


# takes in a 10coord point and returns row/col indices, and [0-1[ alpha indices
def extract_coord(pt_10coords):
    
    return [pt_10coords[0]//10, pt_10coords[1]//10, 
            pt_10coords[0]%10/10, pt_10coords[1]%10/10]

# takes in row/col pixel indices, and [0-1[ alpha indices
def build_coord(i_row, i_col, alpha_row, alpha_col):
    
    return [int(i_row*10+alpha_row*10), int(i_col*10+alpha_col*10)]

# 
# takes a 4 pixel square, the [0-1[ alpha indices and interpolates
def bilinear_interp(lab_img_square, alpha_row, alpha_col):
    
    top = (1-alpha_col)*lab_img_square[0][0] + alpha_col*lab_img_square[0][1]
    bot = (1-alpha_col)*lab_img_square[1][0] + alpha_col*lab_img_square[1][1]
    center = (1-alpha_row)*top + alpha_row*bot
    
    return center


# 
def point_iteration(lab_img, pt_10coords, diff_threshold, size_scaler):
    
    new_pt_coords = []
    
    tree = KDTree(pt_10coords)
    nbr_dist, nbr_ind = tree.query(pt_10coords, k = [2])
    nbr_coords = pt_10coords[nbr_ind]
    
    for i in range(len(pt_10coords)):
        first_coord = pt_10coords[i]
        nbr_coord = nbr_coords[i][0]
        
        first_extract = extract_coord(first_coord)
        first_square = lab_img[first_extract[0]:first_extract[0]+2, first_extract[1]:first_extract[1]+2]
        first_lab = bilinear_interp(first_square, first_extract[2], first_extract[3])
        
        nbr_extract = extract_coord(nbr_coord)
        nbr_square = lab_img[nbr_extract[0]:nbr_extract[0]+2, nbr_extract[1]:nbr_extract[1]+2]
        nbr_lab = bilinear_interp(nbr_square, nbr_extract[2], nbr_extract[3])
        
        diff = np.linalg.norm(first_lab - nbr_lab)
        
        #TODOrng
        if diff > diff_threshold:
            new_pt_coords.append(list(first_coord))
            new_pt_coords.append(list(nbr_coord))
        else:
            mid = ((nbr_coord + first_coord)*0.5).astype(int)
            new_pt_coords.append(list(mid))
        # 
    # 
    new_pt_coords = np.unique(new_pt_coords, axis = 0)
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
        
        if np.linalg.norm(first_coord - nbr_coord) < 15:
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
    
    return storage_pt_coords


# 
# gives a map of same size as img with the distance to closest pt
# and another with the indice of that pt
def distance_indices_maps(lab_img, tree):
    
    canvas_img = [np.linspace([row_i, 0], [row_i, lab_img.shape[1]-1], lab_img.shape[1]).astype(int)
                  for row_i in range(0, lab_img.shape[0])]
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


#%% main run

# 
def run_details_iters(rgb_img, lab_img, title = '', 
                      nb_pts = 10_000, seed = None, 
                      nb_iters = 30, nb_plot = 15, 
                      diff_threshold = 0.1,
                      size_scaler = 1):
    # 
    show_img(rgb_img, title = title, size_scaler = size_scaler)
    pt_10coords = point_generation(lab_img, nb_pts, seed)
    
    storage_pt_coords = np.copy(pt_10coords)
    for i_iter in range(nb_iters):
        print("step", i_iter)
        storage_pt_coords = point_iteration(lab_img, storage_pt_coords, 
                                            diff_threshold = diff_threshold, 
                                            size_scaler = size_scaler)
        if i_iter > nb_plot:
            tree = KDTree(storage_pt_coords/10) # shift to pixel indices for the rest
            dist_map, ind_map = distance_indices_maps(lab_img, tree)
            avg_rgb_per_pt = build_colours_per_pt(ind_map, lab_img)
            mosaic_img = build_mosaic(rgb_img, ind_map, avg_rgb_per_pt)
            show_img(mosaic_img, title = title+f'\nstep {i_iter}, {len(storage_pt_coords)} pts, no close pts', size_scaler = size_scaler)
    # 
    
    return mosaic_img


#%% sandbox


img_path = "E:\Python_Data\general_img_db/"
img_name = "test_bear.jpg"

img_path = Path(img_path + img_name)

rgb_img = colour.read_image(img_path)

xyz_img = colour.sRGB_to_XYZ(rgb_img)
oklab_img = colour.XYZ_to_Oklab(xyz_img)
cielab_img = colour.XYZ_to_Lab(xyz_img)



































































































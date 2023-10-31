





#%% imports

import numpy as np
import matplotlib.pyplot as plt

import cv2
import colour

from pathlib import Path

from scipy.spatial import KDTree
import scipy.signal as sig


#%% paths




#%% util functions

# 
def show_img(img, size_scaler = 1, title = '', cmap = ''):
    
    screen_dpi = 96
    temp_size = (img.shape[0]*size_scaler/screen_dpi, img.shape[1]*size_scaler/screen_dpi)
    plt.figure(figsize = temp_size)
    
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
# choose random drop, or Lab gradient-based random pts
# noise reductor goes up -> less pts
def point_generation(img, lab_img, nb_pts, method = 'gradient', 
                     noise_reductor = 2, grad_amplifier = 1, 
                     hole_convol_radius_size = 10):
    
    if method == 'random_grid':
        rand_ind_0 = np.random.randint(0, img.shape[0], size = nb_pts)
        rand_ind_1 = np.random.randint(0, img.shape[1], size = nb_pts)
        rand_pt_coords = np.dstack((rand_ind_0, rand_ind_1))
        rand_pt_coords = np.squeeze(rand_pt_coords)
    if method == 'gradient_threshold':
        # make all gradients
        grad_L = np.gradient(lab_img[:, :, 0])
        grad_a = np.gradient(lab_img[:, :, 1])
        grad_b = np.gradient(lab_img[:, :, 2])
        # add abs(horizontal) + abs(vertical)
        grad_L = np.abs(grad_L[0]) + np.abs(grad_L[1])
        grad_a = np.abs(grad_a[0]) + np.abs(grad_a[1])
        grad_b = np.abs(grad_b[0]) + np.abs(grad_b[1])
        # merge all three dim, normalized
        global_grad = make_01(grad_L) + make_01(grad_a) + make_01(grad_b)
        # make a noise map and use it as threshold
        noise = np.random.randn(img.shape[0], img.shape[1])
        temp_pts = np.greater(global_grad, make_01(noise)*noise_reductor)
        show_img(temp_pts)
        # sub-sample pts
        hole_kernel = make_hole_kernel(radius_size = hole_convol_radius_size)
        temp_pts = sig.convolve2d(temp_pts, hole_kernel, mode = "same")
        show_img(temp_pts)
        rand_pt_coords = np.argwhere(temp_pts)
        temp_inds = np.random.choice(np.arange(0, len(rand_pt_coords)), 
                                     size = nb_pts, replace = False)
        rand_pt_coords = rand_pt_coords[temp_inds]
    if method == 'gradient':
        # make all gradients
        grad_L = np.gradient(lab_img[:, :, 0])
        grad_a = np.gradient(lab_img[:, :, 1])
        grad_b = np.gradient(lab_img[:, :, 2])
        # add abs(horizontal) + abs(vertical)
        grad_L = np.abs(grad_L[0]) + np.abs(grad_L[1])
        grad_a = np.abs(grad_a[0]) + np.abs(grad_a[1])
        grad_b = np.abs(grad_b[0]) + np.abs(grad_b[1])
        # merge all three dim, normalized
        global_grad = make_01(grad_L) + make_01(grad_a) + make_01(grad_b)
        # map of spatial positions
        canvas_img = [np.linspace([row_i, 0], [row_i, img.shape[1]-1], img.shape[1]).astype(int)
                      for row_i in range(0, img.shape[0])]
        canvas_img = np.array(canvas_img)
        # random choice of the positions weighted by the gradient
        global_grad = make_01(global_grad**grad_amplifier)
        show_img(global_grad)
        hole_kernel = make_hole_kernel(radius_size = hole_convol_radius_size)
        global_grad = sig.convolve2d(global_grad, hole_kernel, mode = "same")
        show_img(global_grad)
        gradient_val = global_grad.flatten()/np.sum(global_grad) # proba format
        flat_inds = canvas_img.reshape((canvas_img.shape[0]*canvas_img.shape[1], 2))
        temp_inds = np.random.choice(np.arange(0, len(gradient_val)), nb_pts, False, p=gradient_val)
        temp_inds_2 = flat_inds[temp_inds]

        rand_pt_coords = temp_inds_2
    #
    
    tree = KDTree(rand_pt_coords)
    return rand_pt_coords, tree


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
            key = ind_map[i, j]
            lab_col = lab_img[i, j]
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
def main_run(rgb_img, oklab_img, 
             nb_pts = 2000, noise_reductor = 1, grad_amplifier = 1.5, 
             hole_convol_radius_size = 10, 
             title = '', method = 'gradient'):
    
    rand_pt_coords, tree = point_generation(rgb_img, oklab_img, 
                                            nb_pts = nb_pts, method=method, 
                                            noise_reductor = noise_reductor, 
                                            grad_amplifier=grad_amplifier, 
                                            hole_convol_radius_size=hole_convol_radius_size)
    dist_map, ind_map = distance_indices_maps(rgb_img, tree)
    avg_rgb_per_pt = build_colours_per_pt(ind_map, oklab_img)
    mosaic_img = build_mosaic(rgb_img, ind_map, avg_rgb_per_pt)
    
    show_img(rgb_img, title = title)
    new_title = title+f' {len(rand_pt_coords)} pts, {noise_reductor} noise reduc'
    new_title += f', {grad_amplifier} amplifier, {hole_convol_radius_size} convol radius size'
    show_img(mosaic_img, title = new_title)
    
    return mosaic_img



#%% sandbox


img_path = "E:\Python_Data\general_img_db" + "/alex_dog.jpg"
img_path = Path(img_path)

rgb_img = colour.read_image(img_path)

xyz_img = colour.sRGB_to_XYZ(rgb_img)
oklab_img = colour.XYZ_to_Oklab(xyz_img)
cielab_img = colour.XYZ_to_Lab(xyz_img)


# place in points (grid or semi random?)
# merge neighbouring points if not enough variation
# make voronoi triangles
# compute avg color as color of triangle
# (could keep the other color info as a shade, dither, gradient?)
# impact the shade by external info (2 colors external swatch?)

# replace random grid with np gradient of L, a, b all summed up (2 directions merged)?


#%%

nb_points = 1500
rand_ind_0 = np.random.randint(0, rgb_img.shape[0], size = nb_points)
rand_ind_1 = np.random.randint(0, rgb_img.shape[1], size = nb_points)
rand_pt_coords = np.dstack((rand_ind_0, rand_ind_1))
rand_pt_coords = np.squeeze(rand_pt_coords)

test_img = np.copy(rgb_img)
for pt in rand_pt_coords:
    test_img[pt[0], pt[1]] = [1, 0, 0]
show_img(rgb_img, size_scaler = 2)
show_img(test_img, size_scaler = 2)


#%%

tree = KDTree(rand_pt_coords)
# tree.query(coords, k = nb of closest pts), tree.data[pt indice]


#%%

canvas_img = [np.linspace([row_i, 0], [row_i, rgb_img.shape[1]-1], rgb_img.shape[1]).astype(int)
              for row_i in range(0, rgb_img.shape[0])]
canvas_img = np.array(canvas_img)

dist_map, ind_map = tree.query(canvas_img)

show_img(ind_map)


#%%

colours_per_pt = {}
for i in range(0, rgb_img.shape[0]):
    for j in range(0, rgb_img.shape[1]):
        key = ind_map[i, j]
        lab_col = oklab_img[i, j]
        if key not in colours_per_pt:
            colours_per_pt[key] = []
        colours_per_pt[key].append(lab_col)
# 
avg_col_per_pt = {key:np.average(colours_per_pt[key], axis = 0) for key in colours_per_pt}
avg_rgb_per_pt = {key:colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(avg_col_per_pt[key])) 
                  for key in avg_col_per_pt}

#%%

mosaic_img = np.copy(rgb_img)
for i in range(0, mosaic_img.shape[0]):
    for j in range(0, mosaic_img.shape[1]):
        mosaic_img[i][j] = avg_rgb_per_pt[ind_map[i, j]]
# 
show_img(mosaic_img)


#%%

grad_L = np.gradient(oklab_img[:, :, 0])
grad_a = np.gradient(oklab_img[:, :, 1])
grad_b = np.gradient(oklab_img[:, :, 2])

grad_L = np.abs(grad_L[0]) + np.abs(grad_L[1])
grad_a = np.abs(grad_a[0]) + np.abs(grad_a[1])
grad_b = np.abs(grad_b[0]) + np.abs(grad_b[1])

show_img(grad_L, cmap = 'gray')
show_img(grad_a, cmap = 'gray')
show_img(grad_b, cmap = 'gray')

global_grad = make_01(grad_L) + make_01(grad_a) + make_01(grad_b)
show_img(global_grad, cmap = 'gray')


#%%

noise = np.random.randn(rgb_img.shape[0], rgb_img.shape[1])

temp_pts = np.greater(global_grad, make_01(noise))

show_img(temp_pts, cmap = 'gray')


#%%


gradient_val = make_01(global_grad).flatten()/np.sum(make_01(global_grad))
flat_inds = canvas_img.reshape((canvas_img.shape[0]*canvas_img.shape[1], 2))
temp_inds = np.random.choice(np.arange(0, len(gradient_val)), 100, False, p=gradient_val)
temp_inds_2 = flat_inds[temp_inds]


#%% 


radius_size = 20
diameter = 2*radius_size + 1
hole_kernel = np.zeros((diameter, diameter))
for i in range(len(hole_kernel)):
    for j in range(len(hole_kernel[i])):
        hole_kernel[i][j] = np.sqrt((radius_size-i)**2 + (radius_size-j)**2)
hole_kernel = make_01(hole_kernel)
hole_kernel = hole_kernel/np.sum(hole_kernel)
# show_img(hole_kernel, size_scaler = 20, cmap = 'gray')

aa = sig.convolve2d(global_grad, hole_kernel, mode = "same")

show_img(global_grad, cmap = 'gray')
show_img(aa, cmap = 'gray')






























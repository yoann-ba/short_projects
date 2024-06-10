


#%% imports

import numpy as np
import matplotlib.pyplot as plt

import colour

import os
from pathlib import Path


#%% util functions

# 
def show_img(img, size_scaler = 1, title = '', cmap = '', 
             vmin = None, vmax = None):
    
    screen_dpi = 96
    temp_size = (int(img.shape[0]*size_scaler/plt.rcParams['figure.dpi']), 
                 int(img.shape[1]*size_scaler/plt.rcParams['figure.dpi']))
    # for some reason the dpi is neither the screen dpi or plt params dpi?
    # 55 isnt either but its a close approximation
    temp_size = (int(img.shape[0]*size_scaler/40), 
                 int(img.shape[1]*size_scaler/40))
    plt.figure(figsize = temp_size, dpi = plt.rcParams['figure.dpi'])
    
    if cmap:
        plt.imshow(img, vmin = vmin, vmax = vmax, cmap = cmap)
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
    
    return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))



#%% funcs


# from original mu, std to reference ones
# input axis = (0, 1) vectors for mean and std
def standardise_tf(img, ref_mean, ref_std):
    
    img_out = (img - img.mean(axis = (0,1)))/img.std(axis = (0,1))
    img_out = img_out*ref_std + ref_mean
    
    return img_out


# percentiles : [[min, 10%, ..., 90%, max] for each channel]
def get_percentiles(img, step_size = 10):
    
    img_percentiles = [np.min(img, axis = (0, 1))]
    for percent in range(step_size, 100, step_size):
        img_percentiles.append(np.percentile(img, percent, axis = (0, 1)))
    img_percentiles.append(np.max(img, axis = (0, 1)))
    
    # theres prob a better numpy way here but no time loss
    img_percentiles = np.array(img_percentiles)
    img_percentiles = [img_percentiles[:, k] for k in range(3)]
    
    return np.array(img_percentiles)


# 
def show_distribs(img, title = ''):
    
    plt.figure(figsize = (10, 5))
    plt.plot(np.sort(img[:, :, 0].flatten()), label = 'L')
    plt.plot(np.sort(img[:, :, 1].flatten()), label = 'a')
    plt.plot(np.sort(img[:, :, 2].flatten()), label = 'b')
    plt.legend(loc = 'best')
    if title:
        plt.title(title)
    plt.show()
    
    return


# 
# all img in Lab
# disgusting computation in method v1, could pre compute a bunch of stuff
# > use v2
def match_percentiles(img, ref_img, plot_distribs = False, 
                      method = 'v2', step_size = 2):
    
    img_percentiles = get_percentiles(img, step_size=step_size)
    ref_percentiles = get_percentiles(ref_img, step_size=step_size)
    
    if method == 'v2':
        channel_L = []
        for k in range(3):
            channel = np.copy(img[:, :, k]) #in case it doesnt already
            channel_out = np.copy(channel)
            for i_prct in range(len(img_percentiles[k]) - 1):
                percent_min = img_percentiles[k][i_prct]
                percent_max = img_percentiles[k][i_prct + 1]
                
                # index mask
                mask_min = percent_min < channel
                mask_max = channel <= percent_max
                mask = mask_min & mask_max
                
                # - min/(max - min) and inverse
                channel_out[mask] = channel_out[mask] - img_percentiles[k][i_prct]
                channel_out[mask] = channel_out[mask]/(img_percentiles[k][i_prct+1] - img_percentiles[k][i_prct])
                channel_out[mask] = channel_out[mask]*(ref_percentiles[k][i_prct+1] - ref_percentiles[k][i_prct])
                channel_out[mask] = channel_out[mask] + ref_percentiles[k][i_prct]
            # 
            channel_L.append(channel_out)
        # 
        img_out = np.dstack(channel_L)
    # ---------
    elif method == 'v1': 
        img_out = np.zeros_like(img)
        for i in range(len(img)):
            for j in range(len(img[i])):
                for k in range(3):
                    val = img[i][j][k]
                    index = np.searchsorted(img_percentiles[k], val)
                    
                    val = val - img_percentiles[k][index-1]
                    val = val/(img_percentiles[k][index] - img_percentiles[k][index-1])
                    val = val*(ref_percentiles[k][index] - ref_percentiles[k][index-1])
                    val = val + ref_percentiles[k][index-1]
                    
                    img_out[i][j][k] = val
                # for rgb
        # for pixel
    
    if plot_distribs:
        show_distribs(img, 'original img Lab')
        show_distribs(ref_img, 'reference img Lab')
        show_distribs(img_out, 'transformed img Lab')
    # 
    
    return img_out




#%% main

# 
def main_run_2img(rgb_img, rgb_ref):
    
    test_lab = convert_RGB_to_Lab(rgb_img)
    ref_lab = convert_RGB_to_Lab(rgb_ref)
    
    ref_mean, ref_std = ref_lab.mean(axis = (0, 1)), ref_lab.std(axis = (0, 1))
    test_tf_std = standardise_tf(test_lab, ref_mean, ref_std)
    test_tf = convert_Lab_to_RGB(test_tf_std)
    
    show_distribs(test_lab, 'original img Lab')
    show_distribs(ref_lab, 'reference img Lab')
    show_distribs(test_tf_std, 'transformed img Lab')

    show_img(rgb_img, title = 'test')
    show_img(rgb_ref, title = 'ref')
    show_img(test_tf, title = 'test tf lab mean std')

    aaa = match_percentiles(test_lab, ref_lab, method = 'v2', 
                            plot_distribs = True)
    aaa = convert_Lab_to_RGB(aaa)

    show_img(aaa, title = 'test tf lab 10% percentiles')
    
    aaa = match_percentiles(test_lab, ref_lab, method = 'v2', 
                            step_size = 1, plot_distribs = True)
    aaa = convert_Lab_to_RGB(aaa)

    show_img(aaa, title = 'test tf lab 1% percentiles')
    
    return test_tf


# 
def main_swap_2img(rgb_img, rgb_ref, step_size = 10):
    
    test_lab = convert_RGB_to_Lab(rgb_img)
    ref_lab = convert_RGB_to_Lab(rgb_ref)
    
    show_img(rgb_img)
    show_img(rgb_ref)
    
    # 
    transformed = match_percentiles(test_lab, ref_lab, step_size=step_size)
    transformed = convert_Lab_to_RGB(transformed)

    show_img(transformed)
    
    # 
    transformed = match_percentiles(ref_lab, test_lab, step_size=step_size)
    transformed = convert_Lab_to_RGB(transformed)

    show_img(transformed)
    
    return


#%% sandbox

base_path = "E:/Python_Data/general_img_db/anime_comics_fantasy_game/"
img_list = os.listdir(Path(base_path))
img_list = [path for path in img_list if os.path.isfile(Path(base_path + path))]

random_imgs = np.random.choice(img_list, size = 2, replace = False)
print(random_imgs)
test_img = Path(base_path + random_imgs[1])
ref_img = Path(base_path + random_imgs[0])

# test_img = Path("E:/Python_Data/general_img_db/test_bear.jpg")
# ref_img = Path("E:/Python_Data/general_img_db/mona_lisa.jpg")


test_img = colour.read_image(test_img)[:, :, :3]
ref_img = colour.read_image(ref_img)[:, :, :3]
print(test_img.shape, ref_img.shape)




#%%

# test_tf = main_run_2img(test_img, ref_img)
import time

start = time.time()
# aa = main_swap_2img(test_img, ref_img, step_size = 5)
# aa = main_swap_2img(test_img, ref_img, step_size = 2)
aa = main_swap_2img(test_img, ref_img, step_size = 1)
print(time.time() - start)



#%%


#%%

# test_lab = convert_RGB_to_Lab(test_img)
# ref_lab = convert_RGB_to_Lab(ref_img)


# #%%

# ref_mean, ref_std = ref_lab.mean(axis = (0, 1)), ref_lab.std(axis = (0, 1))
# test_tf_std = standardise_tf(test_lab, ref_mean, ref_std)
# test_tf = convert_Lab_to_RGB(test_tf_std)

# show_img(test_img, title = 'test')
# show_img(ref_img, title = 'ref')
# show_img(test_tf, title = 'test tf lab mean std')

# test_lab_tf = match_percentiles(test_lab, ref_lab, method = 'v2', 
#                                 plot_distribs = True, step_size = 50)
# test_tf = convert_Lab_to_RGB(test_lab_tf)

# show_img(test_tf, title = 'test tf lab 10% percentiles')

# test_lab_tf = match_percentiles(test_lab, ref_lab, 
#                                 plot_distribs = True, step_size = 50)
# test_tf = convert_Lab_to_RGB(test_lab_tf)

# show_img(test_tf, title = 'test tf lab 10% percentiles')


# #%% 


# print(f"test img mean std {test_img.mean(axis = (0, 1))} {test_img.std(axis = (0, 1))}")
# print(f"ref img mean std {ref_img.mean(axis = (0, 1))} {ref_img.std(axis = (0, 1))}")
# print(f"test tf img mean std {test_tf.mean(axis = (0, 1))} {test_tf.std(axis = (0, 1))}")
























































































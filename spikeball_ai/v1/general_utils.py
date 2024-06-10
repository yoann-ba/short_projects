

#%% import

import numpy as np
import matplotlib.pyplot as plt

import colour

#%% global data/info


# in floats, *255 for uint8
colors_to_labels = {
    'isBall': [1, 1, 1], 
    'isGoalPost': [1, 0, 0], 
    'other': [0, 0, 0]
    }

# # in floats, *255 for uint8
# colors_to_labels = {
#     'isBall': [1, 1, 1], 
#     'isGrass': [0, 0, 0], 
#     'isGoalPost': [1, 0, 0], 
#     'isSky': [0, 0, 1], 
#     'isSkin': [0, 1, 0], 
#     'other': [1, 0, 1]
#     }


#%% util functions

# 
def show_img(img, size_scaler = 1, title = '', cmap = '', 
             vmin = None, vmax = None):
    
    screen_dpi = 96
    temp_size = (int(img.shape[0]*size_scaler/plt.rcParams['figure.dpi']), 
                 int(img.shape[1]*size_scaler/plt.rcParams['figure.dpi']))
    # for some reason the dpi is neither the screen dpi or plt params dpi?
    # 55 isnt either but its a close approximation
    temp_size = (int(img.shape[0]*size_scaler/55), 
                 int(img.shape[1]*size_scaler/55))
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
    
    # alternatively
    return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))
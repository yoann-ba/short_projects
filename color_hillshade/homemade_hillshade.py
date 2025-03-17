



#%% imports


import numpy as np
import matplotlib.pyplot as plt

import cv2

import os
from pathlib import Path


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
    # return colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(rgb_img))
    return cv2.cvtColor(rgb_img.astype(np.float32), cv2.COLOR_RGB2LAB)

# 
def convert_Lab_to_RGB(lab_img):
    
    # alternatively
    # return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))
    return cv2.cvtColor(lab_img.astype(np.float32), cv2.COLOR_LAB2RGB)




#%% sandbox


base_path = "E:/Python_Data/general_img_db/"

img = Path(base_path + "alex_dog.jpg")

img = cv2.imread(img)
img = cv2.resize(img, (800, 800))

img = img.astype(np.float32)/255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lab = convert_RGB_to_Lab(img)


#%% 


inclination_angle = 40
direction_angle = 0 #TODO

inclination_angle = np.deg2rad(inclination_angle)
direction_angle = np.deg2rad(direction_angle)

incl = np.tan(inclination_angle)
inverse_inclination = 1./incl
minimum_height = incl


# 0 for max shadow, 1 for max light
shadow_map = np.zeros_like(lab[:, :, 0])
value_map = lab[:, :, 0]
value_map = value_map.astype(np.float32)*(1/1)
shadow_height = np.zeros_like(shadow_map)

show_img(value_map, title = "value map", cmap = "gray")
print(f"the max height of {np.max(value_map):.1f} will have a shadow of ",end = "")
print(f"{np.max(value_map)*inverse_inclination:.1f} pixels")



#%% 

for i in range(len(shadow_map)-1, -1, -1):
    if i%100 == 0:
        print(i)
    for j in range(0, len(shadow_map[i]), 1):
        value = value_map[i][j]
        if value >= shadow_height[i][j]:
            shadow_map[i][j] = 1
            if value > minimum_height:
                shadow_length = value * inverse_inclination
                for c_i in range(1, round(shadow_length)+1):
                    if i - c_i<1:
                        continue
                    shadow_height[i - c_i][j] = (shadow_length - c_i)*incl
        else:
            continue
    #end for j
    # if i%100 == 0:
    #     show_img(shadow_map, title = f"shadow map {i}", cmap = "gray")
    #     show_img(shadow_height, title = f"shadow height {i}", cmap = "gray")
#end for i
show_img(shadow_map, title = f"shadow map {i}", cmap = "gray")
show_img(shadow_height, title = f"shadow height {i}", cmap = "gray")
# show_img(shadow_height > value_map, title = "difference", cmap = "gray")



#%% 

show_img(img, title = "img")

new_L = shadow_map.copy()
new_L[shadow_map > 0] = value_map[shadow_map > 0]*1.1
new_L[shadow_map == 0] = value_map[shadow_map == 0]*0.25

new_lab = np.dstack((new_L, lab[:, :, 1], lab[:, :, 2]))
new_rgb = convert_Lab_to_RGB(new_lab)
show_img(new_rgb, title = "shadow_map as L, with value map as 1s")



new_L = shadow_height.copy() - np.min(shadow_height)
new_L = new_L/(np.max(shadow_height) - np.min(shadow_height))
new_L = np.max(new_L) - new_L # invert it for a coef, 0 to 1

show_img(new_L, title = "inverse shadow h coef", cmap = "gray")
new_L = np.multiply(value_map, new_L)

new_L[shadow_map > 0] = value_map[shadow_map > 0]*1.1

new_lab = np.dstack((new_L, lab[:, :, 1], lab[:, :, 2]))
new_rgb = convert_Lab_to_RGB(new_lab)
show_img(new_rgb, title = "shadow_height as coef for value map")



# new_L = shadow_height.copy() - np.min(shadow_height)
# new_L = new_L/(np.max(shadow_height) - np.min(shadow_height))
# new_L = np.max(new_L) - new_L # invert it for a coef, 0 to 1

# new_lab = np.dstack((new_L*97, lab[:, :, 1], lab[:, :, 2]))
# new_rgb = convert_Lab_to_RGB(new_lab)
# show_img(new_rgb, title = "inverse shadow h as L")





#%%

# L_min = np.min(lab[:, :, 0])
# L_max = np.max(lab[:, :, 0])
# new_L = shadow_map*(L_max - L_min) + L_min

# new_lab = np.dstack((new_L, lab[:, :, 1], lab[:, :, 2]))

# new_rgb = convert_Lab_to_RGB(new_lab)
# show_img(new_rgb, title = "shadow_map as L, stretched to L min max")



# new_L = shadow_height - np.min(shadow_height)
# new_L = new_L/np.max(shadow_height)
# new_L = new_L*(L_max - L_min) + L_min

# new_lab = np.dstack((new_L, lab[:, :, 1], lab[:, :, 2]))

# new_rgb = convert_Lab_to_RGB(new_lab)
# show_img(new_rgb, title = "shadow_height as L, stretched to L min max")


# #%% 

# new_L = shadow_map * (70 - 20) + 20

# new_lab = np.dstack((new_L, lab[:, :, 1], lab[:, :, 2]))
# new_rgb = convert_Lab_to_RGB(new_lab)
# show_img(new_rgb, title = "shadow_map as L, stretched to [40, 70]")


# new_L = shadow_height - np.min(shadow_height)
# new_L = new_L/np.max(shadow_height)
# new_L = new_L * (70 - 20) + 20

# new_lab = np.dstack((new_L, lab[:, :, 1], lab[:, :, 2]))
# new_rgb = convert_Lab_to_RGB(new_lab)
# show_img(new_rgb, title = "shadow_height as L, stretched to [40, 70]")




#%% 



#%%

# Convolution version -> rip due to varying shadow length? 


# target = img[:, :, 0] > 0.2
# target = target.astype(np.float32)

# kernel_size = 2
# kernel = np.empty((2*kernel_size+1, 2*kernel_size+1))
# for i in range(len(kernel)):
#     for j in range(len(kernel[i])):
#         kernel[i][j] = (kernel_size - i)**2 + (kernel_size - j)**2
# # 
# kernel = np.sqrt(kernel)
# kernel = kernel/np.sum(kernel)
# kernel = kernel.max() - kernel

# test = cv2.filter2D(target, ddepth = -1, kernel = kernel)

# show_img(target, cmap = "Reds_r")
# show_img(test, cmap = "Reds_r")













































































#%% imports

import numpy as np
import matplotlib.pyplot as plt

import colour

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
    
    # alternatively
    return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))



#%% funcs

# Neon Science implementation of ESRI ArcGIS
# https://www.neonscience.org/resources/learning-hub/tutorials/create-hillshade-py
def hillshade(array, azimuth = 315, angle_altitude = 45):
    azimuth = 360.0 - azimuth 
    
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm_rad = azimuth*np.pi/180. #azimuth in radians
    alt_rad = angle_altitude*np.pi/180. #altitude in radians
 
    shaded = np.sin(alt_rad)*np.sin(slope) + np.cos(alt_rad)*np.cos(slope)*np.cos((azm_rad - np.pi/2.) - aspect)
    
    return 255*(shaded + 1)/2


# 
def re_scale(img, new_min = 0, new_max = 1):
    
    new_img = (img - img.min())/(img.max() - img.min())
    new_img = new_img*(new_max - new_min) + new_min
    
    return new_img, img.min(), img.max()

# 
def hill_scaled(img, azimuth = 315, angle_altitude = 45):
    
    show_img(img)
    
    r_stats = (img[:, :, 0].min(), img[:, :, 0].max(), 
               img[:, :, 0].mean(), img[:, :, 0].std())
    g_stats = (img[:, :, 1].min(), img[:, :, 1].max(), 
               img[:, :, 1].mean(), img[:, :, 1].std())
    b_stats = (img[:, :, 2].min(), img[:, :, 2].max(), 
               img[:, :, 2].mean(), img[:, :, 2].std())
    
    print(r_stats, g_stats, b_stats)
    
    r_hill = hillshade(img[:, :, 0])
    r_hill = re_scale(r_hill)[0]
    
    g_hill = hillshade(img[:, :, 1])
    g_hill = re_scale(g_hill)[0]
    
    b_hill = hillshade(img[:, :, 2])
    b_hill = re_scale(b_hill)[0]
    
    show_img(r_hill, cmap = 'gray')
    show_img(g_hill, cmap = 'gray')
    show_img(b_hill, cmap = 'gray')
    
    rgb_hill = np.dstack((r_hill*r_stats[2], g_hill*g_stats[2], b_hill*b_stats[2]))
    rgb_hill = re_scale(rgb_hill)[0]
    show_img(rgb_hill)
    
    mult = hillshade(img[:, :, 0]*img[:, :, 1]*img[:, :, 2])
    show_img(mult, cmap = 'gray')
    
    return rgb_hill



#%% main




#%% sandbox

path = r"E:\Python_Data\general_img_db/"
# path += "raw_1009_vesuvius_0.0191_lanczos4.png"
path += "alex_dog.jpg"
path = Path(path)

img = colour.read_image(path)[:, :, :3]

# arcgis_hillshade = hillshade(img[:, :, 0])


#%% 

if img.shape[2] == 3:
    hill_scaled(img)
else:
    show_img(img)
    show_img(hillshade(img), cmap = 'gray')
# 














#%% 

    # if img.shape(2) == 1:
    #     new_img = (img - img.min())/(img.max() - img.min())
    #     new_img = new_img*(new_max - new_min) + new_min
        
    #     return new_img, img.min(), img.max()
    # elif img.shape(2) == 3:
    #     r, rm, rM = re_scale(img[:, :, 0], new_min=new_min, new_max=new_max)
    #     g, gm, gM = re_scale(img[:, :, 1], new_min=new_min, new_max=new_max)
    #     b, bm, bM = re_scale(img[:, :, 2], new_min=new_min, new_max=new_max)
        
    #     new_img = np.dstack((r, g, b))
    #     return new_img, (rm, gm, bm), (rM, gM, bM)
    # else:
    #     print('what')
    #     return










































































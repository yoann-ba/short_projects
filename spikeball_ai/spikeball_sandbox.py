




#%% imports

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import colour

from sklearn.ensemble import RandomForestRegressor


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


#%% main run



#%% sandbox

img_path = 'E:\Python_Data\spikeball\spikeball_quick/'
img_name = "frame_"+str(0)+".png"

img_path = Path(img_path + img_name)

rgb_img = colour.read_image(img_path)[:, :, :3]

lab_img = convert_RGB_to_Lab(rgb_img)


#%% load data and fit models

labels = ['isBall', 'isGrass', 'isGoalPost', 'isSky', 'isSkin']

models = {}

for label in labels:
    temp_path = "E:\Python_Data\spikeball\spikeball_quick\quick_numpy_saves/"
    data_x = np.load(Path(temp_path + f"{label}_rgb.npy"))
    data_y = np.load(Path(temp_path + f"{label}_y.npy"))
    
    rfc = RandomForestRegressor()
    rfc.fit(data_x, data_y)
    models[label] = rfc
# 


#%%

for label in labels:
    prediction = rgb_img.reshape((rgb_img.shape[0]*rgb_img.shape[1], 3)).astype(np.float16)
    prediction = models[label].predict(prediction)
    prediction = prediction.reshape((rgb_img.shape[0], rgb_img.shape[1]))
    
    show_img(prediction, size_scaler = 0.5, title = f"prediction {label}", cmap = 'gray')
# 


#%%

for i_frame in range(8):
    img_path = 'E:\Python_Data\spikeball\spikeball_quick/'
    img_name = "frame_"+str(i_frame)+".png"

    img_path = Path(img_path + img_name)

    rgb_img = colour.read_image(img_path)[:, :, :3]

    prediction = rgb_img.reshape((rgb_img.shape[0]*rgb_img.shape[1], 3)).astype(np.float16)
    prediction = models['isBall'].predict(prediction)
    prediction = prediction.reshape((rgb_img.shape[0], rgb_img.shape[1]))
    
    show_img(prediction, size_scaler = 0.5, title = f"prediction Ball frame {i_frame}", cmap = 'gray')
# 






























































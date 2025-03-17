



#%% imports


import numpy as np
import matplotlib.pyplot as plt

import cv2

import os
from pathlib import Path


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
    # return colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(rgb_img))
    return cv2.cvtColor(rgb_img.astype(np.float32), cv2.COLOR_RGB2LAB)

# 
def convert_Lab_to_RGB(lab_img):
    
    # alternatively
    # return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))
    return cv2.cvtColor(lab_img.astype(np.float32), cv2.COLOR_LAB2RGB)




#%% sandbox


base_path = "E:/Python_Data/general_img_db/"

img = Path(base_path + "test_bear.jpg")

img = cv2.imread(img)

img = img.astype(np.float32)/255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lab = convert_RGB_to_Lab(img)


#%% 

lab_L = lab[:, :, 0]

n_clusters = 2**4
kmeans = KMeans(n_clusters = n_clusters, n_init = 5)
kmeans.fit(lab_L.flatten().reshape(-1, 1))

# for every cluster, Lab space coords
cluster_centers = kmeans.cluster_centers_
# for every pixel, cluster id
cluster_indexes = kmeans.predict(lab_L.flatten().reshape(-1, 1))
# for every pixel, cluster Lab coords
clustered_values = cluster_centers[cluster_indexes]

temp0 = clustered_values.reshape(lab_L.shape)



lab_L = lab[:, :, 1]

n_clusters = 2**2
kmeans = KMeans(n_clusters = n_clusters, n_init = 5)
kmeans.fit(lab_L.flatten().reshape(-1, 1))

# for every cluster, Lab space coords
cluster_centers = kmeans.cluster_centers_
# for every pixel, cluster id
cluster_indexes = kmeans.predict(lab_L.flatten().reshape(-1, 1))
# for every pixel, cluster Lab coords
clustered_values = cluster_centers[cluster_indexes]

temp = clustered_values.reshape(lab_L.shape)



lab_L = lab[:, :, 2]

n_clusters = 2**2
kmeans = KMeans(n_clusters = n_clusters, n_init = 5)
kmeans.fit(lab_L.flatten().reshape(-1, 1))

# for every cluster, Lab space coords
cluster_centers = kmeans.cluster_centers_
# for every pixel, cluster id
cluster_indexes = kmeans.predict(lab_L.flatten().reshape(-1, 1))
# for every pixel, cluster Lab coords
clustered_values = cluster_centers[cluster_indexes]

temp2 = clustered_values.reshape(lab_L.shape)



test = np.dstack((temp0, temp, temp2))
test = convert_Lab_to_RGB(test)

show_img(test, title = n_clusters)













































































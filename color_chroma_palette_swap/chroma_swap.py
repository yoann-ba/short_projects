


#%% imports

import numpy as np
import matplotlib.pyplot as plt

# import colour

import cv2

from pathlib import Path
import os


from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

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
    # return colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(rgb_img))
    return cv2.cvtColor(rgb_img.astype(np.float32), cv2.COLOR_RGB2LAB)

# 
def convert_Lab_to_RGB(lab_img):
    
    # alternatively
    # return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(lab_img))
    return cv2.cvtColor(lab_img.astype(np.float32), cv2.COLOR_LAB2RGB)


#%% ??


# 
def kmeans_clustering(lab_img, n_clusters = 10, small_size = 10000):
    
    color_list = (lab_img.shape[0]*lab_img.shape[1], lab_img.shape[2])
    color_list = np.copy(lab_img).reshape(color_list)
    
    rng = np.random.default_rng()
    mini_color_list = rng.choice(color_list, size = small_size, 
                                 replace = False, shuffle = False)
    # print(mini_color_list.shape)
    
    kmeans = KMeans(n_clusters = n_clusters, n_init = 5)
    kmeans.fit(mini_color_list)
    
    inertia = kmeans.inertia_
    # for every cluster, Lab space coords
    cluster_centers = kmeans.cluster_centers_
    # for every pixel, cluster id
    cluster_indexes = kmeans.predict(color_list)
    # for every pixel, cluster Lab coords
    clustered_values = cluster_centers[cluster_indexes]
    
    new_img = clustered_values.reshape(lab_img.shape)
    
    return new_img, inertia

# 
def parallel_kmeans(base_lab, ref_lab, n_clusters = 10, small_size = 10000):
    
    # Datasets
    base_colors = (base_lab.shape[0]*base_lab.shape[1], base_lab.shape[2])
    base_colors = np.copy(base_lab).reshape(base_colors)
    
    ref_colors = (ref_lab.shape[0]*ref_lab.shape[1], ref_lab.shape[2])
    ref_colors = np.copy(ref_lab).reshape(ref_colors)
    
    # Reduction
    rng = np.random.default_rng()
    mini_base = rng.choice(base_colors, size=small_size, replace=False, shuffle=False)
    mini_ref = rng.choice(ref_colors, size=small_size, replace=False, shuffle=False)
    
    # KMeans
    base_kmeans = KMeans(n_clusters=n_clusters, n_init = 5)
    base_kmeans.fit(mini_base)
    
    base_inertia = base_kmeans.inertia_
    base_cluster_coords = base_kmeans.cluster_centers_
    base_indexes = base_kmeans.predict(base_colors)
    
    ref_kmeans = KMeans(n_clusters=n_clusters, n_init = 5)
    ref_kmeans.fit(mini_ref)
    
    ref_inertia = ref_kmeans.inertia_
    ref_cluster_coords = ref_kmeans.cluster_centers_
    ref_indexes = ref_kmeans.predict(ref_colors)
    
    return


#%% percentile matching


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


# percentiles : [[min, 10%, ..., 90%, max] for each channel]
# WITH ALPHA MASK
# FLATTENING EVERYTHING so that the np percentile accepts the alpha mask
# idk why it doesnt work in (row, col, pixel) format with axis (0, 1) but w/e
def get_percentiles_alpha(img, alpha_mask, step_size = 10):
    
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    flat_img = img.reshape(new_shape)
    flat_alpha = alpha_mask.flatten()
    
    where_mask = flat_alpha > 0.9
    where_mask = np.dstack((where_mask, where_mask, where_mask))[0]
    
    img_percentiles = [np.min(flat_img, axis = 0, 
                              where = where_mask, initial = 1)]
    for percent in range(step_size, 100, step_size):
        img_percentiles.append(np.percentile(flat_img, percent, axis = 0, 
                                             method = "inverted_cdf", 
                                             weights = flat_alpha))
    img_percentiles.append(np.max(flat_img, axis = 0, 
                                  where = where_mask, initial = 0))
    
    # theres prob a better numpy way here but no time loss
    img_percentiles = np.array(img_percentiles)
    img_percentiles = [img_percentiles[:, k] for k in range(3)]
    
    return np.array(img_percentiles)

# 
# all img in Lab
# THE DISTRIUTION PLOT DOESNT USE THE ALPHA MASK
# YOU GET THE WHOLE THING EVERY TIME
def match_percentiles(img, ref_img, 
                      img_alpha, ref_alpha, 
                      plot_distribs = False, 
                      method = 'v2', step_size = 2):
    
    img_percentiles = get_percentiles_alpha(img, img_alpha, 
                                            step_size=step_size)
    ref_percentiles = get_percentiles_alpha(ref_img, ref_alpha, 
                                            step_size=step_size)
    
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
    
    if plot_distribs:
        show_distribs(img, 'original img Lab')
        show_distribs(ref_img, 'reference img Lab')
        show_distribs(img_out, 'transformed img Lab')
    # 
    
    return img_out



#%% funcs


# 
#TODO splits, hyperopt
def train(base_lab, chroma_lab, alpha_mask):
    
    print("Train prep")
    # Scalers
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # Model
    rfr = RandomForestRegressor(n_estimators = 200, n_jobs = 3, 
                                max_depth = 10, min_samples_leaf = 1)
    
    # Reshape data to list of pixels, then minmax all to 0-1
    new_shape = (base_lab.shape[0]*base_lab.shape[1], base_lab.shape[2])
    x_data = base_lab.reshape(new_shape)
    y_data = chroma_lab.reshape(new_shape)
    alpha = alpha_mask.flatten()
    
    x = x_scaler.fit_transform(x_data).astype(np.float32)
    y = y_scaler.fit_transform(y_data).astype(np.float32)
    
    palette = np.random.random((50000, 3)).astype(np.float32)
    x = np.concatenate((x, palette))
    y = np.concatenate((y, palette))
    alpha = np.concatenate((alpha, np.ones(palette.shape[0])))
    
    # Fit
    print("Fitting model")
    rfr.fit(x, y, sample_weight = alpha)
    
    return x_scaler, y_scaler, rfr


# lab in, lab out
def apply(lab_img, x_scaler, y_scaler, rfr):
    
    # Reshape
    new_shape = (lab_img.shape[0]*lab_img.shape[1], lab_img.shape[2])
    test = lab_img.reshape(new_shape)

    # scale, transform, descale
    test = x_scaler.transform(test)
    test = rfr.predict(test)
    #test = np.clip(test, 0, 1)
    test = y_scaler.inverse_transform(test)

    # Reshape and make RGB
    test = test.reshape(lab_img.shape)
    #test = convert_Lab_to_RGB(test)
    
    return test


# 
# WITH ALPHA MASK (continuous, transformed into bool)
def Lab_rescale(base_lab, ref_lab, alpha_mask):
    
    # Awkward 3x copy of the bool mask, to match constraints
    where_mask = alpha_mask > 0.2
    where_mask = np.dstack((where_mask, where_mask, where_mask))
    
    ref_mean = ref_lab.mean(axis = (0, 1), where = where_mask)
    ref_std = ref_lab.std(axis = (0, 1), where = where_mask)
    
    new_lab = (base_lab - base_lab.mean(axis = (0, 1)))/base_lab.std(axis = (0, 1))
    new_lab *= ref_std
    new_lab += ref_mean
    
    return new_lab


# 
def test_printer():
    
    test = Path(base_path + "Print_test_target.png")
    test = cv2.imread(test)[:, :, :3]
    test.shape
    test = test.astype(np.float32)/255
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    test_lab = convert_RGB_to_Lab(test)
    test_out = apply(test_lab, x_scaler, y_scaler, rfr)
    test_out = convert_Lab_to_RGB(test_out)
    show_img(test, title = "printer test image")
    show_img(test_out, title = "transfo")
    
    return




#%% sandbox

base_path = "E:/Python_Data/general_img_db/"

splash_img = Path(base_path + "Leona_MechaKingdomsSkin.jpg")
splash_alpha = Path(base_path + "Leona_MechaKingdomsSkin_mask.png")
base_img = Path(base_path + "Leona_MechaKingdoms_29.png")
chroma_img = Path(base_path + "Leona_MechaKingdoms_29_Obsidian.png")

base_img = cv2.imread(base_img, cv2.IMREAD_UNCHANGED)
base_img, alpha_mask = base_img[:, :, :3], base_img[:, :, 3]
chroma_img = cv2.imread(chroma_img)[:, :, :3]
splash_img = cv2.imread(splash_img)[:, :, :3]
splash_alpha = cv2.imread(splash_alpha)[:, :, 0]

base_img = base_img.astype(np.float32)/255
alpha_mask = alpha_mask.astype(np.float32)/255
chroma_img = chroma_img.astype(np.float32)/255
splash_img = splash_img.astype(np.float32)/255
splash_alpha = splash_alpha.astype(np.float32)/255

base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
chroma_img = cv2.cvtColor(chroma_img, cv2.COLOR_BGR2RGB)
splash_img = cv2.cvtColor(splash_img, cv2.COLOR_BGR2RGB)

print(base_img.shape, chroma_img.shape)

base_lab = convert_RGB_to_Lab(base_img)
chroma_lab = convert_RGB_to_Lab(chroma_img)
splash_lab = convert_RGB_to_Lab(splash_img)


#%%

x_scaler, y_scaler, rfr = train(base_lab, chroma_lab, alpha_mask)


#%%

test = apply(base_lab, x_scaler, y_scaler, rfr)
test = convert_Lab_to_RGB(test)

show_img(base_img, title = "original", size_scaler = 2)
show_img(chroma_img, title = "chroma", size_scaler = 2)
show_img(test, title = "transfo", size_scaler = 2)


#%% 

test_splash = apply(splash_lab, x_scaler, y_scaler, rfr)
test_splash = convert_Lab_to_RGB(test_splash)

show_img(splash_img)
show_img(test_splash)


scaled_lab = Lab_rescale(splash_lab, base_lab, alpha_mask)
show_img(convert_Lab_to_RGB(scaled_lab), title = "re_scaled mean std")

test_splash = apply(scaled_lab, x_scaler, y_scaler, rfr)
test_splash = convert_Lab_to_RGB(test_splash)
show_img(test_splash, title = "applied transfo")


scaled_lab = match_percentiles(splash_lab, base_lab, 
                               splash_alpha, alpha_mask)
show_img(convert_Lab_to_RGB(scaled_lab), title = "re_scaled percentiles 2%")

test_splash = apply(scaled_lab, x_scaler, y_scaler, rfr)
show_img(convert_Lab_to_RGB(test_splash), title = "applied transfo")

# actually this doesnt make sense at all
# test_splash = match_percentiles(test_splash, splash_lab, 
#                                 splash_alpha, splash_alpha)
# test_splash = convert_Lab_to_RGB(test_splash)
# show_img(test_splash, title = "applied transfo")








#%%


# show_img(base_img, title="original", size_scaler = 3)
# for i in range(2, 100, 10):
#     new_img, inertia = kmeans_clustering(base_lab, n_clusters = i)
#     new_img = convert_Lab_to_RGB(new_img)
#     show_img(new_img, title=f"{i} clusters | {inertia:.2f}", size_scaler = 3)
# # 



































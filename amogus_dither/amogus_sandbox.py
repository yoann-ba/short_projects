







#%% imports

import numpy as np
import matplotlib.pyplot as plt

from time import time

from sortedcontainers import SortedList

import cv2

from pathlib import Path

from sklearn.cluster import KMeans




#%% globals

screen_dpi = 96

# amogus indices
# in a (5, 4) matrix
# body indices = rest
bg_ind = np.array([[0, 0], 
                   [3, 0], 
                   [4, 0], 
                   [4, 2]])
visor_ind = np.array([[1, 2], 
                      [1, 3]])
body_ind = np.array([[0, 1], 
                     [0, 2], 
                     [0, 3], 
                     [1, 0], 
                     [1, 1], 
                     [2, 0], 
                     [2, 1], 
                     [2, 2], 
                     [2, 3], 
                     [3, 1], 
                     [3, 2], 
                     [3, 3], 
                     [4, 1], 
                     [4, 3]])

# how many pixels for the body and the visor
weight_body_visor = [14, 2]


#%% funcs

# 
# no frame, no white border
# should be exact size as input, with one pixel per pixel preserved
# > at least without titles, w/ title might be different shape
def show_img(img, title = '', size_scaler = 1):
    
    size = (size_scaler*img.shape[0]/screen_dpi, size_scaler*img.shape[1]/screen_dpi)
    fig = plt.figure(figsize = size, dpi = screen_dpi, frameon = True)
    
    plt.imshow(img)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
    return

# 
def convert_to_Lab(rgb_img):
    
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    
    return lab_img
# 
def convert_to_RGB(lab_img):
    
    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    
    return rgb_img


#%% core functions

# gets point, can use memory or not, adds square to memory
def get_random_point(target_area, hit_memory = []):
    
    ind = [np.random.randint(0, target_area[0]), np.random.randint(0, target_area[1])]
    if ind in hit_memory:
        return -1
    index_list = [[ind[0] +k, ind[1] +j] for k in range(-4, 5) for j in range(-3, 4)]
    hit_memory += index_list
    return ind


# just grabs two random colors from the crop
# already surprisingly decent
def apply_amogus_baseline(img_crop):
    
    # new_img = np.full_like(img_crop, [255, 0, 0])
    new_img = np.copy(img_crop)
    
    body_color = np.random.choice(np.arange(0, 5*4))
    visor_color = np.random.choice(np.arange(0, 5*4))
    
    body_color = img_crop[body_color//4, body_color%4]
    visor_color = img_crop[visor_color//4, visor_color%4]
    
    new_img[tuple(visor_ind.T)] = visor_color
    new_img[tuple(body_ind.T)] = body_color
    
    return new_img, 1


# add a proba depending on color std (in Lab)
def apply_amogus_Lab_proba(img_crop):
    
    new_img = np.copy(img_crop)
    crop_lab = convert_to_Lab(new_img)
    
    color_std = np.std(crop_lab, axis = (0, 1)) # 28 is good (80% prob?)
    # avg_std = np.average(color_std)
    L_std = color_std[0] # 70 = 50%, <20 = 90%?
    
    amogus_proba = (50-90)/(70-20)*L_std + (90 - 20*(50-90)/(70-20))
    amogus_proba = np.clip(amogus_proba, 0, 100)/100
    
    if np.random.random() >= amogus_proba:
        return new_img, 0
    body_color = np.random.choice(np.arange(0, 5*4))
    visor_color = np.random.choice(np.arange(0, 5*4))
    
    body_color = img_crop[body_color//4, body_color%4]
    visor_color = img_crop[visor_color//4, visor_color%4]
    
    new_img[tuple(visor_ind.T)] = visor_color
    new_img[tuple(body_ind.T)] = body_color
    
    return new_img, 1


# tries to use the [14, 2] pixel count weight
# but just ends up all grey
def apply_amogus_proba_weight(img_crop):
    
    new_img = np.copy(img_crop)
    crop_lab = convert_to_Lab(new_img)
    
    color_std = np.std(crop_lab, axis = (0, 1)) # 28 is good (80% prob?)
    # avg_std = np.average(color_std)
    L_std = color_std[0] # 70 = 50%, <20 = 90%?
    
    amogus_proba = (50-90)/(70-20)*L_std + (90 - 20*(50-90)/(70-20))
    amogus_proba = np.clip(amogus_proba, 0, 100)/100
    
    if np.random.random() >= amogus_proba:
        return new_img, 0
    
    body_color = np.random.choice(np.arange(0, 5*4), size = weight_body_visor[0])
    visor_color = np.random.choice(np.arange(0, 5*4), size = weight_body_visor[1])
    
    body_color = [img_crop[elem//4, elem%4] for elem in body_color]
    visor_color = [img_crop[elem//4, elem%4] for elem in visor_color]
    
    body_color = np.average(body_color, axis = (0, 1))
    visor_color = np.average(visor_color, axis = (0, 1))
    
    new_img[tuple(visor_ind.T)] = visor_color
    new_img[tuple(body_ind.T)] = body_color
    
    return new_img, 1


# use a KMeans to make 2 clusters
# use its inertia for the transform proba
# use its color counts to give closest color to each cluster
# > to body and visor
def apply_amogus_prob_kmeans(img_crop):
    
    new_crop = np.copy(img_crop)
    debug_crop = np.copy(img_crop)
    crop_lab = convert_to_Lab(new_crop)
    
    color_std = np.std(crop_lab, axis = (0, 1))
    L_std = color_std[0]
    # tune for visibility : 
    # - strict L_std < 5, random.choice([-2, -1, 1, 2])
    # - visible pattern L_std < high, random.choice(high vals)
    if L_std < 5:
        val_list = [-5, -4, -3, 3, 4, 5] # for high visibility
        val_list = [-2, -1, 1, 2] # for sneaky
        noise_vals = np.random.choice(val_list, (5, 4, 3))
        color_list = crop_lab.astype(int) + noise_vals
        color_list = np.clip(color_list, 0, 255)
        color_list = color_list.astype(np.uint8)
        color_list = color_list.reshape((5*4, 3))
    else:
        color_list = crop_lab.reshape((5*4, 3))
    
    # fit a KMeans of 2 clusters on the Lab colors of the crop
    kmeans = KMeans(n_clusters = 2, n_init = 'auto')
    kmeans.fit(color_list)
    
    # use kmeans inertia to get how well the crop fits to 2 clusters
    # 0 inertia = 100%, 20k = 50%?
    amogus_proba = (50 - 100)/20000*kmeans.inertia_ + 100
    amogus_proba = np.clip(amogus_proba, 0, 100)/100
    
    if np.random.random() >= amogus_proba:
        return new_crop, 0, debug_crop
    
    # get the Lab colors and distance to its cluster, per cluster
    list_c0, diff_c0 = [], []
    list_c1, diff_c1 = [], []
    for i in range(len(kmeans.labels_)):
        l = kmeans.labels_[i]
        if l == 1:
            list_c1.append(color_list[i])
            diff_c1.append(np.average((color_list[i] - kmeans.cluster_centers_[1])**2))
        if l == 0:
            list_c0.append(color_list[i])
            diff_c0.append(np.average((color_list[i] - kmeans.cluster_centers_[0])**2))
    # 
    
    # get the color closest to each cluster for selection
    # then RGB it
    if len(list_c0) == 0:
        return new_crop, 0, debug_crop
    if len(list_c1) == 0:
        return new_crop, 0, debug_crop
    closest_c0 = list_c0[np.argmin(diff_c0)]
    closest_c1 = list_c1[np.argmin(diff_c1)]
    
    closest_c0 = convert_to_RGB(np.array([[closest_c0]], dtype = np.uint8))[0][0]
    closest_c1 = convert_to_RGB(np.array([[closest_c1]], dtype = np.uint8))[0][0]
    closest_c0 = list(closest_c0)
    closest_c1 = list(closest_c1)
    
    # give the cluster with more colors to the body
    if len(list_c0) >= len(list_c1):
        body_color = closest_c0
        visor_color = closest_c1
    else:
        body_color = closest_c1
        visor_color = closest_c0
    
    # apply
    new_crop[tuple(visor_ind.T)] = visor_color
    new_crop[tuple(body_ind.T)] = body_color
    
    debug_crop[tuple(visor_ind.T)] = [255, 255, 255]
    debug_crop[tuple(body_ind.T)] = [255, 0, 0]
    
    return new_crop, 1, debug_crop


#%% macro func


# 
def main_amogus(img_crop, method = "kmeans"):
    
    if method == "kmeans":
        new_crop, nb, debug_crop = apply_amogus_prob_kmeans(img_crop)
    elif method == "baseline":
        new_crop, nb, debug_crop = apply_amogus_baseline(img_crop)
    elif method == "lab_proba":
        new_crop, nb, debug_crop = apply_amogus_Lab_proba(img_crop)
    # 
    
    return new_crop, nb, debug_crop


# 
def main_run(img, method = "kmeans", nb_iter = 5000, size_scaler = 1):
    
    print("image shape", img.shape)
    target_area = (img.shape[0] - 5, img.shape[1] - 4)
    print("target area", target_area)
    hit_memory = SortedList()

    new_img = np.copy(img)
    debug_img = np.copy(img)
    show_img(new_img, title = "original img", size_scaler = size_scaler)

    nb_amogus = 0
    for i in range(nb_iter):
        
        ind = get_random_point(target_area, hit_memory)
        if ind == -1:
            # print("index in hit memory, skip")
            continue
        img_crop = np.copy(new_img[ind[0]:ind[0]+5, ind[1]:ind[1]+4])
        
        new_crop, temp_nb, debug_crop = main_amogus(img_crop, method=method)
        nb_amogus += temp_nb
        
        new_img[ind[0]:ind[0]+5, ind[1]:ind[1]+4] = new_crop
        debug_img[ind[0]:ind[0]+5, ind[1]:ind[1]+4] = debug_crop
        
        # if i%10 == 0:
        #     show_img(new_img, size_scaler = 5)
    # 
    show_img(new_img, title = f"{nb_amogus} amoguses", size_scaler = size_scaler)   
    print("nb amogus made:", nb_amogus)
    show_img(debug_img, title = f"{nb_amogus} amoguses (debug)", size_scaler = size_scaler)
    
    return new_img, nb_amogus, debug_img


# 
def run_as_patches(img, method="kmeans", nb_iter=10000, size_scaler=2):
    
    half_row = img.shape[0]//2 + 1
    half_col = img.shape[1]//2 + 1
    
    TL = np.copy(img[0:half_row, 0:half_col])
    TR = np.copy(img[0:half_row, half_col:])
    BL = np.copy(img[half_row:, 0:half_col])
    BR = np.copy(img[half_row:, half_col:])
    
    total_nb = 0
    TL, temp_nb, tl_d = main_run(TL, method, nb_iter, size_scaler*2)
    total_nb += temp_nb
    TR, temp_nb, tr_d = main_run(TR, method, nb_iter, size_scaler*2)
    total_nb += temp_nb
    BL, temp_nb, bl_d = main_run(BL, method, nb_iter, size_scaler*2)
    total_nb += temp_nb
    BR, temp_nb, br_d = main_run(BR, method, nb_iter, size_scaler*2)
    total_nb += temp_nb
    
    # for some reason np.block doesnt like it?
    # new_img = np.block([
    #     [TL, TR], 
    #     [BL, BR]])
    
    # but this works even tho it should be the same?
    new_img = np.vstack((np.hstack((TL, TR)), np.hstack((BL, BR))))
    debug_img = np.vstack((np.hstack((tl_d, tr_d)), np.hstack((bl_d, br_d))))
    
    show_img(img, "original img", size_scaler)
    show_img(new_img, f"{total_nb} amoguses", size_scaler)
    show_img(debug_img, f"{total_nb} amoguses (debug)", size_scaler)
    
    return new_img, total_nb, debug_img


#%% sandbox

# IDEAS / TODO

# select random points in image (minus amogus size)
# > do stuff in there
# > remove square from possible access

# "do stuff": 
# > get the colors in area
# > get the color variance (better would be a 2-centroid closeness?)
# > if more color variance, less proba of pasting an amogus?
# > get the body and visor color of the amogus from the colors?
# > or from the Lab space that is filled by the colors of the area?


path = "E:\Python_Data\general_img_db/" + "mona_lisa_small.jpg"
#path = "E:\Python_Data/general_img_db/anime_comics_fantasy_game/" + "kingpseak_-_daniel_lieske.jpeg"
path = Path(path)

img = cv2.imread(str(path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img = cv2.resize(img, (512, 600))

# img = img[600:750, 950:1100] # alex dog zoom face


#%%

# method in ["baseline", "lab_proba", "kmeans"]

time_st = time()
# test = main_run(img, method = "kmeans", nb_iter = 10000, size_scaler=1)
test, nb, dbg = run_as_patches(img, method = "kmeans", nb_iter = 8000, size_scaler = 3)
print(f"time taken: {time() - time_st:.2f}")








#%% LPIPS fun fact

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch import Tensor

lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

img1 = Tensor(np.copy(img).astype(np.float64)*(2/255) - 1)
img2 = Tensor(np.copy(test).astype(np.float64)*(2/255) - 1)

img1 = img1.reshape(1, 3, img.shape[0], img.shape[1])
img2 = img2.reshape(1, 3, img.shape[0], img.shape[1])
print("LPIPS", lpips(img1, img2))







#%% FLIP loss bullshit

import sys
import numpy as np
import torch
import torch.nn as nn

class LDRFLIPLoss(nn.Module):
	""" Class for computing LDR FLIP loss """

	def __init__(self):
		""" Init """
		super().__init__()
		self.qc = 0.7
		self.qf = 0.5
		self.pc = 0.4
		self.pt = 0.95
		self.eps = 1e-15

	def forward(self, test, reference, pixels_per_degree=(0.7 * 3840 / 0.7) * np.pi / 180):
		"""
		Computes the LDR-FLIP error map between two LDR images,
		assuming the images are observed at a certain number of
		pixels per degree of visual angle

		:param test: test tensor (with NxCxHxW layout with values in the range [0, 1] in the sRGB color space)
		:param reference: reference tensor (with NxCxHxW layout with values in the range [0, 1] in the sRGB color space)
		:param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer,
								  default corresponds to viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display
		:return: float containing the mean FLIP error (in the range [0,1]) between the LDR reference and test images in the batch
		"""
		# LDR-FLIP expects non-NaN values in [0,1] as input
		reference = torch.clamp(reference, 0, 1)
		test = torch.clamp(test, 0, 1)

		# Transform reference and test to opponent color space
		reference_opponent = color_space_transform(reference, 'linrgb2ycxcz')
		test_opponent = color_space_transform(test, 'linrgb2ycxcz')

		deltaE = compute_ldrflip(test_opponent, reference_opponent, pixels_per_degree, self.qc, self.qf, self.pc, self.pt, self.eps)

		return deltaE


def compute_ldrflip(test, reference, pixels_per_degree, qc, qf, pc, pt, eps):
	"""
	Computes the LDR-FLIP error map between two LDR images,
	assuming the images are observed at a certain number of
	pixels per degree of visual angle

	:param reference: reference tensor (with NxCxHxW layout with values in the YCxCz color space)
	:param test: test tensor (with NxCxHxW layout with values in the YCxCz color space)
	:param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer,
							  default corresponds to viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display
	:param qc: float describing the q_c exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
	:param qf: float describing the q_f exponent in the LDR-FLIP feature pipeline (see FLIP paper for details)
	:param pc: float describing the p_c exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
	:param pt: float describing the p_t exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
	:param eps: float containing a small value used to improve training stability
	:return: tensor containing the per-pixel FLIP errors (with Nx1xHxW layout and values in the range [0, 1]) between LDR reference and test images
	"""
	# --- Color pipeline ---
	# Spatial filtering
	s_a, radius_a = generate_spatial_filter(pixels_per_degree, 'A')
	s_rg, radius_rg = generate_spatial_filter(pixels_per_degree, 'RG')
	s_by, radius_by = generate_spatial_filter(pixels_per_degree, 'BY')
	radius = max(radius_a, radius_rg, radius_by)
	filtered_reference = spatial_filter(reference, s_a, s_rg, s_by, radius)
	filtered_test = spatial_filter(test, s_a, s_rg, s_by, radius)

	# Perceptually Uniform Color Space
	preprocessed_reference = hunt_adjustment(color_space_transform(filtered_reference, 'linrgb2lab'))
	preprocessed_test = hunt_adjustment(color_space_transform(filtered_test, 'linrgb2lab'))

	# Color metric
	deltaE_hyab = hyab(preprocessed_reference, preprocessed_test, eps)
	power_deltaE_hyab = torch.pow(deltaE_hyab, qc)
	hunt_adjusted_green = hunt_adjustment(color_space_transform(torch.tensor([[[0.0]], [[1.0]], [[0.0]]]).unsqueeze(0), 'linrgb2lab'))
	hunt_adjusted_blue = hunt_adjustment(color_space_transform(torch.tensor([[[0.0]], [[0.0]], [[1.0]]]).unsqueeze(0), 'linrgb2lab'))
	cmax = torch.pow(hyab(hunt_adjusted_green, hunt_adjusted_blue, eps), qc).item()
	deltaE_c = redistribute_errors(power_deltaE_hyab, cmax, pc, pt)

	# --- Feature pipeline ---
	# Extract and normalize Yy component
	ref_y = (reference[:, 0:1, :, :] + 16) / 116
	test_y = (test[:, 0:1, :, :] + 16) / 116

	# Edge and point detection
	edges_reference = feature_detection(ref_y, pixels_per_degree, 'edge')
	points_reference = feature_detection(ref_y, pixels_per_degree, 'point')
	edges_test = feature_detection(test_y, pixels_per_degree, 'edge')
	points_test = feature_detection(test_y, pixels_per_degree, 'point')

	# Feature metric
	deltaE_f = torch.max(
		torch.abs(torch.norm(edges_reference, dim=1, keepdim=True) - torch.norm(edges_test, dim=1, keepdim=True)),
		torch.abs(torch.norm(points_test, dim=1, keepdim=True) - torch.norm(points_reference, dim=1, keepdim=True))
	)
	deltaE_f = torch.clamp(deltaE_f, min=eps)  # clamp to stabilize training
	deltaE_f = torch.pow(((1 / np.sqrt(2)) * deltaE_f), qf)

	# --- Final error ---
	return torch.pow(deltaE_c, 1 - deltaE_f)


def tone_map(img, tone_mapper, exposure):
	"""
	Applies exposure compensation and tone mapping.
	Refer to the Visualizing Errors in Rendered High Dynamic Range Images
	paper for details about the formulas.

	:param img: float tensor (with NxCxHxW layout) containing nonnegative values
	:param tone_mapper: string describing the tone mapper to apply
	:param exposure: float tensor (with Nx1x1x1 layout) describing the exposure compensation factor
	"""
	# Exposure compensation
	x = (2 ** exposure) * img

	# Set tone mapping coefficients depending on tone_mapper
	if tone_mapper == "reinhard":
		lum_coeff_r = 0.2126
		lum_coeff_g = 0.7152
		lum_coeff_b = 0.0722

		Y = x[:, 0:1, :, :] * lum_coeff_r + x[:, 1:2, :, :] * lum_coeff_g + x[:, 2:3, :, :] * lum_coeff_b
		return torch.clamp(torch.div(x, 1 + Y), 0.0, 1.0)

	if tone_mapper == "hable":
		# Source: https://64.github.io/tonemapping/
		A = 0.15
		B = 0.50
		C = 0.10
		D = 0.20
		E = 0.02
		F = 0.30
		k0 = A * F - A * E
		k1 = C * B * F - B * E
		k2 = 0
		k3 = A * F
		k4 = B * F
		k5 = D * F * F

		W = 11.2
		nom = k0 * torch.pow(W, torch.tensor([2.0])) + k1 * W + k2
		denom = k3 * torch.pow(W, torch.tensor([2.0])) + k4 * W + k5
		white_scale = torch.div(denom, nom)  # = 1 / (nom / denom)

		# Include white scale and exposure bias in rational polynomial coefficients
		k0 = 4 * k0 * white_scale
		k1 = 2 * k1 * white_scale
		k2 = k2 * white_scale
		k3 = 4 * k3
		k4 = 2 * k4
		# k5 = k5 # k5 is not changed
	else:
		# Source:  ACES approximation: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
		# Include pre-exposure cancelation in constants
		k0 = 0.6 * 0.6 * 2.51
		k1 = 0.6 * 0.03
		k2 = 0
		k3 = 0.6 * 0.6 * 2.43
		k4 = 0.6 * 0.59
		k5 = 0.14

	x2 = torch.pow(x, 2)
	nom = k0 * x2 + k1 * x + k2
	denom = k3 * x2 + k4 * x + k5
	denom = torch.where(torch.isinf(denom), torch.Tensor([1.0]), denom)  # if denom is inf, then so is nom => nan. Pixel is very bright. It becomes inf here, but 1 after clamp below
	y = torch.div(nom, denom)
	return torch.clamp(y, 0.0, 1.0)


def compute_start_stop_exposures(reference, tone_mapper, tmax, tmin):
	"""
	Computes start and stop exposure for HDR-FLIP based on given tone mapper and reference image.
	Refer to the Visualizing Errors in Rendered High Dynamic Range Images
	paper for details about the formulas

	:param reference: float tensor (with NxCxHxW layout) containing reference images (nonnegative values)
	:param tone_mapper: string describing which tone mapper should be assumed
	:param tmax: float describing the t value used to find the start exposure
	:param tmin: float describing the t value used to find the stop exposure
	:return: two float tensors (with Nx1x1x1 layout) containing start and stop exposures, respectively, to use for HDR-FLIP
	"""
	if tone_mapper == "reinhard":
		k0 = 0
		k1 = 1
		k2 = 0
		k3 = 0
		k4 = 1
		k5 = 1

		x_max = tmax * k5 / (k1 - tmax * k4)
		x_min = tmin * k5 / (k1 - tmin * k4)
	elif tone_mapper == "hable":
		# Source: https://64.github.io/tonemapping/
		A = 0.15
		B = 0.50
		C = 0.10
		D = 0.20
		E = 0.02
		F = 0.30
		k0 = A * F - A * E
		k1 = C * B * F - B * E
		k2 = 0
		k3 = A * F
		k4 = B * F
		k5 = D * F * F

		W = 11.2
		nom = k0 * torch.pow(W, torch.tensor([2.0])) + k1 * W + k2
		denom = k3 * torch.pow(W, torch.tensor([2.0])) + k4 * W + k5
		white_scale = torch.div(denom, nom)  # = 1 / (nom / denom)

		# Include white scale and exposure bias in rational polynomial coefficients
		k0 = 4 * k0 * white_scale
		k1 = 2 * k1 * white_scale
		k2 = k2 * white_scale
		k3 = 4 * k3
		k4 = 2 * k4
		# k5 = k5 # k5 is not changed

		c0 = (k1 - k4 * tmax) / (k0 - k3 * tmax)
		c1 = (k2 - k5 * tmax) / (k0 - k3 * tmax)
		x_max = - 0.5 * c0 + torch.sqrt(((torch.tensor([0.5]) * c0) ** 2) - c1)

		c0 = (k1 - k4 * tmin) / (k0 - k3 * tmin)
		c1 = (k2 - k5 * tmin) / (k0 - k3 * tmin)
		x_min = - 0.5 * c0 + torch.sqrt(((torch.tensor([0.5]) * c0) ** 2) - c1)
	else:
		# Source:  ACES approximation: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
		# Include pre-exposure cancelation in constants
		k0 = 0.6 * 0.6 * 2.51
		k1 = 0.6 * 0.03
		k2 = 0
		k3 = 0.6 * 0.6 * 2.43
		k4 = 0.6 * 0.59
		k5 = 0.14

		c0 = (k1 - k4 * tmax) / (k0 - k3 * tmax)
		c1 = (k2 - k5 * tmax) / (k0 - k3 * tmax)
		x_max = - 0.5 * c0 + torch.sqrt(((torch.tensor([0.5]) * c0) ** 2) - c1)

		c0 = (k1 - k4 * tmin) / (k0 - k3 * tmin)
		c1 = (k2 - k5 * tmin) / (k0 - k3 * tmin)
		x_min = - 0.5 * c0 + torch.sqrt(((torch.tensor([0.5]) * c0) ** 2) - c1)

	# Convert reference to luminance
	lum_coeff_r = 0.2126
	lum_coeff_g = 0.7152
	lum_coeff_b = 0.0722
	Y_reference = reference[:, 0:1, :, :] * lum_coeff_r + reference[:, 1:2, :, :] * lum_coeff_g + reference[:, 2:3, :, :] * lum_coeff_b

	# Compute start exposure
	Y_hi = torch.amax(Y_reference, dim=(2, 3), keepdim=True)
	start_exposure = torch.log2(x_max / Y_hi)

	# Compute stop exposure
	dim = Y_reference.size()
	Y_ref = Y_reference.view(dim[0], dim[1], dim[2]*dim[3])
	Y_lo = torch.median(Y_ref, dim=2).values.unsqueeze(2).unsqueeze(3)
	stop_exposure = torch.log2(x_min / Y_lo)

	return start_exposure, stop_exposure


def generate_spatial_filter(pixels_per_degree, channel):
	"""
	Generates spatial contrast sensitivity filters with width depending on
	the number of pixels per degree of visual angle of the observer

	:param pixels_per_degree: float indicating number of pixels per degree of visual angle
	:param channel: string describing what filter should be generated
	:yield: Filter kernel corresponding to the spatial contrast sensitivity function of the given channel and kernel's radius
	"""
	a1_A = 1
	b1_A = 0.0047
	a2_A = 0
	b2_A = 1e-5  # avoid division by 0
	a1_rg = 1
	b1_rg = 0.0053
	a2_rg = 0
	b2_rg = 1e-5  # avoid division by 0
	a1_by = 34.1
	b1_by = 0.04
	a2_by = 13.5
	b2_by = 0.025
	if channel == "A":  # Achromatic CSF
		a1 = a1_A
		b1 = b1_A
		a2 = a2_A
		b2 = b2_A
	elif channel == "RG":  # Red-Green CSF
		a1 = a1_rg
		b1 = b1_rg
		a2 = a2_rg
		b2 = b2_rg
	elif channel == "BY":  # Blue-Yellow CSF
		a1 = a1_by
		b1 = b1_by
		a2 = a2_by
		b2 = b2_by

	# Determine evaluation domain
	max_scale_parameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by])
	r = np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2)) * pixels_per_degree)
	r = int(r)
	deltaX = 1.0 / pixels_per_degree
	x, y = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
	z = (x * deltaX)**2 + (y * deltaX)**2

	# Generate weights
	g = a1 * np.sqrt(np.pi / b1) * np.exp(-np.pi**2 * z / b1) + a2 * np.sqrt(np.pi / b2) * np.exp(-np.pi**2 * z / b2)
	g = g / np.sum(g)
	g = torch.Tensor(g).unsqueeze(0).unsqueeze(0)

	return g, r


def spatial_filter(img, s_a, s_rg, s_by, radius):
	"""
	Filters an image with channel specific spatial contrast sensitivity functions
	and clips result to the unit cube in linear RGB

	:param img: image tensor to filter (with NxCxHxW layout in the YCxCz color space)
	:param s_a: spatial filter matrix for the achromatic channel
	:param s_rg: spatial filter matrix for the red-green channel
	:param s_by: spatial filter matrix for the blue-yellow channel
	:return: input image (with NxCxHxW layout) transformed to linear RGB after filtering with spatial contrast sensitivity functions
	"""
	dim = img.size()
	# Prepare image for convolution
	img_pad = torch.zeros((dim[0], dim[1], dim[2] + 2 * radius, dim[3] + 2 * radius))
	img_pad[:, 0:1, :, :] = nn.functional.pad(img[:, 0:1, :, :], (radius, radius, radius, radius), mode='replicate')
	img_pad[:, 1:2, :, :] = nn.functional.pad(img[:, 1:2, :, :], (radius, radius, radius, radius), mode='replicate')
	img_pad[:, 2:3, :, :] = nn.functional.pad(img[:, 2:3, :, :], (radius, radius, radius, radius), mode='replicate')

	# Apply Gaussian filters
	img_tilde_opponent = torch.zeros((dim[0], dim[1], dim[2], dim[3]))
	img_tilde_opponent[:, 0:1, :, :] = nn.functional.conv2d(img_pad[:, 0:1, :, :], s_a, padding=0)
	img_tilde_opponent[:, 1:2, :, :] = nn.functional.conv2d(img_pad[:, 1:2, :, :], s_rg, padding=0)
	img_tilde_opponent[:, 2:3, :, :] = nn.functional.conv2d(img_pad[:, 2:3, :, :], s_by, padding=0)

	# Transform to linear RGB for clamp
	img_tilde_linear_rgb = color_space_transform(img_tilde_opponent, 'ycxcz2linrgb')

	# Clamp to RGB box
	return torch.clamp(img_tilde_linear_rgb, 0.0, 1.0)


def hunt_adjustment(img):
	"""
	Applies Hunt-adjustment to an image

	:param img: image tensor to adjust (with NxCxHxW layout in the L*a*b* color space)
	:return: Hunt-adjusted image tensor (with NxCxHxW layout in the Hunt-adjusted L*A*B* color space)
	"""
	# Extract luminance component
	L = img[:, 0:1, :, :]

	# Apply Hunt adjustment
	img_h = torch.zeros(img.size())
	img_h[:, 0:1, :, :] = L
	img_h[:, 1:2, :, :] = torch.mul((0.01 * L), img[:, 1:2, :, :])
	img_h[:, 2:3, :, :] = torch.mul((0.01 * L), img[:, 2:3, :, :])

	return img_h


def hyab(reference, test, eps):
	"""
	Computes the HyAB distance between reference and test images

	:param reference: reference image tensor (with NxCxHxW layout in the standard or Hunt-adjusted L*A*B* color space)
	:param test: test image tensor (with NxCxHxW layout in the standard or Hunt-adjusted L*a*b* color space)
	:param eps: float containing a small value used to improve training stability
	:return: image tensor (with Nx1xHxW layout) containing the per-pixel HyAB distances between reference and test images
	"""
	delta = reference - test
	root = torch.sqrt(torch.clamp(torch.pow(delta[:, 0:1, :, :], 2), min=eps))
	delta_norm = torch.norm(delta[:, 1:3, :, :], dim=1, keepdim=True)
	return root + delta_norm  # alternative abs to stabilize training


def redistribute_errors(power_deltaE_hyab, cmax, pc, pt):
	"""
	Redistributes exponentiated HyAB errors to the [0,1] range

	:param power_deltaE_hyab: float tensor (with Nx1xHxW layout) containing the exponentiated HyAb distance
	:param cmax: float containing the exponentiated, maximum HyAB difference between two colors in Hunt-adjusted L*A*B* space
	:param pc: float containing the cmax multiplier p_c (see FLIP paper)
	:param pt: float containing the target value, p_t, for p_c * cmax (see FLIP paper)
	:return: image tensor (with Nx1xHxW layout) containing redistributed per-pixel HyAB distances (in range [0,1])
	"""
	# Re-map error to 0-1 range. Values between 0 and
	# pccmax are mapped to the range [0, pt],
	# while the rest are mapped to the range (pt, 1]
	deltaE_c = torch.zeros(power_deltaE_hyab.size())
	pccmax = pc * cmax
	deltaE_c = torch.where(power_deltaE_hyab < pccmax, (pt / pccmax) * power_deltaE_hyab, pt + ((power_deltaE_hyab - pccmax) / (cmax - pccmax)) * (1.0 - pt))

	return deltaE_c


def feature_detection(img_y, pixels_per_degree, feature_type):
	"""
	Detects edges and points (features) in the achromatic image

	:param imgy: achromatic image tensor (with Nx1xHxW layout, containing normalized Y-values from YCxCz)
	:param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer
	:param feature_type: string indicating the type of feature to detect
	:return: image tensor (with Nx2xHxW layout, with values in range [0,1]) containing large values where features were detected
	"""
	# Set peak to trough value (2x standard deviations) of human edge
	# detection filter
	w = 0.082

	# Compute filter radius
	sd = 0.5 * w * pixels_per_degree
	radius = int(np.ceil(3 * sd))

	# Compute 2D Gaussian
	[x, y] = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1))
	g = np.exp(-(x ** 2 + y ** 2) / (2 * sd * sd))

	if feature_type == 'edge':  # Edge detector
		# Compute partial derivative in x-direction
		Gx = np.multiply(-x, g)
	else:  # Point detector
		# Compute second partial derivative in x-direction
		Gx = np.multiply(x ** 2 / (sd * sd) - 1, g)

	# Normalize positive weights to sum to 1 and negative weights to sum to -1
	negative_weights_sum = -np.sum(Gx[Gx < 0])
	positive_weights_sum = np.sum(Gx[Gx > 0])
	Gx = torch.Tensor(Gx)
	Gx = torch.where(Gx < 0, Gx / negative_weights_sum, Gx / positive_weights_sum)
	Gx = Gx.unsqueeze(0).unsqueeze(0)

	# Detect features
	featuresX = nn.functional.conv2d(nn.functional.pad(img_y, (radius, radius, radius, radius), mode='replicate'), Gx, padding=0)
	featuresY = nn.functional.conv2d(nn.functional.pad(img_y, (radius, radius, radius, radius), mode='replicate'), torch.transpose(Gx, 2, 3), padding=0)
	return torch.cat((featuresX, featuresY), dim=1)


def color_space_transform(input_color, fromSpace2toSpace):
	"""
	Transforms inputs between different color spaces

	:param input_color: tensor of colors to transform (with NxCxHxW layout)
	:param fromSpace2toSpace: string describing transform
	:return: transformed tensor (with NxCxHxW layout)
	"""
	dim = input_color.size()

	# Assume D65 standard illuminant
	reference_illuminant = torch.tensor([[[0.950428545]], [[1.000000000]], [[1.088900371]]])
	inv_reference_illuminant = torch.tensor([[[1.052156925]], [[1.000000000]], [[0.918357670]]])

	if fromSpace2toSpace == "srgb2linrgb":
		limit = 0.04045
		transformed_color = torch.where(
			input_color > limit,
			torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
			input_color / 12.92
		)  # clamp to stabilize training

	elif fromSpace2toSpace == "linrgb2srgb":
		limit = 0.0031308
		transformed_color = torch.where(
			input_color > limit,
			1.055 * torch.pow(torch.clamp(input_color, min=limit), (1.0 / 2.4)) - 0.055,
			12.92 * input_color
		)

	elif fromSpace2toSpace in ["linrgb2xyz", "xyz2linrgb"]:
		# Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
		# Assumes D65 standard illuminant
		if fromSpace2toSpace == "linrgb2xyz":
			a11 = 10135552 / 24577794
			a12 = 8788810  / 24577794
			a13 = 4435075  / 24577794
			a21 = 2613072  / 12288897
			a22 = 8788810  / 12288897
			a23 = 887015   / 12288897
			a31 = 1425312  / 73733382
			a32 = 8788810  / 73733382
			a33 = 70074185 / 73733382
		else:
			# Constants found by taking the inverse of the matrix
			# defined by the constants for linrgb2xyz
			a11 = 3.241003275
			a12 = -1.537398934
			a13 = -0.498615861
			a21 = -0.969224334
			a22 = 1.875930071
			a23 = 0.041554224
			a31 = 0.055639423
			a32 = -0.204011202
			a33 = 1.057148933
		A = torch.Tensor([[a11, a12, a13],
						  [a21, a22, a23],
						  [a31, a32, a33]])

		input_color = input_color.view(dim[0], dim[1], dim[2]*dim[3])  # NC(HW)

		transformed_color = torch.matmul(A, input_color)
		transformed_color = transformed_color.view(dim[0], dim[1], dim[2], dim[3])

	elif fromSpace2toSpace == "xyz2ycxcz":
		input_color = torch.mul(input_color, inv_reference_illuminant)
		y = 116 * input_color[:, 1:2, :, :] - 16
		cx = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
		cz = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
		transformed_color = torch.cat((y, cx, cz), 1)

	elif fromSpace2toSpace == "ycxcz2xyz":
		y = (input_color[:, 0:1, :, :] + 16) / 116
		cx = input_color[:, 1:2, :, :] / 500
		cz = input_color[:, 2:3, :, :] / 200

		x = y + cx
		z = y - cz
		transformed_color = torch.cat((x, y, z), 1)

		transformed_color = torch.mul(transformed_color, reference_illuminant)

	elif fromSpace2toSpace == "xyz2lab":
		input_color = torch.mul(input_color, inv_reference_illuminant)
		delta = 6 / 29
		delta_square = delta * delta
		delta_cube = delta * delta_square
		factor = 1 / (3 * delta_square)

		clamped_term = torch.pow(torch.clamp(input_color, min=delta_cube), 1.0 / 3.0).to(dtype=input_color.dtype)
		div = (factor * input_color + (4 / 29)).to(dtype=input_color.dtype)
		input_color = torch.where(input_color > delta_cube, clamped_term, div)  # clamp to stabilize training

		L = 116 * input_color[:, 1:2, :, :] - 16
		a = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
		b = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])

		transformed_color = torch.cat((L, a, b), 1)

	elif fromSpace2toSpace == "lab2xyz":
		y = (input_color[:, 0:1, :, :] + 16) / 116
		a = input_color[:, 1:2, :, :] / 500
		b = input_color[:, 2:3, :, :] / 200

		x = y + a
		z = y - b

		xyz = torch.cat((x, y, z), 1)
		delta = 6 / 29
		delta_square = delta * delta
		factor = 3 * delta_square
		xyz = torch.where(xyz > delta, torch.pow(xyz, 3), factor * (xyz - 4 / 29))

		transformed_color = torch.mul(xyz, reference_illuminant)

	elif fromSpace2toSpace == "srgb2xyz":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
	elif fromSpace2toSpace == "srgb2ycxcz":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
	elif fromSpace2toSpace == "linrgb2ycxcz":
		transformed_color = color_space_transform(input_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
	elif fromSpace2toSpace == "srgb2lab":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	elif fromSpace2toSpace == "linrgb2lab":
		transformed_color = color_space_transform(input_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	elif fromSpace2toSpace == "ycxcz2linrgb":
		transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
	elif fromSpace2toSpace == "lab2srgb":
		transformed_color = color_space_transform(input_color, 'lab2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
	elif fromSpace2toSpace == "ycxcz2lab":
		transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	else:
		sys.exit('Error: The color transform %s is not defined!' % fromSpace2toSpace)

	return transformed_color








#%% FLIP loss use

img1 = img1+1/2
img2 = img2+1/2
FLIP = LDRFLIPLoss()
loss_img = FLIP(img2, img1)


#%% dev main

# print("image shape", img.shape)
# target_area = (img.shape[0] - 5, img.shape[1] - 4)
# print("target area", target_area)
# hit_memory = []

# new_img = np.copy(img)
# show_img(new_img, title = "original img", size_scaler = 3)

# nb_amogus = 0
# for i in range(5000):
    
#     ind = get_random_point(target_area, hit_memory)
#     if ind == -1:
#         # print("index in hit memory, skip")
#         continue
#     img_crop = np.copy(new_img[ind[0]:ind[0]+5, ind[1]:ind[1]+4])
    
#     method = "kmeans" # ["baseline", "lab_proba", "kmeans"]
#     new_crop, temp_nb = main_amogus(img_crop, method=method)
#     nb_amogus += temp_nb
    
#     new_img[ind[0]:ind[0]+5, ind[1]:ind[1]+4] = new_crop
    
#     # if i%10 == 0:
#     #     show_img(new_img, size_scaler = 5)
# # 
# show_img(new_img, title = f"{nb_amogus} amoguses", size_scaler = 3)   
# print("nb amogus made:", nb_amogus)


# show_img(img_crop, title = f"{np.std(convert_to_Lab(img_crop), axis = (0, 1))}", size_scaler = 50)




#%%

# generate some random amoguses

for _ in range(1):
    blank = np.zeros((5, 4, 3), dtype = np.uint8)
    
    body_color = np.random.randint(0, 255, 3)
    visor_color = np.random.randint(0, 255, 3)
    bg_color = np.random.randint(0, 255, 3)
    
    blank = blank + body_color
    blank[tuple(visor_ind.T)] = visor_color
    blank[tuple(bg_ind.T)] = bg_color
    
    plt.imshow(blank)
    plt.show()










































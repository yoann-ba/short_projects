


#%% imports 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import opensimplex as osx

# plotly for jupyter interactive 3D plots


#%% plot functions


# https://plotly.com/python/reference/surface/
# https://plotly.com/python/reference/layout/#layout-margin
# https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html

def plot_2d(img, title = '', vmin = None, vmax = None, cmap = 'gray', size_scaler = 1):

  px = 1/plt.rcParams['figure.dpi']  # pixel in inches

  figure_size = (img.shape[0]*px*size_scaler, img.shape[1]*px*size_scaler)
  plt.figure(figsize = figure_size)
  
  plt.imshow(img, vmin = vmin, vmax = vmax, cmap = cmap)
  plt.title(title)
  plt.show()

  return


# 3D surface plot
def plot_3d_plotly(img, title = '', size_scaler = 1, height = 1):

  Z = img

  Z[0][0] = 1/height

  figure_size = (img.shape[0]*size_scaler, img.shape[1]*size_scaler)
  
  fig = go.Figure(data=[go.Surface(z=Z)])

  fig.update_layout(title=title, autosize=False,
                    width=figure_size[0], height=figure_size[1])
  
  # camera
  fig.update_traces(lighting_fresnel=0, selector=dict(type='surface'))
  fig.update_traces(lighting_specular=0.05, selector=dict(type='surface'))

  # material
  fig.update_traces(colorscale='Greys', selector=dict(type='surface'))
  #fig.update_traces(colorscale=[[0, 'rgb(155,118,83)'], [1, 'rgb(155,118,83)']], selector=dict(type='surface'))
  fig.update_traces(lighting_roughness=0.5, selector=dict(type='surface'))

  # light
  fig.update_traces(lighting_ambient=0.25, selector=dict(type='surface'))
  fig.update_traces(lighting_diffuse=0.8, selector=dict(type='surface'))

  # layout margin
  fig.update_layout(margin_b=1)
  fig.update_layout(margin_l=1)
  fig.update_layout(margin_r=1)
  fig.update_layout(margin_t=1)

  fig.show()

  return


# plot 2 3D surface plots side by side for comparison
def plot_3d_double(img1, img2, title = '', size_scaler = 1, height = 1):

  fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]], horizontal_spacing = 0)

  # adding surfaces to subplots.
  Z = img1
  Z[0][0] = 1/height
  fig.add_trace(go.Surface(z=Z), row=1, col=1)

  Z = img2
  Z[0][0] = 1/height
  fig.add_trace(go.Surface(z=Z), row=1, col=2)

  figure_size = (img1.shape[0]*size_scaler, img1.shape[1]*size_scaler)
  fig.update_layout(title=title, autosize=False,
                    width=2*figure_size[0], height=figure_size[1])
  
  # camera
  fig.update_traces(lighting_fresnel=0, selector=dict(type='surface'))
  fig.update_traces(lighting_specular=0.05, selector=dict(type='surface'))

  # material
  fig.update_traces(colorscale='Greys', selector=dict(type='surface'))
  #fig.update_traces(colorscale=[[0, 'rgb(155,118,83)'], [1, 'rgb(155,118,83)']], selector=dict(type='surface'))
  fig.update_traces(lighting_roughness=0.5, selector=dict(type='surface'))

  # light
  fig.update_traces(lighting_ambient=0.25, selector=dict(type='surface'))
  fig.update_traces(lighting_diffuse=0.8, selector=dict(type='surface'))

  # layout margin
  fig.update_layout(margin_b=1)
  fig.update_layout(margin_l=1)
  fig.update_layout(margin_r=1)
  fig.update_layout(margin_t=1)

  fig.show()

  return

#%% terrain generation

# copied over from some of my other scripts

# juste one octave, opensimplex call
def generate_octave(size = (200, 200), feature_size = 24, seed = 1234):

  if seed == 'random':
    osx.seed(np.random.randint(0, 1000))
  else:
    osx.seed(seed)

  base_x = np.arange(size[0]) / feature_size
  base_y = np.arange(size[1]) / feature_size

  test = osx.noise2array(base_x, base_y)

  return test


# classic FBM, halves the amplitude, halves the period
def generate_map(nb_octave = 4, reduction_coef = 0.5, size = (200, 200), start_feature_size = 48, seed = 1234):

  map = generate_octave(size = size, feature_size = start_feature_size, seed = seed)

  for i_oct in range(1, nb_octave):
    map += reduction_coef**i_oct * generate_octave(size = size, feature_size = start_feature_size/(2**i_oct), seed = seed)

  return map

  
# homemade "integral"-ish equivalent to the classic FBM
def generate_map_continuous(reduction_coef = 0.5, size = (200, 200), start_feature_size = 48, seed = 1234):

  map = 0 * generate_octave(size = size, feature_size = start_feature_size, seed = seed)

  current_fs = start_feature_size
  i_oct = 1
  while current_fs >= 2:
    #print(current_fs, i_oct, reduction_coef**np.log2(start_feature_size/current_fs))
    map += reduction_coef**np.log2(start_feature_size/current_fs) * generate_octave(size = size, feature_size = current_fs, seed = seed)
    current_fs -= 1
    i_oct += 1

  return (map - np.min(map))/(np.max(map) - np.min(map))
# math for the log: 
# 48/2**n -> n
# n' -> n
# n' = 48/2**n
# 2**n = 48/n'
# n = log2(48/n')


#%% erosion funcs

# Sebastian Lague (Hans Bayer) https://github.com/SebLague/Hydraulic-Erosion
# Hans Bayer 
# https://www.firespark.de/resources/downloads/implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf
# Jacob Olsen, old, global terrain generation http://web.mit.edu/cesium/Public/terrain.pdf


# could change the randint by the newer np random generator
# hydraulic erosion
def particle_drop(img, nb_iters = 10):

  temp_border = np.full(img.shape[1], np.max(img)+1)
  new_img = np.vstack([temp_border, img, temp_border])
  temp_border = np.full((new_img.shape[0], 1), np.max(img)+1)
  new_img = np.hstack([temp_border, new_img, temp_border])

  #print(new_img)

  for i_dr in range(nb_iters):
    pt_i = np.random.randint(1, new_img.shape[0]-2)
    pt_j = np.random.randint(1, new_img.shape[1]-2)

    droplet_charge = 0.0

    stopped = False
    while not stopped:
      list_inds = [[pt_i, pt_j-1], 
                  [pt_i, pt_j+1], 
                  [pt_i-1, pt_j], 
                  [pt_i+1, pt_j]]
      nbrs = [new_img[pt_i][pt_j-1], 
              new_img[pt_i][pt_j+1], 
              new_img[pt_i-1][pt_j], 
              new_img[pt_i+1][pt_j]]
      if np.min(nbrs) >= new_img[pt_i][pt_j]:
        stopped = True
        temp_min = np.min(nbrs)
        amount_left = temp_min - new_img[pt_i][pt_j] # some disappear
        new_img[pt_i][pt_j] += amount_left * 1.01 # if ended, leave all material here
      else:
        [temp_pt_i, temp_pt_j] = list_inds[np.argmin(nbrs)]
        delta_h = new_img[pt_i][pt_j] - new_img[temp_pt_i][temp_pt_j]
        gathered_content = 0.5*delta_h
        droplet_charge += gathered_content
        new_img[pt_i][pt_j] -= gathered_content

        deposited_content = min(0.5*delta_h, droplet_charge)
        new_img[temp_pt_i][temp_pt_j] += deposited_content

        pt_i, pt_j = temp_pt_i, temp_pt_j
    #
    #print(new_img[1:-1, 1:-1])
    #print('')
  #

  return new_img[1:-1, 1:-1]



def get_gradient(img, i, j, u, v):

  temp_x = (img[i+1][j] - img[i][j]) * (1-v) + (img[i+1][j+1] - img[i][j+1]) * v # coefs swap, that's fine
  temp_y = (img[i][j+1] - img[i][j]) * (1-u) + (img[i+1][j+1] - img[i+1][j]) * u

  return np.array([temp_x, temp_y])


# Row-first then column 
# same result as if we had done column first
def get_height(img, i, j, u, v):

  temp_h = img[i][j] * (1-v) + img[i][j+1] * v # first row
  temp_h2 = img[i+1][j] * (1-v) + img[i+1][j+1] * v # second row, same parameters

  return temp_h * (1-u) + temp_h2 * u # 'middle' column


def update_position(pt_i, pt_j, pt_u, pt_v, dir):
  
  #print('input update', pt_i, pt_u, pt_j, pt_v, dir)
  pt_i_new = pt_i + pt_u + dir[0]
  pt_j_new = pt_j + pt_v + dir[1]
  
  #print(pt_i_new, pt_j_new)

  pt_u_new = pt_i_new - int(pt_i_new)
  pt_v_new = pt_j_new - int(pt_j_new)

  pt_i_new = int(pt_i_new)
  pt_j_new = int(pt_j_new)

  return pt_i_new, pt_j_new, pt_u_new, pt_v_new


def deposit_sediment(img, pt_i, pt_j, pt_u, pt_v, deposited_amount):

  img[pt_i][pt_j] += deposited_amount * (1 - pt_u) * (1 - pt_v)
  img[pt_i][pt_j + 1] += deposited_amount * (1 - pt_u) * pt_v
  img[pt_i + 1][pt_j] += deposited_amount * pt_u * (1 - pt_v)
  img[pt_i + 1][pt_j + 1] += deposited_amount * pt_u * pt_v

  return img


def erode_sediment(img, pt_i, pt_j, pt_u, pt_v, gathered_amount, radius):

  x0 = pt_i - radius
  y0 = pt_j - radius

  # construct the square that fits in the image
  x_start = max(1, x0)
  y_start = max(1, y0)
  x_end = min(img.shape[1]-2, x0 + 2*radius+1)
  y_end = min(img.shape[0]-2, y0 + 2*radius+1)

  store_weight = []
  store_sum = 0
  for x_i in range(x_start, x_end):
    for y_j in range(y_start, y_end):
      dx = x_i - (pt_i + pt_u)
      dy = y_j - (pt_j + pt_v)
      dist = np.sqrt(dx*dx + dy*dy)
      temp_weight = max(0, radius - dist)
      store_weight.append(temp_weight)
      store_sum += temp_weight
  #

  list_idx = 0
  for x_i in range(x_start, x_end):
    for y_j in range(y_start, y_end):
      img[x_i][y_j] -= gathered_amount * store_weight[list_idx] / store_sum
      list_idx += 1
  #
  #print('weights', store_weight)
  #print('sum weights', store_sum)

  return img


# https://www.firespark.de/resources/downloads/implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf
# https://github.com/henrikglass/erodr/blob/master/src/erodr.c
# sub method
# 'img' is the base image + padding
def drop_hans_bayer(img, w_inertia, w_min_slope, w_capacity, w_erosion, w_gravity, w_evaporation, w_deposition, w_radius, max_iters = 20):

  # start pos = (i + u, j + v)
  pt_i = np.random.randint(1, img.shape[0]-2)
  pt_j = np.random.randint(1, img.shape[1]-2)

  pt_u = np.random.random() # for i
  pt_v = np.random.random() # for j

  direction = np.array([0.0, 0.0])
  velocity = 0.0
  sediment_content = 0.0
  water_content = 1.0

  for i in range(max_iters):
    # Gradient at the old position
    gradient = get_gradient(img, pt_i, pt_j, pt_u, pt_v)
    # break if gradient too low?%%

    # Update direction, normalize it
    direction = direction * w_inertia - gradient * (1 - w_inertia)
    direction = direction / (np.linalg.norm(direction) + 1e-10) # normalize to move per grid

    # Get old height, new position, new height
    height_old = get_height(img, pt_i, pt_j, pt_u, pt_v)
    pt_i_n, pt_j_n, pt_u_n, pt_v_n = update_position(pt_i, pt_j, pt_u, pt_v, direction)
    if pt_i_n <= 0: # <= 0 because padding
      #print('drop left img')
      break
    if pt_j_n <= 0:
      #print('drop left img')
      break
    if pt_i_n >= img.shape[0] - 1: # >= n-1 because padding
      #print('drop left img')
      break
    if pt_j_n >= img.shape[1] - 1:
      #print('drop left img')
      break
    height_new = get_height(img, pt_i_n, pt_j_n, pt_u_n, pt_v_n)

    # Get height difference
    height_dif = height_new - height_old

    # Update sediment carry capacity
    carry_capacity = max(-height_dif, w_min_slope) * velocity * water_content * w_capacity

    # Erode or deposit
    if (height_dif >= 0) or (sediment_content > carry_capacity): # we're depositing
      if height_dif >= 0: # we're going uphill
        deposited_amount = min(sediment_content, height_dif)
      else: # we're overloaded
        deposited_amount = (sediment_content - carry_capacity) * w_deposition
      sediment_content -= deposited_amount
      img = deposit_sediment(img, pt_i, pt_j, pt_u, pt_v, deposited_amount)
    else: # downhill
      gathered_amount = min((carry_capacity - sediment_content) * w_erosion, -height_dif) # can't take more than hdif
      sediment_content += gathered_amount
      img = erode_sediment(img, pt_i, pt_j, pt_u, pt_v, gathered_amount, w_radius)
    #
    # Update Velocity and Water content
    velocity = np.sqrt(velocity*velocity + abs(height_dif) * w_gravity) #% changed to abs
    # break if velocity too low?%%
    water_content *= (1 - w_evaporation) # mb water should depend on time passed per iter? dx/vel ?
    if water_content <= 0.01:
      break

    # Old pos = new pos
    pt_i, pt_j = pt_i_n, pt_j_n
    pt_u, pt_v = pt_u_n, pt_v_n
  #

  return img


# https://www.firespark.de/resources/downloads/implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf
# nb_iters : nb of drops
def particle_hans_bayer(img, nb_iters = 1000, w_inertia = 0.3, w_min_slope = 0.01, w_capacity = 8, w_erosion = 0.7, 
                        w_gravity = 10, w_evaporation = 0.05, w_deposition = 0.2, w_radius = 4):

  # border to avoid issues
  # copy the corresponding part, and add a bit
  new_img = np.vstack([img[0] + 0.1, img, img[-1] + 0.1])
  new_img = np.hstack([np.expand_dims(new_img[:, 0], axis = 1) + 0.1, new_img, np.expand_dims(new_img[:, -1], axis = 1) + 0.1])

  for i_it in range(nb_iters):
    if i_it%500 == 0:
      print(f'\r {100*i_it/nb_iters:0.2f}%', end = '')
    drop_hans_bayer(new_img, 
                    w_inertia = w_inertia, w_min_slope = w_min_slope, w_capacity = w_capacity, 
                    w_erosion = w_erosion, w_gravity = w_gravity, w_evaporation = w_evaporation, 
                    w_deposition = w_deposition, w_radius = w_radius)
    #

  return new_img[1:-1, 1:-1]



#%% main/call


# test_erosion = generate_map(nb_octave = 20, start_feature_size = 2**15, 
#                             size = (200, 200), seed = 'random', 
#                             reduction_coef = 0.5)
# test_erosion_hb = particle_hans_bayer(test_erosion, nb_iters = int(200*200*1))






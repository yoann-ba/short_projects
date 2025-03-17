

#%% imports


import numpy as np
import matplotlib.pyplot as plt

import os



#%% funcs

def make_palette(rgb_1, rgb_2, 
                 title = '', img_size = (50, 30, 3)):
    
    left = np.full(img_size, rgb_1)
    right = np.full(img_size, rgb_2)
    
    left = np.hstack((left, right))
    # print(left.shape)
    plt.imshow(left)
    plt.title(label = title)
    plt.show()
    
    return


#%% main


def run_palettes(size = 20, seed = None):
    
    rng_gen = np.random.default_rng(seed = seed)
    table_1 = rng_gen.integers(low = 0, high = 255+1, size = (size, 3), dtype = np.uint8)
    
    end_table = []
    
    for i in range(0, len(table_1)):
        triple = table_1[i]
        triple_2 = triple + rng_gen.integers(low = -5, high = 5+1, size = 3)
        triple_2 = np.clip(triple_2, 0, 255)
        make_palette(triple, triple_2, title = i)
        is_different = input(f"Are images of pair {i} {triple} {triple_2} different?")
        if is_different == 'y' or is_different == '1':
            is_different = 1
        elif is_different == 'n' or is_different == '0':
            is_different = 0
        else:
            print("1, y, 0, n accepted as values, skipping..")
            continue
        end_table.append(list(triple) + list(triple_2) + [is_different])
    # 
    
    return end_table



#%% sandbox


path = "" + "color_difference_dictionary.npy"

if os.path.isfile(path):
    storage = np.load(path)
else:
    # R G B | R G B | is_different
    storage = np.array([[0, 0, 0, 0, 0, 0, 0]], dtype = np.uint8)

new_data = run_palettes()
new_data = np.array(new_data, dtype = np.uint8)

storage = np.concatenate((storage, new_data))
print(f"saving at {path}")
np.save(path, storage)
























































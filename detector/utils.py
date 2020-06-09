import numpy as np
import matplotlib.pyplot as plt
from simulator.simulator import *


# generate N random means within img_size
def generate_rand_mean(img_size, N):
    centers = []
    for i in range(N):
        x = np.random.rand(2) * img_size
        x = [int(y) for y in x]
        centers.append(x)

    return np.array(centers)


def generate_rand_psd_matrix(N):
    variances = []
    for i in range(N):
        A = (np.random.rand(2, 2)-0.5)*5
        B = np.dot(A, A.T) 
        variances.append(B)
    return np.array(variances)

# def generate_training_data(img_size, N, signal_noise_ratio):
#     """
#     generate data and label for training
#     Inputs: 
#         img_size: image size
#         N: number of data points
#         signal_noise_ratio: strength of signal / strength of photon noise
#     Outputs:
#         data: N matrices of size img_size X img_size
#         labels: one-hot label vector of size N. 1: there is atom
#     """
#     signal_strength = 100
#     noise_strength = signal_strength / signal_noise_ratio
#     # print(noise_strength)
#     # center = np.array([img_size // 2, img_size // 2])    
#     x0, y0 = img_size//2, img_size//2 

#     signal = create_signal(img_size, 1, x0, y0, spread=5, strength=signal_strength)

#     data = []
#     labels = []

#     for i in range(N):
#         # TODO: background_noise strength should be parametrized by noise_strength
#         background_noise = create_background_shot_noise(img_size, strength=noise_strength)
#         print(background_noise[0,0])
#         a = np.random.rand()
#         if a > 0.5:
#             photon_shot_noise = create_photon_shot_noise(0.1, signal+background_noise, strength=noise_strength)
#             data.append(signal + photon_shot_noise + background_noise)    
#             labels.append(1)
#         else:
#             photon_shot_noise = create_photon_shot_noise(0.1, background_noise, strength=noise_strength) 
#             data.append(photon_shot_noise + background_noise)
#             labels.append(0)

#     return np.array(data), np.array(labels)


def box_luminosity(data, box_size, x, y):
    """
    compute the total luminosity inside a box
    Inputs:
        data: a list of data matrices
        box_size: side length of the box
        x: x coordinate of the upper left corner
        y: y coordinate of the upper left corner
    Outputs:
        output: a list of luminosities
    """
    img_size = data[0].shape[0]
    if x + box_size - 1 > img_size or y + box_size - 1 > img_size:
        raise Exception("Box is not contained in data matrix")

    output = []
    for d in data:
        box = d[x:x+box_size, y:y+box_size]
        luminosity = np.sum(box)
        output.append(luminosity)

    return np.array(output)


def find_threshold(luminosity):
    """
    Calculate the threshold value by computing a weighted average of the luminosity list (centroid method)
    i.e. sum(luminosity * count) / tot_count
    Inputs:
        luminosity: a list where each entry is the luminosity of the corresponding data
    """
    return np.sum(luminosity) / luminosity.shape[0]

#TODO: it's better to create a classifier class, 
# so predict will just take in data as an argument
def predict(data, thresh, box_size, x, y):
    """
    Returns 1 or 0 for if there is an atom in data
    """
    luminosity = box_luminosity(data, box_size, x, y)
    this_thresh = find_threshold(luminosity)

    return None


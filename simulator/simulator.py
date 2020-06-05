import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing as mp
# from astropy.convolution import AiryDisk2DKernel
from astropy.convolution import *
from scipy import signal
import time


img_size = 250

# img_size is the side length of the pixel array
def init_image(img_size):

    return np.random.rand(img_size, img_size)




# very slow: n^2
# supports oblique gaussians
def create_signal_slow(img_size, N, centers, variances=None, spread=1, strength=1, verbose=False):
    # n: num of atoms
    # centers: array of (x,y) tuples
    # variance: array of variance/covariance matrix for each spread
    # strength: intensity at maximum peak
    # spread: used to multiple with cov matrices
    # assumes cov matrix is always invertible. otherwise, we have to add a small diagonal term

    def gaussian(x, mean, var, invertible=True):
        var = var * spread
        # det = abs(np.linalg.det(var))
        # print(det)
        # constant = strength / np.sqrt((2*np.pi)* (2*np.pi) * det)
        constant = strength
        v = x - mean

        if not invertible:
            # add a small number to diagonal
            eps = np.zeros((2,2)) + 1e-4
            var = var + eps
        
        #TODO: let variance be a fixed number, eliminates the inverse
        var_inverse = np.linalg.inv(var)
        # calculate v.T dot var_inverse dot v
        #TODO: scalar multiplication
        b = np.dot(v, np.dot(var_inverse, v))
        exponent = np.exp(-b/2)
        return constant * exponent

    atom_plots = []    

    for idx in range(N):
        # default to the identity matrix
        var = np.identity(2)
        if variances is not None:
            var = variances[idx]

        if len(centers.shape) == 1:
            center = centers
        else:
            center = centers[idx]

        # apply gaussian to each pixel of image
        atom_plot = np.zeros((img_size, img_size))
        for i in range(img_size):
            for j in range(img_size):
                # coord vector
                x = np.array([i, j])
                atom_plot[i,j] = gaussian(x, center, var)
        # if verbose:
        #     visualize(atom_plot)
        atom_plots.append(atom_plot)

        # need a 2d array where each input is a R^2 position vector
    atom_plots = np.array(atom_plots)
    output = np.sum(atom_plots, axis=0)
    
    if verbose:
        visualize(output)
    return output


def create_signal(img_size, N, x0, y0, spread=1, strength=1, verbose=False):
    def gaussian(x, y, x0, y0, s, A):
        return A* np.exp(-(x-x0)**2 / (2*s)) * np.exp(-(y-y0)**2 / (2*s))

    x = np.linspace(0, img_size-1, img_size)
    y = np.linspace(0, img_size-1, img_size)
    X, Y = np.meshgrid(x, y)    
    signal = gaussian(X, Y, x0, y0, spread, strength)
    if verbose:
        visualize(signal)
    return signal

# creates isotropic gaussians with variance
# does not support oblique gaussians
# fast and uses little memory
def create_signal_fast(img_size, N, centers, variance, strength=1, verbose=False):
    return None




# convolute the object signal matrix (gaussians + photon noise) with PSF (airy disk) + noise to create img
def convolution(obj):
    # compute obj conv h
    # of the three different types of noises, read, dark current, photon, photon noise is the only
    # one that needs to be added to obj before convolution
    airy_disk_kernel = AiryDisk2DKernel(10)
    img_conv = convolve(obj, airy_disk_kernel) 
    return img_conv



def create_read_noise(img_size, stdev, offset, c=1):
    #1. read noise: gaussian, IID for each pixel. added an offset as a nonzero mean (we never want the signal to be negative)
    # c = integral of gaussian, aka multiplicative factor

    noise = c * np.random.normal(scale=stdev, size=(img_size, img_size))
    noise = noise + abs(np.min(noise))
    return noise






def create_dark_current_noise():
    return 0

#TODO: remove the strength variable from this
def create_photon_shot_noise(exposure_time, signal, strength=1):
    # generate poisson distribution with lambda = exposure time * num of photons on each pixel
    # for a crude model, we assume num of photon == signal 

    #//TODO: lambda has to be multiplied by strength
    lam = signal * exposure_time *strength
    # print("lam")
    # print(lam)
    # print("lamb:{}".format(lam))
    return np.random.poisson(lam=lam, size=signal.shape) # poisson gives an integer count for number of photons, 
    # each photon gives strength eV
    
def create_background_shot_noise(img_size, strength=10):
    return np.full((img_size, img_size), strength)


    
def visualize(mat2d, figsize=5):
    
    fig = plt.figure(figsize=(figsize, figsize))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(mat2d, cmap='hot')
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    cbaxes = fig.add_axes([0.95, 0.1, 0.03, 0.8]) 
    plt.colorbar(orientation='vertical', cax=cbaxes)
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    # cb = plt.colorbar(ax, cax = cax)  
    plt.show()



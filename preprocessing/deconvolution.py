import os, sys
sys.path.append('../AtomDetector/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from utils import *
import simulator



"""
Implement several deconvolution algorithms
"""

def inv_filter(img, kernel):
    img.astype(np.float)
    kernel.astype(np.float)

    img_fourier = np.fft.fft2(img)
    kernel_fourier = np.fft.fft2(kernel)
    result = img_fourier / kernel_fourier    
    return np.abs(np.fft.ifft2(result))



def regularized_filter(img, kernel, alpha=0.80, high_pass_filter=None):
    if high_pass_filter is None:
        high_pass_filter = make_circ_mask(img.shape, 4)
        high_pass_filter = -high_pass_filter + 1

    high_pass_filter = np.fft.ifftshift(high_pass_filter)
    # plt.imshow(high_pass_filter)
    # plt.colorbar()
    img_fourier = np.fft.fft2(img)
    kernel_fourier = np.fft.fft2(kernel)

    high_pass_filter_abs = np.abs(high_pass_filter)
    filtered_img_fourier = kernel_fourier * img_fourier / (kernel_fourier**2 + alpha * (high_pass_filter_abs**2))
    return np.abs(np.fft.fftshift(np.fft.ifft2(filtered_img_fourier)))





def richardson_lucy_deconv(img, kernel, tot_iter, init=0.5):
    """
    Compute the Richardson Lucy Deconvolution for TOT_ITER times
    """
    if np.ndim(init) > 0:
        g1 = init
    else:
        g1 = np.full(img.shape, init)     # init to something nonnegative
    kernel_mirror = kernel[::-1, ::-1]
    for _ in range(tot_iter):
        denom = np.abs(fftconvolve_wrap(kernel, g1))
        frac = img / denom
        g1 = g1 * np.abs(fftconvolve_wrap(frac, kernel_mirror))

    return g1





# noise_spectra and obj_spectra should be in fourier space
def wiener_deconv(img, kernel, noise_spectra=None, obj_spectra=None):
    """
    computes 1/H * (1/(1+1/(H^2+S/V)))
    Wiener deconv assumes noise has average of 0, but in this case the average is not 0
    """

    # img_spectra = noise_spectra + obj_spectra \times |kernel_fourier|^2
    if noise_spectra is None or obj_spectra is None: 
        img_fourier = np.fft.fft2(img)
        kernel_fourier = np.fft.fft2(kernel)

        img_spectra = np.abs(img_fourier)**2
        noise_spectra = np.full(img_fourier.shape, 1.) # noise has a default mean of 1
        obj_spectra = (img_spectra - noise_spectra) / (np.abs(kernel_fourier) ** 2)


    SNR = noise_spectra / obj_spectra
    kernel_fourier = np.fft.fft2(kernel)
    img_fourier = np.fft.fft2(img)
    a = 1 / kernel_fourier * (1 / (1 + 1/(np.abs(kernel_fourier)**2 + SNR)))
    return np.abs(np.fft.fftshift(np.fft.ifft2(a * img_fourier)))

    






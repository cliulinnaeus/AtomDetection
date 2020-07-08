import numpy as np
from scipy.signal import convolve2d



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


def spectral_filter(img, kernel, alpha):

    

    return None

def richardson_lucy_deconv(measured, kernel, tot_iter, init=0.5):
    """
    Compute the Richardson Lucy Deconvolution for TOT_ITER times
    """
    g1 = np.full(measured.shape, init)     # init to something nonnegative
    kernel_mirror = kernel[::-1, ::-1]
    for _ in range(tot_iter):
        denom = np.abs(convolve(kernel, g1))
        frac = measured / denom
        g1 = g1 * np.abs(convolve(frac, kernel_mirror))

    return g1

def convolve(a, b):
    A = np.fft.fft2(a[::-1, ::-1])
    B = np.fft.fft2(b)
    result = np.abs(np.fft.ifft2(A * B))
    return result

def wiener_deconv(img, kernel, noise_spectra, obj_spectra):
    """
    computes 1/H * (1/(1+1/(H^2+S/V)))
    Wiener deconv assumes noise has average of 0, but in this case the average is not 0
    """
    SNR = noise_spectra / obj_spectra
    kernel_fourier = np.fft.fft2(kernel)
    img_fourier = np.fft.fft2(img)
    a = 1 / kernel_fourier * (1 / (1 + 1/(np.abs(kernel_fourier)**2 + SNR)))
    return np.fft.ifft(a * img_fourier)

    




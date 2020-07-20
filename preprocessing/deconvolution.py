import numpy as np
from scipy.signal import convolve2d
from scipy.signal import fftconvolve


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



def regularized_filter(img, kernel, alpha, high_pass_filter=None):
    if high_pass_filter == None:
        high_pass_filter = make_circ_mask(img.shape, 20)
        high_pass_filter = -high_pass_filter + 1
    img_fourier = np.fft.fft2(img)
    kernel_fourier = np.fft.fft2(kernel)

    high_pass_filter_abs = np.abs(high_pass_filter)
    filtered_img_fourier = kernel_fourier * img_fourier / (high_pass_filter_abs**2 + alpha * (high_pass_filter_abs**2))
    return np.abs(np.fft.ifft2(filtered_img_fourier))


def richardson_lucy_deconv(measured, kernel, tot_iter, init=0.5):
    """
    Compute the Richardson Lucy Deconvolution for TOT_ITER times
    """
    if np.ndim(init) > 0:
        g1 = init
    else:
        g1 = np.full(measured.shape, init)     # init to something nonnegative
    kernel_mirror = kernel[::-1, ::-1]
    for _ in range(tot_iter):
        denom = np.abs(fftconvolve(kernel, g1, 'same'))
        frac = measured / denom
        g1 = g1 * np.abs(fftconvolve(frac, kernel_mirror, 'same'))

    return g1



# noise_spectra and obj_spectra should be in fourier space
def wiener_deconv(img, kernel, noise_spectra, obj_spectra):
    """
    computes 1/H * (1/(1+1/(H^2+S/V)))
    Wiener deconv assumes noise has average of 0, but in this case the average is not 0
    """
    SNR = noise_spectra / obj_spectra
    kernel_fourier = np.fft.fft2(kernel)
    img_fourier = np.fft.fft2(img)
    a = 1 / kernel_fourier * (1 / (1 + 1/(np.abs(kernel_fourier)**2 + SNR)))
    return np.abs(np.fft.fftshift(np.fft.ifft2(a * img_fourier)))

    
def make_circ_mask(shape, radii):
    h, w = shape
    Y, X = np.ogrid[:h, :w]
    center = (int(w / 2), int(h / 2))
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radii
    mask = np.where(mask, 1., 0.)
    return mask




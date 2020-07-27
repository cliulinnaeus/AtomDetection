import sys
sys.path.append('../AtomDetector/')
import numpy as np 
import matplotlib.pyplot as plt
import time 
import simulator


def visualize(mat2d, figsize=5, title=None, xlabel=None, ylabel=None):
    
    fig = plt.figure(figsize=(figsize, figsize))

    ax = fig.add_subplot(111)
    if title == None:
        ax.set_title('colorMap')
    else: 
        ax.set_title(title)
    plt.imshow(mat2d, cmap='hot')
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    cbaxes = fig.add_axes([0.95, 0.1, 0.03, 0.8]) 
    plt.colorbar(orientation='vertical', cax=cbaxes)
    if xlabel == None or ylabel == None:
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    plt.show()


def gaussian_kernel(img_size, variance, verbose=False):
    x = y = np.linspace(0, img_size-1, img_size)
    xgrid, ygrid = np.meshgrid(x, y)  
    def _gaussian(x, y, x0, y0, s):
        return np.exp(-(x-x0)**2 / (2*s)) * np.exp(-(y-y0)**2 / (2*s))
    kernel = _gaussian(xgrid, ygrid, img_size//2, img_size//2, variance) * 1/(2*np.pi*variance)

    if verbose:
        visualize(kernel)
    return kernel

def gaussian(x, y, x0, y0, s):
        return np.exp(-(x-x0)**2 / (2*s)) * np.exp(-(y-y0)**2 / (2*s))


def make_circ_mask(shape, radii):
    h, w = shape
    Y, X = np.ogrid[:h, :w]
    center = (int(w / 2), int(h / 2))
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radii
    mask = np.where(mask, 1., 0.)
    return mask


def make_square_mask(shape, height, width):
    mask = np.zeros(shape)
    H, W = shape
    x = (H - height) // 2
    y = (W - width) // 2
    mask[x:x+height+1, y:y+width+1] = 1
    return mask



def fftconvolve_wrap(img, kernel):
    # padded_img = np.zeros(img.shape * 3)
    # padded_img 
    padded_img = np.pad(img, img.shape[0], mode='wrap')
    padded_kernel = np.pad(kernel, kernel.shape[0], mode='constant')
    padded_img_fourier = np.fft.fft2(padded_img)
    padded_kernel_fourier = np.fft.fft2(padded_kernel)
    padded_result = np.abs(np.fft.fftshift(np.fft.ifft2(padded_img_fourier * padded_kernel_fourier)))
    H, W = img.shape
    result = padded_result[H:2*H, W:2*W]
    return result


# data label pairs generator 
# SNR must be an array
def generate_data_labels(img_size, x0, y0, SNR, N):
    prev_time = time.time()

    sim = simulator.simulator(img_size, 1)

    data_tr_for_different_snr = []
    labels_tr_for_different_snr = []
    for snr in SNR:
        data_tr = []
        labels_tr = []
        for i in range(N):
            r = np.random.rand()
            if r > 0.5: 
                labels_tr.append(1)
                
                data = sim.create_simulation_from_SNR(x0, y0, snr)
                data_tr.append(data)
            else:
                labels_tr.append(0)
                data = sim.create_simulation_from_SNR(x0, y0, snr, no_atom=True)
                data_tr.append(data)
        data_tr_for_different_snr.append(np.array(data_tr))
        labels_tr_for_different_snr.append(np.array(labels_tr))

    print(f"time used: {time.time() - prev_time}")
    return np.array(data_tr_for_different_snr), np.array(labels_tr_for_different_snr)


# normalize to have a variance of 1, mean of 0
def normalize(data_batch): 
    mean = np.mean(data_batch, axis=0)
    std = np.std(data_batch, axis=0)

    return (data_batch - mean) / std




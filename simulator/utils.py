import numpy as np
import matplotlib.pyplot as plt

class simulator():

    variance = 5
    quantum_efficiency = 1 
    def __init__(self, img_size, exposure_time):
        self.img_size = img_size
        self.exposure_time = exposure_time

    def _photon_signal_to_electric_signal(self, signal):
        electric_signal = self.exposure_time * signal * simulator.quantum_efficiency
        return electric_signal

    # poisson returns integers
    def _shot_noise_from_signal(self, signal):
        lam = self._photon_signal_to_electric_signal(signal)
        return np.random.poisson(lam=lam, size=signal.shape)
    
    def _create_signal(self, x0, y0, photons_from_atom, verbose=False):       
        x = y = np.linspace(0, self.img_size-1, self.img_size)
        xgrid, ygrid = np.meshgrid(x, y)  
        signal = gaussian(xgrid, ygrid, x0, y0, simulator.variance) * photons_from_atom
        if verbose:
            visualize(signal)
        return signal
        
    def _create_background(self, photons_in_background):
        return np.full((self.img_size, self.img_size), photons_in_background)


    def create_simulation(self, x0, y0, photons_from_atom, photons_in_background, no_atom=False, verbose=False):
        
        if no_atom:
            atom_count = 0
            N = 1
        elif np.ndim(x0) == 1:
            if x0.shape[0] != y0.shape[0]:
                raise Exception(f"Shape doesn't match for x0 and y0:\n {x0.shape[0]} is not {y0.shape[0]}")
            atom_count = N = x0.shape[0]
        else:
            atom_count = N = 1        
        
        signals = []
        for i in range(N):           
            # is a number
            if no_atom: 
                atom_signal = 0
            else:
                if np.ndim(x0) == 0:
                    atom_signal = self._create_signal(x0, y0, photons_from_atom)    
                else:
                    atom_signal = self._create_signal(x0[i], y0[i], photons_from_atom)
            background_signal = self._create_background(photons_in_background)

            shot_noise = self._shot_noise_from_signal(atom_signal + background_signal)
            signals.append(shot_noise + self._photon_signal_to_electric_signal(atom_signal + background_signal))
        output = np.sum(np.array(signals), axis=0)
        if verbose:
            visualize(output)
            print(self)
            print(f"atom count: {atom_count}")
            print(f"photons per atom: {photons_from_atom}")
            print(f"photons_in_background: {photons_in_background}")
            print(f"x0: {x0}")
            print(f"y0: {y0}")
        return output

    def create_simulation_from_SNR(self, x0, y0, SNR, no_atom=False, verbose=False):
        if no_atom:
            atom_count = 0
        elif np.ndim(x0) == 1:
            if x0.shape[0] != y0.shape[0]:
                raise Exception(f"Shape doesn't match for x0 and y0:\n {x0.shape[0]} is not {y0.shape[0]}")
            atom_count = x0.shape[0]
        else:
            atom_count = 1   
        output = self.create_simulation(x0, y0, SNR, 1, no_atom=no_atom)
        if verbose:
            visualize(output)
            print(self)
            print(f"atom count: {atom_count}")
            print("photons_in_background is set to 1 by default")
            print(f"SNR: {SNR}")
            print(f"x0: {x0}")
            print(f"y0: {y0}")

        return output

    def __str__(self):
        string = f"Instance info:\nimg_size: {self.img_size}\nexposure_time: {self.exposure_time}\natom_variance: {simulator.variance}\nquantum_efficiency: {simulator.quantum_efficiency}"

        return string



def gaussian(x, y, x0, y0, s):
        return np.exp(-(x-x0)**2 / (2*s)) * np.exp(-(y-y0)**2 / (2*s))


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

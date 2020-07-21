import sys
sys.path.append('../AtomDetector/')
import numpy as np
import matplotlib.pyplot as plt
from utils import gaussian, visualize


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
            x0 = np.array(x0)
            y0 = np.array(y0)
            if x0.shape[0] != y0.shape[0]:
                raise Exception(f"Shape doesn't match for x0 and y0:\n {x0.shape[0]} is not {y0.shape[0]}")
            atom_count = N = x0.shape[0]
        else:
            atom_count = N = 1        
        
        signals = []
        for i in range(N):           
            if no_atom: 
                atom_signal = 0
            else:
                if np.ndim(x0) == 0:
                    atom_signal = self._create_signal(x0, y0, photons_from_atom)    
                else:
                    atom_signal = self._create_signal(x0[i], y0[i], photons_from_atom)
            # visualize(background_signal)
            shot_noise = self._shot_noise_from_signal(atom_signal)
            # signals.append(shot_noise + self._photon_signal_to_electric_signal(atom_signal + background_signal))
            signals.append(shot_noise)
        background_noise = self._shot_noise_from_signal(self._create_background(photons_in_background))
        output = np.sum(np.array(signals), axis=0) + background_noise
        if verbose:
            visualize(output)
            print(self)
            print(f"atom count: {atom_count}")
            print(f"photons per atom: {photons_from_atom}")
            print(f"photons_in_background: {photons_in_background}")
            print(f"x0: {x0}")
            print(f"y0: {y0}")
        return output.astype(np.float)

    def create_simulation_from_SNR(self, x0, y0, SNR, no_atom=False, verbose=False):
        if no_atom:
            atom_count = 0
        elif np.ndim(x0) == 1:
            x0 = np.array(x0)
            y0 = np.array(y0)
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







import numpy as np
from skimage.transform import radon

def simulate_measurements(phantom, theta, I0, radon_func=radon):
    """ Adds Beer-Lambert attenuation and Poisson noise (Eq. 1-3) """
    sino_clean = radon_func(phantom, theta=theta)
    
    if I0 is None or I0 == np.inf:
        return sino_clean
        
    # Scale sinogram to represent realistic attenuation (e.g., max 5.0) for Poisson physics
    sino_max = np.max(sino_clean)
    if sino_max == 0: sino_max = 1.0
    scale_factor = 5.0 / sino_max
    
    sino_scaled = sino_clean * scale_factor
    
    # Beer-Lambert Law & Poisson formulation
    I_d = I0 * np.exp(-sino_scaled)
    I_meas = np.random.poisson(I_d)
    I_meas = np.maximum(I_meas, 1) # Prevent log(0)
    
    sino_noisy_scaled = -np.log(I_meas / I0)
    
    # Rescale back to original mathematical range for iradon and metrics
    return sino_noisy_scaled / scale_factor

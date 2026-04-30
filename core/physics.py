import numpy as np
from skimage.transform import radon

def simulate_measurements(phantom, theta, I0, radon_func=radon):
    """
    Simulates the physical process of an X-ray CT scan, including:
    1. Radon Transform (Ideal line integrals)
    2. Beer-Lambert Attenuation
    3. Poisson Quantum Noise
    """
    # 1. Obtain the ideal analytical line integrals (the mathematical sinogram)
    # This represents the sum of attenuation coefficients along each ray
    sino_clean = radon_func(phantom, theta=theta)
    
    # If infinite dose is specified, return the perfect mathematical sinogram (Noiseless baseline)
    if I0 is None or I0 == np.inf:
        return sino_clean
        
    # 2. Rescale the sinogram for realistic attenuation
    # A human body attenuates X-rays significantly. We scale the max attenuation to 5.0
    # meaning the thickest part of the phantom will transmit exp(-5) ~ 0.6% of the photons.
    sino_max = np.max(sino_clean)
    if sino_max == 0: sino_max = 1.0 # Prevent division by zero
    scale_factor = 5.0 / sino_max
    
    # Apply the scaling to get physically realistic attenuation values
    sino_scaled = sino_clean * scale_factor
    
    # 3. Apply the Beer-Lambert Law: I = I0 * exp(-mu * x)
    # I_d is the expected (deterministic) number of photons reaching the detector
    I_d = I0 * np.exp(-sino_scaled)
    
    # 4. Apply Poisson Noise
    # X-ray detection is governed by Poisson statistics. 
    # I_meas is the actual, noisy count of discrete photons hitting the detector
    I_meas = np.random.poisson(I_d)
    
    # Prevent taking the log of 0 in the next step (if no photons reached the detector)
    I_meas = np.maximum(I_meas, 1) 
    
    # 5. Inverse Log-Transform
    # Convert the measured photon counts back into attenuation integrals
    sino_noisy_scaled = -np.log(I_meas / I0)
    
    # 6. Rescale back to the original mathematical range
    # This ensures our reconstructor and metric calculations operate on the original 0.0 - 1.0 scale
    return sino_noisy_scaled / scale_factor

import numpy as np
import logging
from skimage.transform import radon

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_measurements(phantom, theta, I0, radon_func=radon):
    """
    Simulates the physical process of an X-ray CT scan, including:
    1. Radon Transform (Ideal line integrals)
    2. Beer-Lambert Attenuation
    3. Poisson Quantum Noise
    """
    logger.info(f"Starting CT measurement simulation: I0={I0}, angles={len(theta)}")

    # 1. Obtain the ideal analytical line integrals (the mathematical sinogram)
    # This represents the sum of attenuation coefficients along each ray
    logger.debug("Step 1: Computing ideal Radon transform (sinogram)")
    sino_clean = radon_func(phantom, theta=theta)

    # If infinite dose is specified, return the perfect mathematical sinogram (Noiseless baseline)
    if I0 is None or I0 == np.inf:
        logger.info("Infinite dose specified - returning noiseless sinogram")
        return sino_clean

    # 2. Rescale the sinogram for realistic attenuation
    # A human body attenuates X-rays significantly. We scale the max attenuation to 5.0
    # meaning the thickest part of the phantom will transmit exp(-5) ~ 0.6% of the photons.
    logger.debug("Step 2: Rescaling sinogram for realistic attenuation")
    sino_max = np.max(sino_clean)
    if sino_max == 0:
        sino_max = 1.0  # Prevent division by zero
    scale_factor = 5.0 / sino_max
    logger.debug(f"Scale factor: {scale_factor:.4f}")

    # Apply the scaling to get physically realistic attenuation values
    sino_scaled = sino_clean * scale_factor

    # 3. Apply the Beer-Lambert Law: I = I0 * exp(-mu * x)
    # I_d is the expected (deterministic) number of photons reaching the detector
    logger.debug("Step 3: Applying Beer-Lambert law (exponential attenuation)")
    I_d = I0 * np.exp(-sino_scaled)
    logger.debug(f"Expected photon counts - min: {I_d.min():.2f}, max: {I_d.max():.2f}")

    # 4. Apply Poisson Noise
    # X-ray detection is governed by Poisson statistics.
    # I_meas is the actual, noisy count of discrete photons hitting the detector
    logger.debug("Step 4: Applying Poisson quantum noise")
    I_meas = np.random.poisson(I_d)
    logger.debug(f"Measured photons - min: {I_meas.min()}, max: {I_meas.max()}")

    # Prevent taking the log of 0 in the next step (if no photons reached the detector)
    I_meas = np.maximum(I_meas, 1)

    # 5. Inverse Log-Transform
    # Convert the measured photon counts back into attenuation integrals
    logger.debug("Step 5: Applying inverse log transform")
    sino_noisy_scaled = -np.log(I_meas / I0)

    # 6. Rescale back to the original mathematical range
    # This ensures our reconstructor and metric calculations operate on the original 0.0 - 1.0 scale
    logger.debug("Step 6: Rescaling back to original range")
    result = sino_noisy_scaled / scale_factor
    logger.info(f"Finished CT measurement simulation - output shape: {result.shape}")
    return result


def to_hounsfield_units(image, mu_water=0.206):
    """
    Converts generic attenuation coefficients into Hounsfield Units (HU).
    Air is -1000 HU, Water is 0 HU.
    """
    return 1000 * ((image - mu_water) / mu_water)

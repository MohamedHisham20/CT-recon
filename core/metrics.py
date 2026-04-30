import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import mean_squared_error

def calculate_rmse(image_true, image_test):
    """ Calculates the Root Mean Squared Error between the test image and ground truth. """
    return np.sqrt(mean_squared_error(image_true, image_test))

def calculate_psnr(image_true, image_test):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR).
    We enforce data_range=1.0 because the Shepp-Logan phantom mathematically ranges from 0 to 1.
    """
    return psnr_skimage(image_true, image_test, data_range=1.0)

def calculate_cnr(image, target_coords=(180, 128), bg_coords=(128, 128), roi_size=5):
    """
    Calculates Contrast-to-Noise Ratio (CNR).
    CNR measures how well a structure (target) stands out against its background noise.
    """
    r_t, c_t = target_coords
    r_b, c_b = bg_coords
    
    # Extract Region of Interest (ROI) patches for the target structure and the background
    target_roi = image[max(0, r_t-roi_size):r_t+roi_size, max(0, c_t-roi_size):c_t+roi_size]
    bg_roi = image[max(0, r_b-roi_size):r_b+roi_size, max(0, c_b-roi_size):c_b+roi_size]
    
    # Calculate mean signals
    s_target = np.mean(target_roi)
    s_bg = np.mean(bg_roi)
    
    # Calculate background noise (standard deviation)
    sigma_bg = np.std(bg_roi)
    
    if sigma_bg == 0:
        return 0.0 # Prevent division by zero if there's no noise
        
    # CNR Formula: |Signal_Target - Signal_Background| / Noise_Background
    return np.abs(s_target - s_bg) / sigma_bg

def get_1d_nps(image_true, image_test):
    """
    Computes the 1D radial Noise Power Spectrum (NPS).
    NPS shows how the noise variance is distributed across different spatial frequencies.
    """
    # 1. Isolate the pure noise
    noise = image_test - image_true
    
    # 2. Compute the 2D Fast Fourier Transform (FFT) and shift low frequencies to the center
    noise_fft = np.fft.fftshift(np.fft.fft2(noise))
    
    # 3. Calculate absolute spectral power
    # Normalized by the total number of pixels to reflect true absolute variance
    N = noise.size
    nps_2d = (np.abs(noise_fft)**2) / N
    
    # 4. Compute the 1D radial average (averaging concentric rings around the center)
    y, x = np.indices(nps_2d.shape)
    center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    
    # Calculate radial distance for every pixel
    r = np.hypot(x - center[1], y - center[0]).astype(int)
    
    # Accumulate power into discrete radial bins
    tbin = np.bincount(r.ravel(), nps_2d.ravel())
    nr = np.bincount(r.ravel()) # Count number of pixels in each bin
    
    # Divide total power in bin by number of pixels in bin (average)
    radial_profile = tbin / np.maximum(nr, 1)
    
    return radial_profile

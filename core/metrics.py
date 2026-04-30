import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import mean_squared_error

def calculate_rmse(image_true, image_test):
    return np.sqrt(mean_squared_error(image_true, image_test))

def calculate_psnr(image_true, image_test):
    # Enforcing data_range=1.0 since the phantom max possible theoretical value is 1.0 (Eq. 9)
    return psnr_skimage(image_true, image_test, data_range=1.0)

def calculate_cnr(image, target_coords=(180, 128), bg_coords=(128, 128), roi_size=5):
    """ Calculates Contrast-to-Noise Ratio (Eq. 10) """
    r_t, c_t = target_coords
    r_b, c_b = bg_coords
    
    target_roi = image[max(0, r_t-roi_size):r_t+roi_size, max(0, c_t-roi_size):c_t+roi_size]
    bg_roi = image[max(0, r_b-roi_size):r_b+roi_size, max(0, c_b-roi_size):c_b+roi_size]
    
    s_target = np.mean(target_roi)
    s_bg = np.mean(bg_roi)
    sigma_bg = np.std(bg_roi)
    
    if sigma_bg == 0:
        return 0.0
    return np.abs(s_target - s_bg) / sigma_bg

def get_1d_nps(image_true, image_test):
    """ Computes 1D radial Noise Power Spectrum (Eq. 11) """
    noise = image_test - image_true
    # 2D FFT
    noise_fft = np.fft.fftshift(np.fft.fft2(noise))
    
    # Normalized by the number of pixels to reflect true absolute spectral power
    N = noise.size
    nps_2d = (np.abs(noise_fft)**2) / N
    
    # Calculate 1D radial average
    y, x = np.indices(nps_2d.shape)
    center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(x - center[1], y - center[0]).astype(int)
    
    tbin = np.bincount(r.ravel(), nps_2d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    
    return radial_profile

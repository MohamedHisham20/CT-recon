import numpy as np
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import mean_squared_error
from scipy.ndimage import sobel

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_rmse(image_true, image_test):
    """Calculates the Root Mean Squared Error between the test image and ground truth."""
    rmse = np.sqrt(mean_squared_error(image_true, image_test))
    logger.debug(f"RMSE calculated: {rmse:.6f}")
    return rmse


def calculate_psnr(image_true, image_test):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR).
    We enforce data_range=1.0 because the Shepp-Logan phantom mathematically ranges from 0 to 1.
    """
    psnr = psnr_skimage(image_true, image_test, data_range=1.0)
    logger.debug(f"PSNR calculated: {psnr:.2f} dB")
    return psnr


def calculate_cnr(image, target_coords=(180, 128), bg_coords=(128, 128), roi_size=5):
    """
    Calculates Contrast-to-Noise Ratio (CNR).
    CNR measures how well a structure (target) stands out against its background noise.
    """
    logger.debug(
        f"Calculating CNR: target={target_coords}, bg={bg_coords}, roi={roi_size}"
    )
    r_t, c_t = target_coords
    r_b, c_b = bg_coords

    # Extract Region of Interest (ROI) patches for the target structure and the background
    target_roi = image[
        max(0, r_t - roi_size) : r_t + roi_size, max(0, c_t - roi_size) : c_t + roi_size
    ]
    bg_roi = image[
        max(0, r_b - roi_size) : r_b + roi_size, max(0, c_b - roi_size) : c_b + roi_size
    ]

    # Calculate mean signals
    s_target = np.mean(target_roi)
    s_bg = np.mean(bg_roi)

    # Calculate background noise (standard deviation)
    sigma_bg = np.std(bg_roi)

    if sigma_bg == 0:
        logger.warning("Background sigma is 0, returning CNR=0")
        return 0.0  # Prevent division by zero if there's no noise

    # CNR Formula: |Signal_Target - Signal_Background| / Noise_Background
    cnr = np.abs(s_target - s_bg) / sigma_bg
    logger.debug(
        f"CNR calculated: {cnr:.4f} (target_mean={s_target:.4f}, bg_mean={s_bg:.4f}, sigma={sigma_bg:.4f})"
    )
    return cnr


def get_1d_nps(image_true, image_test):
    """
    Computes the 1D radial Noise Power Spectrum (NPS).
    NPS shows how the noise variance is distributed across different spatial frequencies.
    """
    logger.debug("Computing 1D NPS")
    # 1. Isolate the pure noise
    noise = image_test - image_true

    # 2. Compute the 2D Fast Fourier Transform (FFT) and shift low frequencies to the center
    noise_fft = np.fft.fftshift(np.fft.fft2(noise))

    # 3. Calculate absolute spectral power
    # Normalized by the total number of pixels to reflect true absolute variance
    N = noise.size
    nps_2d = (np.abs(noise_fft) ** 2) / N

    # 4. Compute the 1D radial average (averaging concentric rings around the center)
    y, x = np.indices(nps_2d.shape)
    center = np.array([(y.max() - y.min()) / 2.0, (x.max() - x.min()) / 2.0])

    # Calculate radial distance for every pixel
    r = np.hypot(x - center[1], y - center[0]).astype(int)

    # Accumulate power into discrete radial bins
    tbin = np.bincount(r.ravel(), nps_2d.ravel())
    nr = np.bincount(r.ravel())  # Count number of pixels in each bin

    # Divide total power in bin by number of pixels in each bin (average)
    radial_profile = tbin / np.maximum(nr, 1)

    logger.debug(f"1D NPS computed: {len(radial_profile)} frequency bins")
    return radial_profile


def get_2d_nps(image_true, image_test):
    """
    Computes the 2D Noise Power Spectrum.
    Returns the absolute spectral power in frequency domain.
    """
    # Isolate pure noise
    noise = image_test - image_true

    # 2D FFT and absolute spectral power calculation
    noise_fft = np.fft.fftshift(np.fft.fft2(noise))
    nps_2d = (np.abs(noise_fft) ** 2) / noise.size

    return nps_2d


def calculate_epv(image_true, image_test):
    """
    Calculates Edge Preservation Value (EPV).
    Computes ratio of gradient magnitudes where 1.0 means perfect preservation.
    """
    grad_true_x = sobel(image_true, axis=0)
    grad_true_y = sobel(image_true, axis=1)
    grad_test_x = sobel(image_test, axis=0)
    grad_test_y = sobel(image_test, axis=1)

    grad_true_mag = np.hypot(grad_true_x, grad_true_y)
    grad_test_mag = np.hypot(grad_test_x, grad_test_y)

    sum_true = np.sum(grad_true_mag)
    sum_test = np.sum(grad_test_mag)

    if sum_true == 0:
        return 0.0
    return sum_test / sum_true

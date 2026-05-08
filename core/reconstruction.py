import numpy as np
import scipy.ndimage
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def custom_radon(image, theta):
    """
    Custom implementation of the Radon transform.
    Instead of tracing rays through a static image, we keep the rays vertical
    and rotate the entire image to simulate different projection angles.
    """
    logger.info(
        f"Starting custom Radon transform: image_size={image.shape}, angles={len(theta)}"
    )
    size = image.shape[0]
    sinogram = np.zeros((size, len(theta)))

    for i, angle in enumerate(theta):
        logger.debug(f"Processing angle {i + 1}/{len(theta)}: {angle}°")
        # Rotate image clockwise (negative angle) to simulate the scanner rotating counter-clockwise
        rotated = scipy.ndimage.rotate(image, -angle, reshape=False, order=1)

        # Project the image by summing along the vertical axis (axis=0)
        sinogram[:, i] = np.sum(rotated, axis=0)

    logger.info(f"Custom Radon complete: sinogram shape {sinogram.shape}")
    return sinogram


def custom_iradon(sinogram, theta, filter_name="hann"):
    """
    Custom implementation of Filtered Backprojection (FBP).
    Analytically reconstructs the image using the Fourier Slice Theorem.
    """
    logger.info(
        f"Starting custom FBP: sinogram={sinogram.shape}, filter={filter_name}, angles={len(theta)}"
    )
    n_det, n_angles = sinogram.shape

    # 1. Filtering in the frequency domain
    # Generate frequency bins
    logger.debug("Step 1: Creating frequency-domain filter")
    freqs = np.fft.fftfreq(n_det)

    # Create the Ramp Filter (|w|) to amplify high frequencies, compensating for the 1/r blurring of backprojection
    ramp_filter = 2 * np.abs(freqs)

    # Apply standard reconstruction filters to roll off high frequencies and reduce noise
    if filter_name == "hann":
        ramp_filter *= 0.5 + 0.5 * np.cos(2 * np.pi * freqs)
    elif filter_name == "shepp-logan":
        # np.sinc(x) is sin(pi*x)/(pi*x) in numpy
        ramp_filter *= np.sinc(2 * freqs)
    elif filter_name == "hamming":
        ramp_filter *= 0.54 + 0.46 * np.cos(2 * np.pi * freqs)
    elif filter_name == "cosine":
        ramp_filter *= np.cos(np.pi * freqs)

    logger.debug("Step 2: Applying filter in frequency domain")
    # Transform sinogram to frequency domain, apply filter, and inverse transform back
    sino_fft = np.fft.fft(sinogram, axis=0)
    filtered_sino = np.real(np.fft.ifft(sino_fft * ramp_filter[:, np.newaxis], axis=0))

    # 2. Backprojection
    logger.debug("Step 3: Performing backprojection")
    recon = np.zeros((n_det, n_det))

    # Create a coordinate grid centered at (0, 0)
    X, Y = np.meshgrid(np.arange(n_det) - n_det // 2, np.arange(n_det) - n_det // 2)

    for i, angle in enumerate(theta):
        if i % 20 == 0:
            logger.debug(f"Backprojecting angle {i + 1}/{n_angles}: {angle}°")
        angle_rad = np.deg2rad(angle)

        # Find the corresponding detector bin 't' for each pixel (X, Y) at this projection angle
        t = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        t += n_det // 2  # Shift center back to array indices

        # Filter out pixels that map outside the detector bounds
        valid = (t >= 0) & (t < n_det - 1)
        t_floor = np.floor(t[valid]).astype(int)
        t_frac = t[valid] - t_floor

        # Linear interpolation: take a weighted average of the two nearest detector bins
        recon[valid] += (1 - t_frac) * filtered_sino[
            t_floor, i
        ] + t_frac * filtered_sino[t_floor + 1, i]

    # Normalize by the number of angles and pi
    recon *= np.pi / (2 * n_angles)

    logger.debug("Step 4: Applying non-negativity constraint")
    # Physical constraint: density cannot be negative
    result = np.maximum(recon, 0)
    logger.info(
        f"Custom FBP complete: output range [{result.min():.4f}, {result.max():.4f}]"
    )
    return result


def custom_art(sinogram, theta, iterations=3, relaxation=1.0):
    """
    Custom Algebraic Reconstruction Technique (ART).
    Uses the exact same image rotation geometry as `custom_radon` to prevent mismatch artifacts.
    """
    logger.info(
        f"Starting custom ART: sinogram={sinogram.shape}, iterations={iterations}, relaxation={relaxation}"
    )
    size = sinogram.shape[0]
    recon = np.zeros((size, size))  # Start with a completely blank image

    # Pre-calculate the ray length passing through the image (geometric weight)
    # Rotating a square creates variable length rays at different angles!
    logger.debug("Pre-calculating geometric ray lengths")
    ones_img = np.ones((size, size))
    ray_lengths = {}
    for angle in theta:
        rotated_ones = scipy.ndimage.rotate(ones_img, -angle, reshape=False, order=1)
        ray_lengths[angle] = np.maximum(np.sum(rotated_ones, axis=0), 1.0)

    for it in range(iterations):
        logger.debug(f"ART iteration {it + 1}/{iterations}")
        for i, angle in enumerate(theta):
            # 1. Forward project our current guess at this angle
            rotated = scipy.ndimage.rotate(recon, -angle, reshape=False, order=1)
            proj = np.sum(rotated, axis=0)

            # 2. Calculate the error between the true measurement and our guess
            error = sinogram[:, i] - proj

            # 3. Backproject the error
            update = np.zeros((size, size))

            # Distribute error proportionally to the geometric ray lengths (corrects corner artifacts)
            update[:, :] = error[np.newaxis, :] / ray_lengths[angle][np.newaxis, :]

            # Rotate the correction back to the image frame of reference
            update_rotated_back = scipy.ndimage.rotate(
                update, angle, reshape=False, order=1
            )

            # Apply the correction to the reconstruction scaled by relaxation parameter
            recon += relaxation * update_rotated_back

            # 4. Enforce Non-negativity Constraint (improves stability and artifact reduction)
            recon = np.maximum(recon, 0)

    logger.info(
        f"Custom ART complete: output range [{recon.min():.4f}, {recon.max():.4f}]"
    )
    return recon


def tv_denoise_gd(image, lambda_tv, iterations=200, step_size=0.05, eps=1e-6):
    """
    Total Variation (TV) Denoising using Gradient Descent.
    Minimizes the TV norm to smooth noise while preserving sharp edges.
    """
    logger.info(
        f"Starting TV denoising: lambda={lambda_tv}, iterations={iterations}, step_size={step_size}"
    )
    u = image.copy()

    for it in range(iterations):
        # 1. Compute forward finite differences (gradients in x and y)
        u_x = np.roll(u, -1, axis=1) - u
        u_y = np.roll(u, -1, axis=0) - u

        # Enforce Neumann boundary conditions instead of Periodic to prevent edge wrap-around artifacts
        u_x[:, -1] = 0
        u_y[-1, :] = 0

        # Compute the gradient magnitude (norm). Add 'eps' to prevent division by zero.
        norm = np.sqrt(u_x**2 + u_y**2 + eps)

        # 2. Compute the divergence of the normalized gradient
        div_x = (u_x / norm) - np.roll(u_x / norm, 1, axis=1)
        div_y = (u_y / norm) - np.roll(u_y / norm, 1, axis=0)

        # Enforce Neumann backward boundaries
        div_x[:, 0] = (u_x / norm)[:, 0]
        div_y[0, :] = (u_y / norm)[0, :]
        div = div_x + div_y

        # 3. Gradient update step
        # The energy derivative is (u - f) - lambda * div.
        # We subtract the step_size * gradient to move downhill.
        gradient = (u - image) - lambda_tv * div
        u = u - step_size * gradient

        if it % 50 == 0:
            logger.debug(
                f"TV iteration {it}/{iterations}: gradient_norm={np.linalg.norm(gradient):.6f}"
            )

    logger.info(f"TV denoising complete: output range [{u.min():.4f}, {u.max():.4f}]")
    return u

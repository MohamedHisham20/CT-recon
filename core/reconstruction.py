import numpy as np
import scipy.ndimage

def custom_radon(image, theta):
    """ Custom implementation of the Radon transform using image rotation """
    size = image.shape[0]
    sinogram = np.zeros((size, len(theta)))
    for i, angle in enumerate(theta):
        # Rotate image to simulate projection angle
        rotated = scipy.ndimage.rotate(image, -angle, reshape=False, order=1)
        sinogram[:, i] = np.sum(rotated, axis=0)
    return sinogram

def custom_iradon(sinogram, theta, filter_name='hann'):
    """ Custom implementation of Filtered Backprojection (FBP) """
    n_det, n_angles = sinogram.shape
    
    # 1. Filtering in frequency domain
    freqs = np.fft.fftfreq(n_det)
    ramp_filter = 2 * np.abs(freqs)
    
    if filter_name == 'hann':
        ramp_filter *= np.hanning(n_det)
        
    sino_fft = np.fft.fft(sinogram, axis=0)
    filtered_sino = np.real(np.fft.ifft(sino_fft * ramp_filter[:, np.newaxis], axis=0))
    
    # 2. Backprojection
    recon = np.zeros((n_det, n_det))
    X, Y = np.meshgrid(np.arange(n_det) - n_det//2, np.arange(n_det) - n_det//2)
    
    for i, angle in enumerate(theta):
        angle_rad = np.deg2rad(angle)
        # Find corresponding detector bin for each pixel
        t = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        t += n_det // 2
        
        valid = (t >= 0) & (t < n_det - 1)
        t_floor = np.floor(t[valid]).astype(int)
        t_frac = t[valid] - t_floor
        
        # Linear interpolation
        recon[valid] += (1 - t_frac) * filtered_sino[t_floor, i] + t_frac * filtered_sino[t_floor + 1, i]
        
    recon *= (np.pi / (2 * n_angles))
    return np.maximum(recon, 0)

def custom_art(sinogram, theta, iterations=3):
    """
    Custom Algebraic Reconstruction Technique (ART)
    Geometrically matched to `custom_radon` via rotation.
    """
    size = sinogram.shape[0]
    recon = np.zeros((size, size))
    
    for _ in range(iterations):
        for i, angle in enumerate(theta):
            # Forward project current reconstruction at this angle
            rotated = scipy.ndimage.rotate(recon, -angle, reshape=False, order=1)
            proj = np.sum(rotated, axis=0)
            
            # Calculate error
            error = sinogram[:, i] - proj
            
            # Backproject the error
            update = np.zeros((size, size))
            # Distribute error evenly across the ray path
            update[:, :] = error[np.newaxis, :] / size
            
            # Rotate the update back to the image domain
            update_rotated_back = scipy.ndimage.rotate(update, angle, reshape=False, order=1)
            recon += update_rotated_back
            
            # Non-negativity constraint
            recon = np.maximum(recon, 0)
            
    return recon

def tv_denoise_gd(image, lambda_tv, iterations=200, step_size=0.05, eps=1e-6):
    """ Total Variation Denoising using Gradient Descent (Eq. 13-14) """
    u = image.copy()
    for _ in range(iterations):
        # Finite differences
        u_x = np.roll(u, -1, axis=1) - u
        u_y = np.roll(u, -1, axis=0) - u
        norm = np.sqrt(u_x**2 + u_y**2 + eps)
        
        # Divergence backwards differences (adjoint of forward differences)
        div_x = (u_x / norm) - np.roll(u_x / norm, 1, axis=1)
        div_y = (u_y / norm) - np.roll(u_y / norm, 1, axis=0)
        div = div_x + div_y
        
        # Gradient update step
        gradient = (u - image) - lambda_tv * div
        u = u - step_size * gradient
    return u

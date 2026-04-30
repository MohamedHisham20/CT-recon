import numpy as np
import scipy.ndimage

def custom_radon(image, theta):
    """
    Custom implementation of the Radon transform.
    Instead of tracing rays through a static image, we keep the rays vertical 
    and rotate the entire image to simulate different projection angles.
    """
    size = image.shape[0]
    sinogram = np.zeros((size, len(theta)))
    
    for i, angle in enumerate(theta):
        # Rotate image clockwise (negative angle) to simulate the scanner rotating counter-clockwise
        rotated = scipy.ndimage.rotate(image, -angle, reshape=False, order=1)
        
        # Project the image by summing along the vertical axis (axis=0)
        sinogram[:, i] = np.sum(rotated, axis=0)
        
    return sinogram

def custom_iradon(sinogram, theta, filter_name='hann'):
    """
    Custom implementation of Filtered Backprojection (FBP).
    Analytically reconstructs the image using the Fourier Slice Theorem.
    """
    n_det, n_angles = sinogram.shape
    
    # 1. Filtering in the frequency domain
    # Generate frequency bins
    freqs = np.fft.fftfreq(n_det)
    
    # Create the Ramp Filter (|w|) to amplify high frequencies, compensating for the 1/r blurring of backprojection
    ramp_filter = 2 * np.abs(freqs)
    
    # Apply a Hann window to roll off high frequencies and reduce noise amplification
    if filter_name == 'hann':
        ramp_filter *= np.hanning(n_det)
        
    # Transform sinogram to frequency domain, apply filter, and inverse transform back
    sino_fft = np.fft.fft(sinogram, axis=0)
    filtered_sino = np.real(np.fft.ifft(sino_fft * ramp_filter[:, np.newaxis], axis=0))
    
    # 2. Backprojection
    recon = np.zeros((n_det, n_det))
    
    # Create a coordinate grid centered at (0, 0)
    X, Y = np.meshgrid(np.arange(n_det) - n_det//2, np.arange(n_det) - n_det//2)
    
    for i, angle in enumerate(theta):
        angle_rad = np.deg2rad(angle)
        
        # Find the corresponding detector bin 't' for each pixel (X, Y) at this projection angle
        t = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        t += n_det // 2 # Shift center back to array indices
        
        # Filter out pixels that map outside the detector bounds
        valid = (t >= 0) & (t < n_det - 1)
        t_floor = np.floor(t[valid]).astype(int)
        t_frac = t[valid] - t_floor
        
        # Linear interpolation: take a weighted average of the two nearest detector bins
        recon[valid] += (1 - t_frac) * filtered_sino[t_floor, i] + t_frac * filtered_sino[t_floor + 1, i]
        
    # Normalize by the number of angles and pi
    recon *= (np.pi / (2 * n_angles))
    
    # Physical constraint: density cannot be negative
    return np.maximum(recon, 0)

def custom_art(sinogram, theta, iterations=3):
    """
    Custom Algebraic Reconstruction Technique (ART).
    Uses the exact same image rotation geometry as `custom_radon` to prevent mismatch artifacts.
    """
    size = sinogram.shape[0]
    recon = np.zeros((size, size)) # Start with a completely blank image
    
    for _ in range(iterations):
        for i, angle in enumerate(theta):
            # 1. Forward project our current guess at this angle
            rotated = scipy.ndimage.rotate(recon, -angle, reshape=False, order=1)
            proj = np.sum(rotated, axis=0)
            
            # 2. Calculate the error between the true measurement and our guess
            error = sinogram[:, i] - proj
            
            # 3. Backproject the error
            update = np.zeros((size, size))
            
            # Distribute the error uniformly across the entire length of the ray (size)
            update[:, :] = error[np.newaxis, :] / size
            
            # Rotate the correction back to the image frame of reference
            update_rotated_back = scipy.ndimage.rotate(update, angle, reshape=False, order=1)
            
            # Apply the correction to the reconstruction
            recon += update_rotated_back
            
            # 4. Enforce Non-negativity Constraint (improves stability and artifact reduction)
            recon = np.maximum(recon, 0)
            
    return recon

def tv_denoise_gd(image, lambda_tv, iterations=200, step_size=0.05, eps=1e-6):
    """
    Total Variation (TV) Denoising using Gradient Descent.
    Minimizes the TV norm to smooth noise while preserving sharp edges.
    """
    u = image.copy()
    
    for _ in range(iterations):
        # 1. Compute forward finite differences (gradients in x and y)
        u_x = np.roll(u, -1, axis=1) - u
        u_y = np.roll(u, -1, axis=0) - u
        
        # Compute the gradient magnitude (norm). Add 'eps' to prevent division by zero.
        norm = np.sqrt(u_x**2 + u_y**2 + eps)
        
        # 2. Compute the divergence of the normalized gradient
        # We use backwards differences here (p_i - p_{i-1}) because it is the mathematical adjoint of the forward difference.
        div_x = (u_x / norm) - np.roll(u_x / norm, 1, axis=1)
        div_y = (u_y / norm) - np.roll(u_y / norm, 1, axis=0)
        div = div_x + div_y
        
        # 3. Gradient update step
        # The energy derivative is (u - f) - lambda * div.
        # We subtract the step_size * gradient to move downhill.
        gradient = (u - image) - lambda_tv * div
        u = u - step_size * gradient
        
    return u

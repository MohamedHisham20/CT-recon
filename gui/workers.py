import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from skimage.transform import radon, iradon, iradon_sart

from core.physics import simulate_measurements
from core.reconstruction import custom_radon, custom_iradon, custom_art, tv_denoise_gd
from core.metrics import calculate_psnr, calculate_rmse, calculate_cnr, get_1d_nps

class WorkerScenario1(QThread):
    """
    Worker Thread for Scenario 1: Dose vs Noise Trade-off.
    This runs in the background so the GUI doesn't freeze during intensive computations.
    """
    # Signal emitted when the simulation is finished, carrying the results payload
    finished = pyqtSignal(dict)
    
    def __init__(self, phantom, use_custom=False):
        super().__init__()
        self.phantom = phantom
        self.use_custom = use_custom
        
    def run(self):
        # Determine whether to use the custom implementation or the fast skimage implementation
        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon
        
        # We use a very dense angular sampling (720 angles over 180 degrees) for the baseline
        theta = np.linspace(0.0, 180.0, 720, endpoint=False)
        
        # Test a range of incident photon doses (I0) from very low to high
        doses = [10**2, 10**3, 10**4, 10**5, 10**6]
        
        # Generate the ground truth noiseless baseline (I0 = infinity)
        sino_clean = simulate_measurements(self.phantom, theta, np.inf, radon_fn)
        recon_clean = iradon_fn(sino_clean, theta=theta) # Default uses Hann filter
        
        # Lists to hold the evaluation metrics for each dose
        psnr_results, rmse_results, reconstructions = [], [], []
        nps_profiles = []
        
        # Simulate and reconstruct for each dose level
        for I0 in doses:
            # Simulate physics: add Beer-Lambert attenuation and Poisson quantum noise
            sino_noisy = simulate_measurements(self.phantom, theta, I0, radon_fn)
            
            # Reconstruct the noisy sinogram using FBP
            recon_noisy = iradon_fn(sino_noisy, theta=theta)
            
            # Save the image and calculate metrics against the noiseless baseline
            reconstructions.append(recon_noisy)
            psnr_results.append(calculate_psnr(recon_clean, recon_noisy))
            rmse_results.append(calculate_rmse(recon_clean, recon_noisy))
            
            # Calculate the 1D Noise Power Spectrum to observe the frequency distribution of the noise
            nps_profiles.append(get_1d_nps(recon_clean, recon_noisy))
            
        # Emit all calculated data back to the main GUI thread to be plotted
        self.finished.emit({
            'doses': doses, 'recon_clean': recon_clean, 
            'reconstructions': reconstructions, 'psnr': psnr_results, 
            'rmse': rmse_results, 'nps_profiles': nps_profiles
        })


class WorkerScenario2(QThread):
    """
    Worker Thread for Scenario 2: Iterative Total Variation (TV) Denoising.
    Demonstrates how adding a TV penalty term can suppress noise while preserving edges.
    """
    finished = pyqtSignal(dict)
    
    def __init__(self, phantom, use_custom=False):
        super().__init__()
        self.phantom = phantom
        self.use_custom = use_custom
        
    def run(self):
        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon
        
        # Dense sampling baseline
        theta = np.linspace(0.0, 180.0, 720, endpoint=False)
        I0_low = 10**3 # Use a relatively low dose to ensure visible noise
        
        # We will test different strengths of the TV regularization parameter (lambda)
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]
        
        # Ground truth clean image (infinite dose)
        sino_clean = simulate_measurements(self.phantom, theta, np.inf, radon_fn)
        recon_clean = iradon_fn(sino_clean, theta=theta)
        
        # Low dose simulation
        sino_noisy = simulate_measurements(self.phantom, theta, I0_low, radon_fn)
        
        # Standard FBP reconstruction of the noisy data (our noisy baseline)
        recon_fbp = iradon_fn(sino_noisy, theta=theta)
        
        # Calculate baseline metrics for FBP before any denoising is applied
        baseline_psnr = calculate_psnr(recon_clean, recon_fbp)
        baseline_cnr = calculate_cnr(recon_fbp) # Contrast-to-Noise Ratio
        
        tv_reconstructions, psnr_results, cnr_results = [], [], []
        
        # Apply TV gradient descent denoising for each lambda value
        for lda in lambdas:
            # We use 200 iterations of gradient descent.
            # Higher lambda enforces stronger piecewise constancy (flatter regions, sharper edges)
            recon_tv = tv_denoise_gd(recon_fbp, lambda_tv=lda, iterations=200, step_size=0.05)
            
            # Record the denoised image and its quality metrics
            tv_reconstructions.append(recon_tv)
            psnr_results.append(calculate_psnr(recon_clean, recon_tv))
            cnr_results.append(calculate_cnr(recon_tv))
            
        # Emit results to GUI
        self.finished.emit({
            'lambdas': lambdas, 'recon_fbp': recon_fbp, 
            'baseline_psnr': baseline_psnr, 'baseline_cnr': baseline_cnr,
            'tv_reconstructions': tv_reconstructions, 
            'psnr': psnr_results, 'cnr': cnr_results
        })


class WorkerScenario3(QThread):
    """
    Worker Thread for Scenario 3: Sparse Angular Sampling and ART.
    Compares traditional analytical reconstruction (FBP) against iterative algebraic techniques (ART)
    under conditions where very few projection angles are measured.
    """
    finished = pyqtSignal(dict)
    
    def __init__(self, phantom, use_custom=False):
        super().__init__()
        self.phantom = phantom
        self.use_custom = use_custom
        
    def run(self):
        # Choose which mathematical library to use
        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon
        
        # Define experiment parameters
        # We use a moderate photon count so noise isn't the primary issue, only sparse data
        I0_mod = 10**5
        
        # Progressively reduce the number of projection angles (Sparse Sampling)
        angles_list = [180, 120, 90, 60, 45, 36, 24, 18]  
        
        # We only want to plot these specific subset of angles to avoid overcrowding the UI
        viz_angles = [180, 90, 45, 18]
        
        # Create a "perfect" ground truth reference using 720 dense angles and infinite dose
        theta_full = np.linspace(0.0, 180.0, 720, endpoint=False)
        sino_full = simulate_measurements(self.phantom, theta_full, np.inf, radon_fn)
        recon_full = iradon_fn(sino_full, theta=theta_full)
        
        psnr_fbp, psnr_art = [], []
        recons_fbp, recons_art = [], []
        
        # Iterate over our decreasing list of available angles
        for n_angles in angles_list:
            # Generate the sparse angles array
            theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
            
            # Simulate the scan with the reduced number of angles
            sino_sparse = simulate_measurements(self.phantom, theta, I0_mod, radon_fn)
            
            # 1. Reconstruct using traditional FBP
            # FBP will exhibit severe streak artifacts when the angles are sparse
            recon_fbp_sparse = iradon_fn(sino_sparse, theta=theta)
            psnr_fbp.append(calculate_psnr(recon_full, recon_fbp_sparse))
            
            # 2. Reconstruct using Iterative ART
            # ART solves the system Ax = b iteratively and can enforce non-negativity constraints,
            # allowing it to dramatically reduce streak artifacts compared to FBP.
            if self.use_custom:
                # If custom math is selected, we MUST use our custom_art.
                # It shares the exact same rotation-based forward projection geometric model as custom_radon.
                recon_art_sparse = custom_art(sino_sparse, theta=theta, iterations=3)
            else:
                # If using standard skimage, we use SART (Simultaneous ART) for robust convergence.
                # We seed the first iteration, then feed the result back in for 3 total passes.
                recon_art_sparse = iradon_sart(sino_sparse, theta=theta)
                for _ in range(3):
                    recon_art_sparse = iradon_sart(sino_sparse, theta=theta, image=recon_art_sparse)
                    
            psnr_art.append(calculate_psnr(recon_full, recon_art_sparse))
            
            # If the current angle count is in our visualization list, save the images
            if n_angles in viz_angles:
                recons_fbp.append(recon_fbp_sparse)
                recons_art.append(recon_art_sparse)
                
        # Send all data to the UI thread
        self.finished.emit({
            'angles_list': angles_list, 'viz_angles': viz_angles,
            'psnr_fbp': psnr_fbp, 'psnr_art': psnr_art,
            'recons_fbp': recons_fbp, 'recons_art': recons_art
        })

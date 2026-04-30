import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from skimage.transform import radon, iradon, iradon_sart

from core.physics import simulate_measurements
from core.reconstruction import custom_radon, custom_iradon, custom_art, tv_denoise_gd
from core.metrics import calculate_psnr, calculate_rmse, calculate_cnr, get_1d_nps

class WorkerScenario1(QThread):
    finished = pyqtSignal(dict)
    
    def __init__(self, phantom, use_custom=False):
        super().__init__()
        self.phantom = phantom
        self.use_custom = use_custom
        
    def run(self):
        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon
        
        theta = np.linspace(0.0, 180.0, 720, endpoint=False)
        doses = [10**2, 10**3, 10**4, 10**5, 10**6]
        
        sino_clean = simulate_measurements(self.phantom, theta, np.inf, radon_fn)
        recon_clean = iradon_fn(sino_clean, theta=theta) # Default hann
        
        psnr_results, rmse_results, reconstructions = [], [], []
        nps_profiles = []
        
        for I0 in doses:
            sino_noisy = simulate_measurements(self.phantom, theta, I0, radon_fn)
            recon_noisy = iradon_fn(sino_noisy, theta=theta)
            
            reconstructions.append(recon_noisy)
            psnr_results.append(calculate_psnr(recon_clean, recon_noisy))
            rmse_results.append(calculate_rmse(recon_clean, recon_noisy))
            nps_profiles.append(get_1d_nps(recon_clean, recon_noisy))
            
        self.finished.emit({
            'doses': doses, 'recon_clean': recon_clean, 
            'reconstructions': reconstructions, 'psnr': psnr_results, 
            'rmse': rmse_results, 'nps_profiles': nps_profiles
        })


class WorkerScenario2(QThread):
    finished = pyqtSignal(dict)
    
    def __init__(self, phantom, use_custom=False):
        super().__init__()
        self.phantom = phantom
        self.use_custom = use_custom
        
    def run(self):
        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon
        
        theta = np.linspace(0.0, 180.0, 720, endpoint=False)
        I0_low = 10**3
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]
        
        sino_clean = simulate_measurements(self.phantom, theta, np.inf, radon_fn)
        recon_clean = iradon_fn(sino_clean, theta=theta)
        
        sino_noisy = simulate_measurements(self.phantom, theta, I0_low, radon_fn)
        recon_fbp = iradon_fn(sino_noisy, theta=theta)
        
        baseline_psnr = calculate_psnr(recon_clean, recon_fbp)
        baseline_cnr = calculate_cnr(recon_fbp)
        
        tv_reconstructions, psnr_results, cnr_results = [], [], []
        for lda in lambdas:
            recon_tv = tv_denoise_gd(recon_fbp, lambda_tv=lda, iterations=200, step_size=0.05)
            tv_reconstructions.append(recon_tv)
            psnr_results.append(calculate_psnr(recon_clean, recon_tv))
            cnr_results.append(calculate_cnr(recon_tv))
            
        self.finished.emit({
            'lambdas': lambdas, 'recon_fbp': recon_fbp, 
            'baseline_psnr': baseline_psnr, 'baseline_cnr': baseline_cnr,
            'tv_reconstructions': tv_reconstructions, 
            'psnr': psnr_results, 'cnr': cnr_results
        })


class WorkerScenario3(QThread):
    finished = pyqtSignal(dict)
    
    def __init__(self, phantom, use_custom=False):
        super().__init__()
        self.phantom = phantom
        self.use_custom = use_custom
        
    def run(self):
        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon
        
        angles_list = [180, 120, 90, 60, 45, 36, 24, 18]
        I0_mod = 10**5
        viz_angles = [180, 90, 45, 18]
        
        theta_full = np.linspace(0.0, 180.0, 720, endpoint=False)
        sino_full = simulate_measurements(self.phantom, theta_full, np.inf, radon_fn)
        recon_full = iradon_fn(sino_full, theta=theta_full)
        
        psnr_fbp, psnr_art = [], []
        recons_fbp, recons_art = [], []
        
        for n_angles in angles_list:
            theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
            sino_sparse = simulate_measurements(self.phantom, theta, I0_mod, radon_fn)
            
            # FBP
            recon_fbp_sparse = iradon_fn(sino_sparse, theta=theta)
            psnr_fbp.append(calculate_psnr(recon_full, recon_fbp_sparse))
            
            # Iterative ART
            if self.use_custom:
                # Use custom ART which shares the same forward projection geometric model
                recon_art_sparse = custom_art(sino_sparse, theta=theta, iterations=3)
            else:
                # Use SART via Skimage for stability when using skimage.radon
                recon_art_sparse = iradon_sart(sino_sparse, theta=theta)
                for _ in range(3):
                    recon_art_sparse = iradon_sart(sino_sparse, theta=theta, image=recon_art_sparse)
                    
            psnr_art.append(calculate_psnr(recon_full, recon_art_sparse))
            
            if n_angles in viz_angles:
                recons_fbp.append(recon_fbp_sparse)
                recons_art.append(recon_art_sparse)
                
        self.finished.emit({
            'angles_list': angles_list, 'viz_angles': viz_angles,
            'psnr_fbp': psnr_fbp, 'psnr_art': psnr_art,
            'recons_fbp': recons_fbp, 'recons_art': recons_art
        })

"""
Project: CT Simulation - Dose, Reconstruction & Sparse Sampling (GUI Version)
Requirements:
    pip install numpy scipy matplotlib scikit-image PyQt5
"""

import sys
import numpy as np
import scipy.ndimage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                             QScrollArea, QProgressBar, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import mean_squared_error

# ==========================================
# 1. Core Mathematical Functions & Metrics
# ==========================================

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
    nps_2d = np.abs(noise_fft)**2
    
    # Calculate 1D radial average
    y, x = np.indices(nps_2d.shape)
    center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(x - center[1], y - center[0]).astype(int)
    
    tbin = np.bincount(r.ravel(), nps_2d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    
    return radial_profile

# ==========================================
# 2. Custom Forward / Backprojection (Section 3.1)
# ==========================================

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

# ==========================================
# 3. Physics & Denoising Simulators
# ==========================================

def simulate_measurements(phantom, theta, I0, radon_func=radon):
    """ Adds Beer-Lambert attenuation and Poisson noise (Eq. 1-3) """
    sino_clean = radon_func(phantom, theta=theta)
    
    if I0 is None or I0 == np.inf:
        return sino_clean
        
    # Scale sinogram to represent realistic attenuation (e.g., max 5.0) for Poisson physics
    sino_max = np.max(sino_clean)
    if sino_max == 0: sino_max = 1.0
    scale_factor = 5.0 / sino_max
    
    sino_scaled = sino_clean * scale_factor
    
    # Beer-Lambert Law & Poisson formulation
    I_d = I0 * np.exp(-sino_scaled)
    I_meas = np.random.poisson(I_d)
    I_meas = np.maximum(I_meas, 1) # Prevent log(0)
    
    sino_noisy_scaled = -np.log(I_meas / I0)
    
    # Rescale back to original mathematical range for iradon and metrics
    return sino_noisy_scaled / scale_factor

def tv_denoise_gd(image, lambda_tv, iterations=200, step_size=0.05, eps=1e-6):
    """ Total Variation Denoising using Gradient Descent (Eq. 13-14) """
    u = image.copy()
    for _ in range(iterations):
        # Finite differences
        u_x = np.roll(u, -1, axis=1) - u
        u_y = np.roll(u, -1, axis=0) - u
        norm = np.sqrt(u_x**2 + u_y**2 + eps)
        
        div_x = (u_x / norm) - np.roll(u_x / norm, 1, axis=1)
        div_y = (u_y / norm) - np.roll(u_y / norm, 1, axis=0)
        div = div_x + div_y
        
        # Gradient update step
        gradient = (u - image) - lambda_tv * div
        u = u - step_size * gradient
    return u

# ==========================================
# 4. Qt Workers for Background Processing
# ==========================================

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
            
            # Iterative ART (Using SART via Skimage for stability, proposal mentions ART)
            # Custom unoptimized ART loop in pure python is impractically slow for UI
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

# ==========================================
# 5. GUI Components
# ==========================================

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

class BaseScenarioTab(QWidget):
    def __init__(self, phantom, parent=None):
        super().__init__(parent)
        self.phantom = phantom
        self.layout = QVBoxLayout(self)
        
        # Controls
        self.control_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Scenario")
        self.run_btn.clicked.connect(self.run_scenario)
        
        self.custom_math_cb = QCheckBox("Use Custom Math (Strict Proposal, Slower!)")
        self.custom_math_cb.setChecked(False)
        
        self.status_label = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        
        self.control_layout.addWidget(self.run_btn)
        self.control_layout.addWidget(self.custom_math_cb)
        self.control_layout.addWidget(self.status_label)
        self.control_layout.addWidget(self.progress_bar)
        self.control_layout.addStretch()
        
        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        
        self.canvas_img = MplCanvas(self, width=10, height=7)
        self.canvas_plot = MplCanvas(self, width=10, height=4)
        
        self.scroll_layout.addWidget(self.canvas_img)
        self.scroll_layout.addWidget(self.canvas_plot)
        self.scroll.setWidget(self.scroll_content)
        
        self.layout.addLayout(self.control_layout)
        self.layout.addWidget(self.scroll)

    def set_running_state(self, is_running):
        self.run_btn.setEnabled(not is_running)
        self.custom_math_cb.setEnabled(not is_running)
        self.progress_bar.setVisible(is_running)
        msg = "Running... Custom Math takes ~30s." if self.custom_math_cb.isChecked() else "Running (Skimage Fast Mode)..."
        self.status_label.setText(msg if is_running else "Simulation Complete.")

    def run_scenario(self):
        pass

class TabScenario1(BaseScenarioTab):
    def run_scenario(self):
        self.set_running_state(True)
        self.worker = WorkerScenario1(self.phantom, self.custom_math_cb.isChecked())
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, data):
        self.set_running_state(False)
        
        # Plot Images
        self.canvas_img.fig.clear()
        axs = self.canvas_img.fig.subplots(2, 3)
        self.canvas_img.fig.suptitle('Scenario 1: Dose vs. Noise Trade-off (FBP with Hann Filter)', fontsize=14)
        
        axs[0, 0].imshow(data['recon_clean'], cmap='gray')
        axs[0, 0].set_title("Noise-Free Baseline")
        axs[0, 0].axis('off')
        
        for i, I0 in enumerate(data['doses']):
            ax = axs[(i+1)//3, (i+1)%3]
            ax.imshow(data['reconstructions'][i], cmap='gray')
            ax.set_title(f"$I_0 = 10^{int(np.log10(I0))}$ | PSNR: {data['psnr'][i]:.1f}")
            ax.axis('off')
            
        self.canvas_img.fig.tight_layout()
        self.canvas_img.draw()
        
        # Plot Metrics (PSNR, RMSE, NPS)
        self.canvas_plot.fig.clear()
        ax1, ax_nps = self.canvas_plot.fig.subplots(1, 2)
        
        x_vals = np.log10(data['doses'])
        ax2 = ax1.twinx()
        ax1.plot(x_vals, data['psnr'], 'g-o', label='PSNR (dB)')
        ax2.plot(x_vals, data['rmse'], 'b-s', label='RMSE')
        
        ax1.set_xlabel('Log10(Incident Photons $I_0$)')
        ax1.set_ylabel('PSNR (dB)', color='g')
        ax2.set_ylabel('RMSE', color='b')
        ax1.set_title('Quality Metrics vs Photon Dose')
        ax1.grid(True)
        
        # NPS Plot (Plotting highest noise scenario)
        freq_bins = np.arange(len(data['nps_profiles'][0]))
        ax_nps.plot(freq_bins, data['nps_profiles'][0], 'r-', label=f'$I_0=10^{int(np.log10(data["doses"][0]))}$')
        ax_nps.plot(freq_bins, data['nps_profiles'][-1], 'k-', label=f'$I_0=10^{int(np.log10(data["doses"][-1]))}$')
        ax_nps.set_xlabel('Spatial Frequency (bins)')
        ax_nps.set_ylabel('Noise Power')
        ax_nps.set_yscale('log')
        ax_nps.set_title('1D Noise Power Spectrum (NPS)')
        ax_nps.legend()
        ax_nps.grid(True)
        
        self.canvas_plot.fig.tight_layout()
        self.canvas_plot.draw()

class TabScenario2(BaseScenarioTab):
    def run_scenario(self):
        self.set_running_state(True)
        self.worker = WorkerScenario2(self.phantom, self.custom_math_cb.isChecked())
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, data):
        self.set_running_state(False)
        
        # Plot Images
        self.canvas_img.fig.clear()
        axs = self.canvas_img.fig.subplots(2, 3)
        self.canvas_img.fig.suptitle(f'Scenario 2: TV Denoising on Low-Dose FBP ($I_0=10^3$)', fontsize=14)
        
        axs[0, 0].imshow(data['recon_fbp'], cmap='gray')
        axs[0, 0].set_title(f"Baseline FBP | PSNR: {data['baseline_psnr']:.1f}")
        axs[0, 0].axis('off')
        
        for i, lda in enumerate(data['lambdas']):
            ax = axs[(i+1)//3, (i+1)%3]
            ax.imshow(data['tv_reconstructions'][i], cmap='gray')
            ax.set_title(f"TV ($\lambda={lda}$) | CNR: {data['cnr'][i]:.2f}")
            ax.axis('off')
            
        self.canvas_img.fig.tight_layout()
        self.canvas_img.draw()
        
        # Plot Metrics (PSNR & CNR)
        self.canvas_plot.fig.clear()
        ax1 = self.canvas_plot.fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        ax1.plot(data['lambdas'], data['psnr'], 'r-o', label='PSNR')
        ax1.axhline(y=data['baseline_psnr'], color='r', linestyle='--', alpha=0.5)
        
        ax2.plot(data['lambdas'], data['cnr'], 'm-s', label='CNR')
        ax2.axhline(y=data['baseline_cnr'], color='m', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('TV Regularization Parameter ($\lambda$)')
        ax1.set_ylabel('PSNR (dB)', color='r')
        ax2.set_ylabel('Contrast-to-Noise Ratio (CNR)', color='m')
        ax1.set_title('Impact of $\lambda$ on PSNR and CNR')
        ax1.grid(True)
        
        self.canvas_plot.fig.tight_layout()
        self.canvas_plot.draw()

class TabScenario3(BaseScenarioTab):
    def run_scenario(self):
        self.set_running_state(True)
        self.worker = WorkerScenario3(self.phantom, self.custom_math_cb.isChecked())
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, data):
        self.set_running_state(False)
        
        # Plot Images
        self.canvas_img.fig.clear()
        viz_angles = data['viz_angles']
        axs = self.canvas_img.fig.subplots(len(viz_angles), 2)
        self.canvas_img.fig.suptitle('Scenario 3: FBP vs ART under Sparse Sampling', fontsize=14)
        
        # Dynamic canvas size mapping & Minimum height enforcement for Scroll Area
        self.canvas_img.fig.set_figheight(4 * len(viz_angles))
        self.canvas_img.setMinimumHeight(350 * len(viz_angles))
        
        for i, n_angles in enumerate(viz_angles):
            axs[i, 0].imshow(data['recons_fbp'][i], cmap='gray')
            axs[i, 0].set_title(f"FBP ({n_angles} angles)")
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(data['recons_art'][i], cmap='gray')
            axs[i, 1].set_title(f"ART/SART ({n_angles} angles)")
            axs[i, 1].axis('off')
            
        self.canvas_img.fig.tight_layout()
        self.canvas_img.draw()
        
        # Plot Metrics
        self.canvas_plot.fig.clear()
        ax = self.canvas_plot.fig.add_subplot(111)
        ax.plot(data['angles_list'], data['psnr_fbp'], 'm-o', label='FBP PSNR')
        ax.plot(data['angles_list'], data['psnr_art'], 'c-s', label='ART PSNR')
        ax.set_xlabel('Number of Projection Angles')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('FBP vs Iterative ART Across Angular Sampling Densities')
        ax.legend()
        ax.grid(True)
        ax.invert_xaxis()
        
        self.canvas_plot.fig.tight_layout()
        self.canvas_plot.draw()

class CTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Simulation & Reconstruction Dashboard")
        self.resize(1100, 800)
        
        # Initialize Phantom
        raw_phantom = shepp_logan_phantom()
        self.phantom = resize(raw_phantom, (256, 256), mode='reflect', anti_aliasing=True)
        
        # Tabs Setup
        self.tabs = QTabWidget()
        self.tab1 = TabScenario1(self.phantom)
        self.tab2 = TabScenario2(self.phantom)
        self.tab3 = TabScenario3(self.phantom)
        
        self.tabs.addTab(self.tab1, "1: Dose vs Noise (NPS)")
        self.tabs.addTab(self.tab2, "2: FBP vs TV Denoising (CNR)")
        self.tabs.addTab(self.tab3, "3: Sparse Angular Sampling (ART)")
        
        self.setCentralWidget(self.tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CTMainWindow()
    window.show()
    sys.exit(app.exec_())
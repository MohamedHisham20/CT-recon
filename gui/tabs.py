import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QScrollArea, QProgressBar, QCheckBox)
from gui.components import MplCanvas
from gui.workers import WorkerScenario1, WorkerScenario2, WorkerScenario3

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
        
        self.canvas_img.setMinimumHeight(500)
        self.canvas_plot.setMinimumHeight(350)
        
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
            ax.set_title(rf"TV ($\lambda={lda}$) | CNR: {data['cnr'][i]:.2f}")
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
        
        ax1.set_xlabel(r'TV Regularization Parameter ($\lambda$)')
        ax1.set_ylabel('PSNR (dB)', color='r')
        ax2.set_ylabel('Contrast-to-Noise Ratio (CNR)', color='m')
        ax1.set_title(r'Impact of $\lambda$ on PSNR and CNR')
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

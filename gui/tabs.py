import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QScrollArea, QProgressBar, QCheckBox, QFrame,
                             QComboBox, QSpinBox, QDoubleSpinBox)
import qtawesome as qta
from PyQt5.QtCore import Qt
from gui.components import MplCanvas
from gui.workers import WorkerScenario1, WorkerScenario2, WorkerScenario3

class BaseScenarioTab(QWidget):
    def __init__(self, phantom, parent=None):
        super().__init__(parent)
        self.phantom = phantom
        self.layout = QHBoxLayout(self) # Changed from QVBoxLayout -> Main Split (Left & Right)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        
        # --- Left Side Panel ---
        self.side_panel = QFrame()
        self.side_panel.setObjectName("SidePanel")
        self.side_panel.setFixedWidth(280)
        self.side_layout = QVBoxLayout(self.side_panel)
        self.side_layout.setContentsMargins(15, 15, 15, 15)
        self.side_layout.setSpacing(15)
        self.side_panel.setProperty("class", "sidebar")
        
        # Generator / Title Icon
        lbl_icon = QLabel()
        lbl_icon.setPixmap(qta.icon('fa5s.microchip', color='#63B3ED').pixmap(48, 48))
        lbl_icon.setAlignment(Qt.AlignCenter)
        self.side_layout.addWidget(lbl_icon)
        
        lbl_title = QLabel("Evaluation Settings")
        lbl_title.setAlignment(Qt.AlignCenter)
        font = lbl_title.font()
        font.setPointSize(12)
        font.setBold(True)
        lbl_title.setFont(font)
        self.side_layout.addWidget(lbl_title)
        
        # Add visual separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #4A5568;")
        self.side_layout.addWidget(line)
        
        # Hyperparameters
        self.params_layout = QVBoxLayout()
        self.add_hyperparameters()
        self.side_layout.addLayout(self.params_layout)
        
        # Add visual separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #4A5568;")
        self.side_layout.addWidget(line2)

        # Controls
        self.custom_math_cb = QCheckBox("Use Custom Math\n(Strict Proposal ~30s)")
        self.custom_math_cb.setChecked(False)
        self.side_layout.addWidget(self.custom_math_cb)
        
        self.run_btn = QPushButton(" Execute Scenario")
        self.run_btn.setIcon(qta.icon('fa5s.play', color='white'))
        self.run_btn.clicked.connect(self.run_scenario)
        self.side_layout.addWidget(self.run_btn)
        
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: #A0AEC0;")
        self.status_label.setWordWrap(True)
        self.side_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.side_layout.addWidget(self.progress_bar)
        
        self.side_layout.addStretch() # Push everything up
        
        # --- Right Side (Main Visualizations) ---
        self.main_content = QWidget()
        self.main_layout = QVBoxLayout(self.main_content)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll Area for right side content
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_inner_layout = QVBoxLayout(self.scroll_content)
        
        # Setup canvases without borders for clean look inside right pane
        self.canvas_img = MplCanvas(self, width=10, height=7)
        self.canvas_plot = MplCanvas(self, width=10, height=4)
        
        self.canvas_img.setMinimumHeight(450)
        self.canvas_plot.setMinimumHeight(350)
        
        self.scroll_inner_layout.addWidget(self.canvas_img)
        self.scroll_inner_layout.addWidget(self.canvas_plot)
        self.scroll.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll)
        
        # Assemble
        self.layout.addWidget(self.side_panel)
        self.layout.addWidget(self.main_content)

    def set_running_state(self, is_running):
        self.run_btn.setEnabled(not is_running)
        self.custom_math_cb.setEnabled(not is_running)
        self.progress_bar.setVisible(is_running)
        msg = "Running... Custom Math takes ~30s." if self.custom_math_cb.isChecked() else "Running (Skimage Fast Mode)..."
        self.status_label.setText(msg if is_running else "Simulation Complete.")

    def add_hyperparameters(self):
        pass # Implemented by child classes

    def run_scenario(self):
        pass


class TabScenario1(BaseScenarioTab):
    def add_hyperparameters(self):
        self.params_layout.addWidget(QLabel("FBP Filter:"))
        self.filter_combobox = QComboBox()
        self.filter_combobox.addItems(["ramp", "shepp-logan", "hann", "hamming"])
        self.filter_combobox.setCurrentText("hann")
        self.params_layout.addWidget(self.filter_combobox)
        
        desc1 = QLabel("Controls noise vs. sharpness tradeoff. 'ramp' is sharpest but noisiest, 'hann' smoothly suppresses high-frequency noise.")
        desc1.setStyleSheet("color: #718096; font-size: 11px; margin-bottom: 10px;")
        desc1.setWordWrap(True)
        self.params_layout.addWidget(desc1)

    def run_scenario(self):
        self.set_running_state(True)
        self.worker = WorkerScenario1(self.phantom, self.custom_math_cb.isChecked())
        self.worker.filter_name = self.filter_combobox.currentText()
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
        ax1, ax_nps, ax_theo = self.canvas_plot.fig.subplots(1, 3)
        
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

        # Theoretical vs Measured Noise StdDev Plot
        ax_theo.plot(x_vals, data['expected_noise'], 'c--', label=r'Theoretical $\propto 1/\sqrt{I_0}$')
        ax_theo.plot(x_vals, data['rmse'], 'b-s', label='Measured RMSE')
        ax_theo.set_yscale('log')
        ax_theo.set_title('Noise Validation')
        ax_theo.set_xlabel('Log10(Incident Photons $I_0$)')
        ax_theo.legend()
        ax_theo.grid(True)
        
        self.canvas_plot.fig.tight_layout()
        self.canvas_plot.draw()


class TabScenario2(BaseScenarioTab):
    def add_hyperparameters(self):
        self.params_layout.addWidget(QLabel("FBP Filter (Baseline):"))
        self.filter_combobox = QComboBox()
        self.filter_combobox.addItems(["ramp", "shepp-logan", "hann", "hamming"])
        self.filter_combobox.setCurrentText("hann")
        self.params_layout.addWidget(self.filter_combobox)
        
        desc1 = QLabel("Filter used to generate the noisy starting image before applying TV Denoising.")
        desc1.setStyleSheet("color: #718096; font-size: 11px; margin-bottom: 5px;")
        desc1.setWordWrap(True)
        self.params_layout.addWidget(desc1)
        
        self.params_layout.addWidget(QLabel("TV Iterations:"))
        self.tv_iter_spin = QSpinBox()
        self.tv_iter_spin.setRange(10, 1000)
        self.tv_iter_spin.setValue(100) # Slightly lower for faster UX default
        self.tv_iter_spin.setSingleStep(10)
        self.params_layout.addWidget(self.tv_iter_spin)
        
        desc2 = QLabel("Number of gradient descent steps. More steps yield stronger smoothing but take longer.")
        desc2.setStyleSheet("color: #718096; font-size: 11px; margin-bottom: 5px;")
        desc2.setWordWrap(True)
        self.params_layout.addWidget(desc2)
        
        self.params_layout.addWidget(QLabel("TV Step Size:"))
        self.tv_step_spin = QDoubleSpinBox()
        self.tv_step_spin.setRange(0.01, 1.0)
        self.tv_step_spin.setValue(0.05)
        self.tv_step_spin.setSingleStep(0.01)
        self.params_layout.addWidget(self.tv_step_spin)
        
        desc3 = QLabel("Gradient descent learning rate. Too high causes instability, too low converges slowly.")
        desc3.setStyleSheet("color: #718096; font-size: 11px; margin-bottom: 10px;")
        desc3.setWordWrap(True)
        self.params_layout.addWidget(desc3)

    def run_scenario(self):
        self.set_running_state(True)
        self.worker = WorkerScenario2(self.phantom, self.custom_math_cb.isChecked())
        self.worker.filter_name = self.filter_combobox.currentText()
        self.worker.tv_iterations = self.tv_iter_spin.value()
        self.worker.tv_step = self.tv_step_spin.value()
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
        
        # Plot Metrics (PSNR & CNR & EPV)
        self.canvas_plot.fig.clear()
        ax1, ax3 = self.canvas_plot.fig.subplots(1, 2)
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

        ax3.plot(data['lambdas'], data['epv'], 'y-d', label='EPV')
        ax3.axhline(y=data['baseline_epv'], color='y', linestyle='--', alpha=0.5)
        ax3.set_xlabel(r'TV $\lambda$')
        ax3.set_ylabel('Edge Preservation Value (EPV)', color='y')
        ax3.set_title('Edge Preservation vs Noise Smoothing')
        ax3.grid(True)
        ax3.legend()
        
        self.canvas_plot.fig.tight_layout()
        self.canvas_plot.draw()


class TabScenario3(BaseScenarioTab):
    def add_hyperparameters(self):
        self.params_layout.addWidget(QLabel("FBP Filter:"))
        self.filter_combobox = QComboBox()
        self.filter_combobox.addItems(["ramp", "shepp-logan", "hann", "hamming"])
        self.filter_combobox.setCurrentText("hann")
        self.params_layout.addWidget(self.filter_combobox)
        
        desc1 = QLabel("Convolution Kernel used. Affects only the FBP comparator results, not ART.")
        desc1.setStyleSheet("color: #718096; font-size: 11px; margin-bottom: 5px;")
        desc1.setWordWrap(True)
        self.params_layout.addWidget(desc1)
        
        self.params_layout.addWidget(QLabel("ART Iterations:"))
        self.art_iter_spin = QSpinBox()
        self.art_iter_spin.setRange(1, 20)
        self.art_iter_spin.setValue(3)
        self.params_layout.addWidget(self.art_iter_spin)
        
        desc2 = QLabel("Number of full mathematical passes over all sparse projection angles.")
        desc2.setStyleSheet("color: #718096; font-size: 11px; margin-bottom: 5px;")
        desc2.setWordWrap(True)
        self.params_layout.addWidget(desc2)
        
        self.params_layout.addWidget(QLabel("ART Relaxation (λ):"))
        self.art_relax_spin = QDoubleSpinBox()
        self.art_relax_spin.setRange(0.1, 2.0)
        self.art_relax_spin.setValue(1.0)
        self.art_relax_spin.setSingleStep(0.1)
        self.params_layout.addWidget(self.art_relax_spin)
        
        desc3 = QLabel("Step size of error backprojection. Values < 1.0 suppress noise but slow convergence.")
        desc3.setStyleSheet("color: #718096; font-size: 11px; margin-bottom: 10px;")
        desc3.setWordWrap(True)
        self.params_layout.addWidget(desc3)

    def run_scenario(self):
        self.set_running_state(True)
        self.worker = WorkerScenario3(self.phantom, self.custom_math_cb.isChecked())
        self.worker.filter_name = self.filter_combobox.currentText()
        self.worker.art_iterations = self.art_iter_spin.value()
        self.worker.art_relaxation = self.art_relax_spin.value()
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

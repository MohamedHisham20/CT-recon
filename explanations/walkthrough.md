# CT-Recon Refactoring & Math Fix Walkthrough

## Summary of Changes

The project has been successfully restructured from a monolithic `project.py` file into a professional, modular python package structure. In addition, critical mathematical and logical flaws in the original code have been resolved.

### 1. Modular Structure Implemented
The application is now divided logically:
- **`main.py`**: The application entry point.
- **`core/`**: Houses the mathematical and physical logic.
  - `metrics.py`: Contains standard metrics (`RMSE`, `PSNR`, `CNR`, `NPS`).
  - `reconstruction.py`: Contains custom implementations of FBP, TV Denoising, and the newly added ART algorithm.
  - `physics.py`: Contains the simulation code with Beer-Lambert attenuation and Poisson noise models.
- **`gui/`**: Contains all PyQt components.
  - `main_window.py`: The `CTMainWindow` assembly.
  - `tabs.py`: The UI layouts for the three scenarios.
  - `workers.py`: The background `QThread` workers handling heavy computation.
  - `components.py`: The Matplotlib canvas integrations.

### 2. Math and Logic Fixes

#### a) NPS Normalization
In `core/metrics.py`, the 1D Noise Power Spectrum computation lacked absolute scaling. The magnitude of the spectral power was raw, meaning it was dependent on the input image resolution instead of pixel counts.
> [!NOTE]
> The absolute squared FFT is now properly normalized by the total number of pixels `N` (i.e. `nps_2d = (np.abs(noise_fft)**2) / N`).

#### b) Geometric Mismatch in ART (Scenario 3)
The original `WorkerScenario3` used `iradon_sart` from skimage for the iterative reconstruction. However, when the user enabled "Use Custom Math", the projections were generated using `custom_radon` (which uses interpolation-based rotation). The SART implementation in `skimage` uses strict Siddon ray-tracing line integrals, resulting in a geometric mismatch that causes convergence artifacts.
> [!IMPORTANT]
> A `custom_art` implementation was written in `core/reconstruction.py`. It uses the exact same `scipy.ndimage.rotate` logic for forward/backward projections as `custom_radon`. `WorkerScenario3` has been updated to seamlessly switch to this `custom_art` algorithm when "Use Custom Math" is checked, ensuring geometric consistency and stable convergence!

The application is now ready to be run using:
```bash
python main.py
```

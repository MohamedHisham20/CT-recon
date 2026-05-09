import numpy as np
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from skimage.transform import radon, iradon, iradon_sart

from core.physics import simulate_measurements
from core.reconstruction import custom_radon, custom_iradon, custom_art, tv_denoise_gd
from core.metrics import (
    calculate_psnr,
    calculate_rmse,
    calculate_cnr,
    get_1d_nps,
    calculate_epv,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        self.filter_name = "hann"

    def run(self):
        logger.info("=== Scenario 1: Dose vs Noise Trade-off ===")
        logger.info(f"Using custom implementation: {self.use_custom}")

        # Determine whether to use the custom implementation or the fast skimage implementation
        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon

        # We use a very dense angular sampling (720 angles over 180 degrees) for the baseline
        # add these to be controlled by the user later if they want to see the effect of angular sampling as well
        theta = np.linspace(0.0, 180.0, 720, endpoint=False)
        logger.info(f"Number of projection angles: {len(theta)}")

        # Test a range of incident photon doses (I0) from very low to high
        # also add an infinite dose case to show the perfect mathematical sinogram without noise as a baseline and makr this list controllable by the user later if they want to see more or fewer dose levels
        doses = [10**2, 10**3, 10**4, 10**5, 10**6]
        logger.info(f"Testing doses: {doses}")

        # Generate the ground truth noiseless baseline (I0 = infinity)
        logger.info("Generating noiseless baseline (I0 = infinity)")
        sino_clean = simulate_measurements(self.phantom, theta, np.inf, radon_fn)
        logger.info("Reconstructing baseline image")
        recon_clean = (
            iradon_fn(sino_clean, theta=theta, filter_name=self.filter_name)
            if self.use_custom
            else iradon_fn(sino_clean, theta=theta, filter_name=self.filter_name)
        )

        # Lists to hold the evaluation metrics for each dose
        psnr_results, rmse_results, reconstructions = [], [], []
        nps_profiles = []
        expected_noise = []

        # Simulate and reconstruct for each dose level
        for idx, I0 in enumerate(doses):
            logger.info(f"Processing dose {idx + 1}/{len(doses)}: I0 = {I0}")

            # Simulate physics: add Beer-Lambert attenuation and Poisson quantum noise
            sino_noisy = simulate_measurements(self.phantom, theta, I0, radon_fn)
            logger.debug(f"  Sinogram generated, shape: {sino_noisy.shape}")

            # Reconstruct the noisy sinogram using FBP
            recon_noisy = (
                iradon_fn(sino_noisy, theta=theta, filter_name=self.filter_name)
                if self.use_custom
                else iradon_fn(sino_noisy, theta=theta, filter_name=self.filter_name)
            )
            logger.debug(
                f"  Reconstruction complete, range: [{recon_noisy.min():.4f}, {recon_noisy.max():.4f}]"
            )

            # Save the image and calculate metrics against the noiseless baseline
            reconstructions.append(recon_noisy)
            psnr = calculate_psnr(recon_clean, recon_noisy)
            rmse = calculate_rmse(recon_clean, recon_noisy)
            psnr_results.append(psnr)
            rmse_results.append(rmse)
            logger.info(f"  Metrics: PSNR={psnr:.2f} dB, RMSE={rmse:.6f}")

            # Theoretical standard deviation is proportional to 1/sqrt(I0)
            expected_noise.append(1.0 / np.sqrt(I0))

            # Calculate the 1D Noise Power Spectrum to observe the frequency distribution of the noise
            nps_profiles.append(get_1d_nps(recon_clean, recon_noisy))

        logger.info("Scenario 1 complete - emitting results")
        # Emit all calculated data back to the main GUI thread to be plotted
        self.finished.emit(
            {
                "doses": doses,
                "recon_clean": recon_clean,
                "reconstructions": reconstructions,
                "psnr": psnr_results,
                "rmse": rmse_results,
                "nps_profiles": nps_profiles,
                "expected_noise": expected_noise,
            }
        )


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
        self.filter_name = "hann"
        self.tv_iterations = 200 #this should be controlled by the user from the UI
        self.tv_step = 0.05 #this also should be controlled by the user from the ui 

    def run(self):
        logger.info("=== Scenario 2: TV Regularization ===")
        logger.info(f"Using custom implementation: {self.use_custom}")

        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon

        # Dense sampling baseline
        #  this should be controlled by the user from the UI
        theta = np.linspace(0.0, 180.0, 720, endpoint=False) 
        #this should be controlled by the user from the UI
        I0_low = 10**3  # Use a relatively low dose to ensure visible noise
        logger.info(f"Low dose I0 = {I0_low}")

        # We will test different strengths of the TV regularization parameter (lambda)
        #this should be controlled by the user from the UI
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]
        logger.info(f"Testing lambda values: {lambdas}")

        # Ground truth clean image (infinite dose)
        logger.info("Generating ground truth baseline")
        sino_clean = simulate_measurements(self.phantom, theta, np.inf, radon_fn)
        recon_clean = (
            iradon_fn(sino_clean, theta=theta, filter_name=self.filter_name)
            if self.use_custom
            else iradon_fn(sino_clean, theta=theta, filter_name=self.filter_name)
        )

        # Low dose simulation
        logger.info(f"Simulating low-dose scan (I0={I0_low})")
        sino_noisy = simulate_measurements(self.phantom, theta, I0_low, radon_fn)

        # Standard FBP reconstruction of the noisy data (our noisy baseline)
        logger.info("Reconstructing noisy FBP image")
        recon_fbp = (
            iradon_fn(sino_noisy, theta=theta, filter_name=self.filter_name)
            if self.use_custom
            else iradon_fn(sino_noisy, theta=theta, filter_name=self.filter_name)
        )

        # Calculate baseline metrics for FBP before any denoising is applied
        logger.info("Calculating baseline metrics")
        baseline_psnr = calculate_psnr(recon_clean, recon_fbp)
        baseline_cnr = calculate_cnr(recon_fbp)  # Contrast-to-Noise Ratio
        baseline_epv = calculate_epv(recon_clean, recon_fbp)
        logger.info(
            f"Baseline: PSNR={baseline_psnr:.2f} dB, CNR={baseline_cnr:.4f}, EPV={baseline_epv:.4f}"
        )

        tv_reconstructions, psnr_results, cnr_results, epv_results = [], [], [], []

        # Apply TV gradient descent denoising for each lambda value
        for idx, lda in enumerate(lambdas):
            logger.info(
                f"Applying TV denoising: lambda={lda} ({idx + 1}/{len(lambdas)})"
            )
            recon_tv = tv_denoise_gd(
                recon_fbp,
                lambda_tv=lda,
                iterations=self.tv_iterations,
                step_size=self.tv_step,
            )

            # Record the denoised image and its quality metrics
            tv_reconstructions.append(recon_tv)
            psnr = calculate_psnr(recon_clean, recon_tv)
            cnr = calculate_cnr(recon_tv)
            epv = calculate_epv(recon_clean, recon_tv)
            psnr_results.append(psnr)
            cnr_results.append(cnr)
            epv_results.append(epv)
            logger.info(f"  Results: PSNR={psnr:.2f} dB, CNR={cnr:.4f}, EPV={epv:.4f}")

        logger.info("Scenario 2 complete - emitting results")

        # Emit results to GUI
        self.finished.emit(
            {
                "lambdas": lambdas,
                "recon_fbp": recon_fbp,
                "baseline_psnr": baseline_psnr,
                "baseline_cnr": baseline_cnr,
                "baseline_epv": baseline_epv,
                "tv_reconstructions": tv_reconstructions,
                "psnr": psnr_results,
                "cnr": cnr_results,
                "epv": epv_results,
            }
        )


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
        self.filter_name = "hann"
        self.art_iterations = 3 #this should be controlled by the user from the UI
        self.art_relaxation = 1.0 #this should be controlled by the user from the UI

    def run(self):
        logger.info("=== Scenario 3: Sparse Angular Sampling ===")
        logger.info(f"Using custom implementation: {self.use_custom}")

        def _gain_match(reference, image):
            """
            Match global intensity scale of `image` to `reference` before PSNR.
            This avoids penalizing custom ART for amplitude bias when structure is correct.
            """
            denom = float(np.sum(image * image))
            if denom <= 1e-12:
                return image
            alpha = float(np.sum(reference * image) / denom)
            return np.clip(alpha * image, 0.0, None)

        # Choose which mathematical library to use
        radon_fn = custom_radon if self.use_custom else radon
        iradon_fn = custom_iradon if self.use_custom else iradon

        # Define experiment parameters
        # We use a moderate photon count so noise isn't the primary issue, only sparse data
        #this should be controlled by the user from the UI
        I0_mod = 10**5

        # Progressively reduce the number of projection angles (Sparse Sampling)
        #this should be controlled by the user from the UI
        angles_list = [180, 120, 90, 60, 45, 36, 24, 18]

        # We only want to plot these specific subset of angles to avoid overcrowding the UI
        #this should be controlled by the user from the UI
        viz_angles = [180, 90, 45, 18]

        # Create a "perfect" ground truth reference using 720 dense angles and infinite dose
        #this should be info add to the UI
        logger.info("Generating ground truth reference (720 angles, infinite dose)")
        theta_full = np.linspace(0.0, 180.0, 720, endpoint=False)
        sino_full = simulate_measurements(self.phantom, theta_full, np.inf, radon_fn)
        recon_full = (
            iradon_fn(sino_full, theta=theta_full, filter_name=self.filter_name)
            if self.use_custom
            else iradon_fn(sino_full, theta=theta_full, filter_name=self.filter_name)
        )

        psnr_fbp, psnr_art = [], []
        recons_fbp, recons_art = [], []

        # Iterate over our decreasing list of available angles
        for idx, n_angles in enumerate(angles_list):
            logger.info(f"Processing: {n_angles} angles ({idx + 1}/{len(angles_list)})")

            # Generate the sparse angles array
            theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)

            # Simulate the scan with the reduced number of angles
            sino_sparse = simulate_measurements(self.phantom, theta, I0_mod, radon_fn)

            # FBP reconstruction
            logger.debug(f"  FBP reconstruction...")
            recon_fbp_sparse = (
                iradon_fn(sino_sparse, theta=theta, filter_name=self.filter_name)
                if self.use_custom
                else iradon_fn(sino_sparse, theta=theta, filter_name=self.filter_name)
            )
            recon_fbp_for_metric = (
                _gain_match(recon_full, recon_fbp_sparse)
                if self.use_custom
                else recon_fbp_sparse
            )
            psnr_fbp_val = calculate_psnr(recon_full, recon_fbp_for_metric)
            psnr_fbp.append(psnr_fbp_val)
            logger.debug(f"  FBP PSNR: {psnr_fbp_val:.2f} dB")

            # ART reconstruction
            logger.debug(f"  ART reconstruction (iterations={self.art_iterations})...")
            if self.use_custom:
                recon_art_sparse = custom_art(
                    sino_sparse,
                    theta=theta,
                    iterations=self.art_iterations,
                    relaxation=self.art_relaxation,
                )
            else:
                recon_art_sparse = iradon_sart(
                    sino_sparse, theta=theta, relaxation=self.art_relaxation
                )
                for _ in range(self.art_iterations):
                    recon_art_sparse = iradon_sart(
                        sino_sparse,
                        theta=theta,
                        image=recon_art_sparse,
                        relaxation=self.art_relaxation,
                    )

            recon_art_for_metric = (
                _gain_match(recon_full, recon_art_sparse)
                if self.use_custom
                else recon_art_sparse
            )
            psnr_art_val = calculate_psnr(recon_full, recon_art_for_metric)
            psnr_art.append(psnr_art_val)
            logger.debug(f"  ART PSNR: {psnr_art_val:.2f} dB")

            # If the current angle count is in our visualization list, save the images
            if n_angles in viz_angles:
                recons_fbp.append(recon_fbp_sparse)
                recons_art.append(recon_art_sparse)
                logger.debug(f"  Saved for visualization")

        logger.info("Scenario 3 complete - emitting results")
        # Send all data to the UI thread
        self.finished.emit(
            {
                "angles_list": angles_list,
                "viz_angles": viz_angles,
                "psnr_fbp": psnr_fbp,
                "psnr_art": psnr_art,
                "recons_fbp": recons_fbp,
                "recons_art": recons_art,
            }
        )

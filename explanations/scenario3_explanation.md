# Scenario 3: Sparse Angular Sampling & Iterative Reconstruction (ART)

Scenario 3 is designed to explore what happens when we reduce the amount of radiation exposure by drastically decreasing the number of projection angles (sparse sampling), and how advanced iterative techniques (ART) compare to traditional techniques (FBP) under these harsh conditions.

Let's break down the code that makes this happen line-by-line, exploring both the software engineering and the physics behind it.

---

## 1. Setting up the Baseline (`WorkerScenario3.run`)

This process happens in the background via a PyQt `QThread` so it doesn't freeze the user interface.

```python
# 1. Choose which mathematical library to use
radon_fn = custom_radon if self.use_custom else radon
iradon_fn = custom_iradon if self.use_custom else iradon

# 2. Define our experiment parameters
angles_list = [180, 120, 90, 60, 45, 36, 24, 18] # Progressively fewer angles
I0_mod = 10**5                                   # Moderate photon dose
viz_angles = [180, 90, 45, 18]                   # We only save images for these specific angles to plot them

# 3. Establish the ground truth (Ideal scenario)
theta_full = np.linspace(0.0, 180.0, 720, endpoint=False) 
sino_full = simulate_measurements(self.phantom, theta_full, np.inf, radon_fn)
recon_full = iradon_fn(sino_full, theta=theta_full)
```

**Physics Insight:**
To measure how "good" our sparse reconstructions are, we need a "Ground Truth" to compare against. We generate `theta_full` consisting of 720 perfectly dense projection angles.
We pass `I0 = np.inf` (infinite photons) to `simulate_measurements`, meaning absolutely zero quantum noise. The resulting `recon_full` acts as our perfect, noiseless baseline for PSNR metric comparisons.

---

## 2. The Physics Simulation (`simulate_measurements`)

Before we loop through the sparse angles, let's look at how the physical data is generated inside `core/physics.py`.

```python
def simulate_measurements(phantom, theta, I0, radon_func=radon):
    # 1. Compute ideal mathematical line integrals (radon transform)
    sino_clean = radon_func(phantom, theta=theta)
    
    if I0 is None or I0 == np.inf:
        return sino_clean
        
    # 2. Scale values for realistic human body attenuation
    sino_max = np.max(sino_clean)
    if sino_max == 0: sino_max = 1.0
    scale_factor = 5.0 / sino_max
    sino_scaled = sino_clean * scale_factor
```
**Physics Insight:** The digital phantom's pixel values are usually between 0.0 and 1.0. If we treat these directly as linear attenuation coefficients ($\mu$), the attenuation would be too low to simulate a realistic CT scan of a human body. By scaling the maximum projection value to `5.0`, we mimic a scenario where the thickest part of the object attenuates a significant portion of the X-ray beam ($e^{-5} \approx 0.006$, meaning only 0.6% of photons make it through the thickest part).

```python
    # 3. Apply the Beer-Lambert Law and Poisson Noise
    I_d = I0 * np.exp(-sino_scaled)
    I_meas = np.random.poisson(I_d)
    I_meas = np.maximum(I_meas, 1) # Prevent log(0) if no photons reach the detector
```
**Physics Insight:** 
- **Beer-Lambert Law** ($I = I_0 e^{-\int \mu dx}$): `I_d` calculates the *expected* deterministic number of photons that hit each detector bin after passing through the object.
- **Quantum Noise**: X-ray emission and detection are statistical processes governed by Poisson statistics. `np.random.poisson(I_d)` simulates the actual, noisy count of photons detected.

```python
    # 4. Inverse Beer-Lambert (Log-Transform) to get attenuation back
    sino_noisy_scaled = -np.log(I_meas / I0)
    
    # 5. Rescale back to mathematical range for the reconstructor
    return sino_noisy_scaled / scale_factor
```
**Physics Insight:** Real CT scanners only measure photon counts (`I_meas`). To reconstruct the image, we must convert counts back into attenuation line integrals using $-\ln(I/I_0)$. Because of Poisson noise, `I_meas` can sometimes be slightly higher than `I0` (e.g., measuring 1005 photons when 1000 were emitted). This results in negative attenuation, which is physically impossible but statistically standard in raw CT data.

---

## 3. The Sparse Sampling Loop

Back in `WorkerScenario3`, we loop through our drastically shrinking list of angles.

```python
        for n_angles in angles_list:
            # 1. Create sparse angles and simulate the scan
            theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
            sino_sparse = simulate_measurements(self.phantom, theta, I0_mod, radon_fn)
            
            # 2. Reconstruct using traditional FBP
            recon_fbp_sparse = iradon_fn(sino_sparse, theta=theta)
            psnr_fbp.append(calculate_psnr(recon_full, recon_fbp_sparse))
```
**Coding Insight:** FBP (Filtered Backprojection) is an analytical method. It expects a dense continuous angular sampling. As `n_angles` drops from 180 to 18, FBP breaks down completely, generating severe "streak artifacts" because there isn't enough angular data to satisfy the Nyquist-Shannon sampling theorem.

```python
            # 3. Reconstruct using Iterative ART
            if self.use_custom:
                # Our custom ART solver
                recon_art_sparse = custom_art(sino_sparse, theta=theta, iterations=3)
            else:
                # Skimage's SART (Simultaneous Algebraic Reconstruction Technique) solver
                recon_art_sparse = iradon_sart(sino_sparse, theta=theta)
                for _ in range(3):
                    recon_art_sparse = iradon_sart(sino_sparse, theta=theta, image=recon_art_sparse)
```
**Coding Insight:** Instead of analytically inverting the data, ART treats reconstruction as a giant system of linear equations ($Ax = b$). We do 3 passes (iterations) over the data to slowly converge toward the solution.

---

## 4. Deep Dive into `custom_art` 

If you selected the Custom Math checkbox, it runs our `custom_art` function in `core/reconstruction.py`. This is the purest realization of the Kaczmarz method (ART) for solving linear systems.

```python
def custom_art(sinogram, theta, iterations=3):
    size = sinogram.shape[0]
    recon = np.zeros((size, size)) # Start with a completely blank image
    
    for _ in range(iterations):
        for i, angle in enumerate(theta):
```
**Physics Insight:** ART updates the image on a *per-projection* basis. We look at one angle, correct the image, then move to the next angle.

```python
            # 1. Forward Project: What does our current guess look like at this angle?
            rotated = scipy.ndimage.rotate(recon, -angle, reshape=False, order=1)
            proj = np.sum(rotated, axis=0)
            
            # 2. Calculate the Error (Measured data - Simulated guess)
            error = sinogram[:, i] - proj
```
**Coding Insight:** We rotate our current reconstructed image so that the X-rays are parallel to the vertical axis, then we sum them up to simulate a projection. We compare this simulated projection to the actual `sinogram` measured by the machine.

```python
            # 3. Backproject the Error
            update = np.zeros((size, size))
            update[:, :] = error[np.newaxis, :] / size # Smear the error uniformly across the ray path
            
            update_rotated_back = scipy.ndimage.rotate(update, angle, reshape=False, order=1)
            recon += update_rotated_back
```
**Physics/Coding Insight:** If our guess resulted in fewer photons being attenuated than the real scanner measured, we have to "add density" to the image along that ray. Because we don't know *where* along the ray the missing mass is, we distribute the error uniformly across the entire length of the ray (`error / size`). We then rotate that correction back to the original angle and add it to our image.

```python
            # 4. Enforce Physical Constraints
            recon = np.maximum(recon, 0)
```
**Physics Insight:** Matter cannot have negative density. By enforcing that all pixels must be $\ge 0$ at the end of every angle update, ART is capable of suppressing streaks that FBP cannot, making it far superior for sparse sampling!

---

## 5. Sending Data to the GUI

```python
            # Store the images if this angle is in our visualizer list
            if n_angles in viz_angles:
                recons_fbp.append(recon_fbp_sparse)
                recons_art.append(recon_art_sparse)
                
        # Send everything back to the UI thread to be plotted!
        self.finished.emit({
            'angles_list': angles_list, 'viz_angles': viz_angles,
            'psnr_fbp': psnr_fbp, 'psnr_art': psnr_art,
            'recons_fbp': recons_fbp, 'recons_art': recons_art
        })
```
**Coding Insight:** Qt requires that GUI updates happen on the main thread. We emit a dictionary via a Qt Signal, which the `TabScenario3` receives in its `on_finished` function. The GUI then loops through `viz_angles`, pulls out the 4 snapshots for FBP and ART, and displays them side-by-side using `matplotlib` along with the PSNR convergence graph!

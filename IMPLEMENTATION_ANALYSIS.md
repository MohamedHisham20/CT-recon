# Implementation Analysis vs. Project Proposal

## 1. Gap Analysis: Implemented vs. Proposed

### ✓ Fully Implemented

| Proposal Section | Status | Files |
|-----------------|--------|-------|
| Beer-Lambert Law + Poisson Noise | ✓ | `core/physics.py:29-43` |
| Radon Transform | ✓ | `core/physics.py`, `core/reconstruction.py:4-20` |
| Convolution Backprojection (FBP) | ✓ | `core/reconstruction.py:22-69` |
| Filter kernels (Ramp, Hann) | ✓ | `core/reconstruction.py:33-38` |
| Total Variation Denoising | ✓ | `core/reconstruction.py:105-131` |
| ART (custom + SART) | ✓ | `core/reconstruction.py:71-103`, `gui/workers.py:181-185` |
| RMSE, PSNR, CNR metrics | ✓ | `core/metrics.py` |
| 1D NPS | ✓ | `core/metrics.py:41-70` |
| Scenario 1: Dose vs. Noise | ✓ | `gui/workers.py:9-62` |
| Scenario 2: TV Regularization | ✓ | `gui/workers.py:65-121` |
| Scenario 3: Sparse Sampling | ✓ | `gui/workers.py:124-199` |

### ❌ Missing / Partially Implemented

| Item | Location | Impact | Recommendation |
|------|----------|--------|-----------------|
| **HU Scale Conversion** | `core/physics.py` | No clinical relevance - can't compare to real CT values | Add `to_hounsfield_units()` function |
| **Shepp-Logan Filter** | `core/reconstruction.py:37` | UI only exposes Hann; proposal includes 3 filters | Add filter selection parameter |
| **2D NPS Visualization** | `core/metrics.py` | Only 1D radial; no 2D frequency domain view | Add `get_2d_nps()` returning full 2D spectrum |
| **Theoretical Noise Validation** | `gui/workers.py` | Doesn't compute theoretical σ = 1/√I₀ to compare with observed | Add noise comparison plot in Scenario 1 |
| **Edge Preservation Quantification** | Scenario 2 | No explicit edge sharpness metric | Add gradient-based edge metric |
| **ART relaxation parameter (λ)** | `core/reconstruction.py:71` | Hardcoded; not exposed as hyperparameter | Add `relaxation` parameter to `custom_art()` |
| **Multiple ART iterations for custom** | `gui/workers.py:179` | Custom ART does 3 iterations but SART loop does 4 total | Make iteration count configurable |

---

## 2. Library Functions: Mathematical Definitions & Controllable Hyperparameters

### A. `skimage.transform.radon`

**Mathematical Definition:**
Implements the discrete Radon transform via the **Joseph method** (line-driven integration):
$$g(l,\theta) = \sum_{(x,y) \in \text{ray}} f(x,y) \cdot \Delta s$$

Where rays are traced through the image at angle θ and detector position l, summing pixel values along each ray path.

**Controllable Hyperparameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `circle` | bool | Assume circular support (reduces computation) | `True` |
| `output_size` | int | Size of square output sinogram | Match input |
| `interpolation` | int | Interpolation order (0=nearest, 1=linear, 3=cubic) | `1` |

---

### B. `skimage.transform.iradon`

**Mathematical Definition:**
Implements filtered backprojection using the **Fourier Slice Theorem**:
$$f(x,y) = \int_0^\pi \int_{-\infty}^{\infty} G(\omega,\theta) |\omega| e^{j2\pi\omega(x\cos\theta + y\sin\theta)} \, d\omega \, d\theta$$

Where:
- $G(\omega,\theta)$ is the 1D Fourier transform of projection $g(l,\theta)$ at angle θ
- $|\omega|$ is the **Ramp filter** that counteracts the 1/ρ blurring inherent in simple backprojection
- The filter is applied in frequency domain before inverse FFT and backprojection

**Controllable Hyperparameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `filter` | str | Filter kernel: `"ramp"`, `"shepp-logan"`, `"hann"`, `"hamming"`, `"cosine"`, `"parzen"`, `"none"` | `"ramp"` |
| `interpolation` | str | Backprojection interpolation: `"linear"`, `"nearest"`, `"cubic"` | `"linear"` |
| `output_size` | int | Reconstructed image size | Match input |
| `circle` | bool | Assume circular image support | `True` |

---

### C. `skimage.transform.iradon_sart` (Simultaneous Algebraic Reconstruction Technique)

**Mathematical Definition:**
SART updates all pixels simultaneously after computing the error for all rays at one projection angle:
$$f^{(k+1)}_j = f^{(k)}_j + \lambda \frac{\sum_{i \in \theta} a_{ij} \frac{g_i - \sum_m a_{im} f^{(k)}_m}{\sum_m a_{im}^2}}{\sum_{i \in \theta} a_{ij}}$$

Where:
- $g_i$ is the $i$-th projection measurement
- $a_{ij}$ are system matrix elements (ray-pixel intersections)
- $\lambda$ is the relaxation parameter (0 < λ ≤ 2)
- The summation in numerator is over all rays at current angle θ

**Controllable Hyperparameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `iterations` | int | Number of SART passes | `1` |
| `relaxation` | float | Relaxation parameter λ (0-2) | `1.0` |
| `clip` | tuple | Clamp values to non-negative range | `(0, None)` |
| `minimizer` | str | Minimization method: `"cg"` (conjugate gradient) or `"adam"` | `"cg"` |
| `verbose` | bool | Print progress | `False` |

---

### D. `numpy.fft.fft` / `numpy.fft.ifft`

**Mathematical Definition:**
Discrete Fourier Transform (DFT):
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn/N}, \quad k = 0, 1, \ldots, N-1$$

Inverse:
$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{j2\pi kn/N}$$

**Controllable Hyperparameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | int | Transform along specified axis |
| `norm` | str | Normalization: `"backward"` (no normalization), `"ortho"` (unitary), `"forward"` (1/N) |

---

### E. `scipy.ndimage.rotate`

**Mathematical Definition:**
Applies a 2D rotation transformation using spline interpolation:
$$x' = x\cos\theta - y\sin\theta$$
$$y' = x\sin\theta + y\cos\theta$$

Pixel values are interpolated using B-splines of order `order` (0=nearest neighbor, 1=linear, 3=cubic).

**Controllable Hyperparameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `angle` | float | Rotation angle in degrees (counterclockwise) | Required |
| `reshape` | bool | Adjust output size to fit rotated image | `True` |
| `order` | int | Spline interpolation order (0-5) | `3` |
| `mode` | str | Boundary handling: `"constant"`, `"reflect"`, `"nearest"`, `"wrap"` | `"constant"` |
| `prefilter` | bool | Apply spline filter before rotation | `True` |

---

### F. `numpy.random.poisson`

**Mathematical Definition:**
Draws samples from Poisson distribution (discrete count statistics):
$$P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

Where λ is the expected (mean) number of events.

**Controllable Hyperparameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `lam` | float or array-like | Expected value λ (must be non-negative) | Required |
| `size` | int or tuple | Output shape | `None` (single sample) |

---

### G. `numpy.fft.fftshift` / `numpy.fft.ifftshift`

**Mathematical Definition:**
Shifts the zero-frequency component to the center of the spectrum for visualization:
- `fftshift`: Moves quadrants by swapping
- `ifftshift`: Reverses the shift

Used in NPS calculation to center low frequencies.

---

## 3. Custom Implementation Mathematical Details

### Custom Radon (`core/reconstruction.py:4-20`)

**Method:** Image rotation + vertical projection
$$g(\theta) = \sum_{y} f(R_{-\theta}(x, y))$$

Where the image is rotated by -θ, then summed along vertical axis (axis=0).

### Custom FBP (`core/reconstruction.py:22-69`)

**Method:**
1. **Filtering:** Apply frequency-domain filter $H(\omega) = |\omega| \cdot W(\omega)$ where $W$ is window (Hann)
2. **Backprojection:** For each angle, interpolate filtered projection onto image grid:
   $$f(x,y) += g_f(t, \theta) \text{ where } t = x\cos\theta + y\sin\theta$$

### Custom ART (`core/reconstruction.py:71-103`)

**Method:**
$$f^{(k+1)}_j = f^{(k)}_j + \lambda \frac{g_i - \sum_j a_{ij} f^{(k)}_j}{\sum_j a_{ij}^2} a_{ij}$$

Simplified uniform ray weights: update distributed as error / size

### TV Denoising (`core/reconstruction.py:105-131`)

**Method:** Gradient descent on:
$$E(u) = \frac{1}{2}\|u - f\|_{2}^{2} + \lambda \sum_{i,j} \sqrt{(u_{i+1,j}-u_{i,j})^{2} + (u_{i,j+1}-u_{i,j})^{2}}$$

Gradient: $\nabla E = (u - f) - \lambda \cdot \text{div}\left(\frac{\nabla u}{|\nabla u|}\right)$

---

## 4. Recommended Additions

1. **Hounsfield Unit conversion** in `core/physics.py`:
```python
def to_hounsfield_units(image, mu_water=0.19):
    return 1000 * (image - mu_water) / mu_water
```

2. **Expose Shepp-Logan filter** - add filter parameter to reconstruction functions

3. **Add theoretical noise comparison** in Scenario 1:
```python
sigma_theory = 1.0 / np.sqrt(np.array(doses))
```

4. **Add 2D NPS** - return full 2D spectrum for frequency domain analysis

5. **Configurable ART iterations** - expose `iterations` parameter

---

## 5. Implementation Completion Guide

### Phase 1: Core Utilities (Priority: HIGH)

#### 5.1 Add Hounsfield Unit Conversion

**File:** `core/physics.py`

**Purpose:** Convert reconstructed images to clinical HU scale for realistic comparison.

**Implementation:**
```python
def to_hounsfield_units(image, mu_water=0.19):
    """
    Convert attenuation values to Hounsfield Units.
    
    Args:
        image: Reconstructed attenuation image (normalized 0-1)
        mu_water: Linear attenuation coefficient of water at effective energy
                  Typical value ~0.19 cm^-1 for CT energy ~70 keV
    
    Returns:
        Image in HU scale: air=-1000, water=0, bone~400-1000
    """
    # Scale factor converts 0-1 normalized range to clinically meaningful attenuation
    # Assuming phantom max ~5.0 (from scale_factor in simulate_measurements)
    mu_image = image * 5.0  # Convert back to physical attenuation
    hu = 1000 * (mu_image - mu_water) / mu_water
    return hu

def from_hounsfield_units(hu, mu_water=0.19):
    """Reverse HU to normalized attenuation."""
    mu_image = (hu / 1000 + 1) * mu_water
    return mu_image / 5.0
```

**Testing:**
- Water (HU=0) should return 0
- Air (HU=-1000) should return -1000
- Bone (HU~500) should return ~500

---

#### 5.2 Add 2D NPS Function

**File:** `core/metrics.py`

**Purpose:** Provide full 2D noise power spectrum for frequency domain analysis.

**Implementation:**
```python
def get_2d_nps(image_true, image_test):
    """
    Computes the 2D Noise Power Spectrum.
    
    NPS(f_x, f_y) = < |N(f_x, f_y)|^2 >
    
    Args:
        image_true: Ground truth (noise-free) image
        image_test: Reconstructed image with noise
    
    Returns:
        nps_2d: 2D noise power spectrum (centered)
        freqs_x, freqs_y: Frequency coordinates in cycles/unit
    """
    noise = image_test - image_true
    N = noise.shape[0]
    
    # 2D FFT with shift
    noise_fft = np.fft.fftshift(np.fft.fft2(noise))
    
    # Power spectrum (magnitude squared normalized)
    nps_2d = np.abs(noise_fft)**2 / (N * N)
    
    # Frequency coordinates
    freqs = np.fft.fftfreq(N)
    freqs_x, freqs_y = np.meshgrid(freqs, freqs, indexing='ij')
    freqs_x = np.fft.fftshift(freqs_x)
    freqs_y = np.fft.fftshift(freqs_y)
    
    return nps_2d, freqs_x, freqs_y
```

---

#### 5.3 Add Edge Preservation Metric

**File:** `core/metrics.py`

**Purpose:** Quantify how well edges are preserved in denoised images.

**Implementation:**
```python
def calculate_edge_preservation(image_true, image_test):
    """
    Calculate Edge Preservation Ratio using Sobel gradients.
    
    EPR = ||∇f_test||_1 / ||∇f_true||_1
    
    Values:
        - EPR = 1.0: Perfect edge preservation
        - EPR < 1.0: Edges blurred
        - EPR > 1.0: Edges amplified (over-enhancement)
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T
    
    grad_true_x = scipy.ndimage.convolve(image_true, sobel_x)
    grad_true_y = scipy.ndimage.convolve(image_true, sobel_y)
    grad_test_x = scipy.ndimage.convolve(image_test, sobel_x)
    grad_test_y = scipy.ndimage.convolve(image_test, sobel_y)
    
    mag_true = np.sqrt(grad_true_x**2 + grad_true_y**2)
    mag_test = np.sqrt(grad_test_x**2 + grad_test_y**2)
    
    epr = np.sum(mag_test) / (np.sum(mag_true) + 1e-10)
    return epr
```

---

### Phase 2: Enhanced Reconstruction (Priority: HIGH)

#### 5.4 Expose Filter Selection

**File:** `core/reconstruction.py`

**Modify:** `custom_iradon` to accept filter parameter

```python
def custom_iradon(sinogram, theta, filter_name='hann'):
    """
    Args:
        filter_name: One of 'ramp', 'shepp-logan', 'hann', 'hamming', 'cosine'
    """
    # ... existing ramp filter creation ...
    
    # Add filter options
    if filter_name == 'ramp':
        pass  # Already ramp
    elif filter_name == 'shepp_logan':
        ramp_filter *= np.sinc(np.linspace(-1, 1, n_det))
    elif filter_name == 'hamming':
        ramp_filter *= np.hamming(n_det)
    elif filter_name == 'cosine':
        ramp_filter *= np.cos(np.linspace(-np.pi/2, np.pi/2, n_det))
```

**Update:** `gui/workers.py` to pass filter parameter from UI

---

#### 5.5 Add Configurable ART Parameters

**File:** `core/reconstruction.py`

```python
def custom_art(sinogram, theta, iterations=3, relaxation=1.0, tolerance=1e-5):
    """
    Args:
        iterations: Number of full passes through all projections
        relaxation: Relaxation parameter λ (0 < λ ≤ 2)
                   - λ=1: Standard ART
                   - λ<1: Under-relaxation (more stable, slower)
                   - λ>1: Over-relaxation (faster, may diverge)
        tolerance: Stop early if update norm < tolerance
    """
    size = sinogram.shape[0]
    recon = np.zeros((size, size))
    
    for it in range(iterations):
        recon_old = recon.copy()
        
        for i, angle in enumerate(theta):
            # ... existing forward/backproject ...
            
            # Apply relaxation
            recon += relaxation * update_rotated_back
        
        # Check convergence
        if np.linalg.norm(recon - recon_old) / size**2 < tolerance:
            break
    
    return recon
```

---

### Phase 3: Scenario Enhancements (Priority: MEDIUM)

#### 5.6 Scenario 1: Add Theoretical Noise Comparison

**File:** `gui/workers.py` - `WorkerScenario1`

```python
# After computing metrics, add theoretical noise estimation
# For log-transformed data: σ_g² ≈ 1/I_0 (approximately)
# After FBP: noise variance scales with 1/I_0 but also filter response

# Compute observed noise std in uniform region
observed_noise_std = []
for recon in reconstructions:
    # Sample from corner region (uniform)
    corner = recon[220:240, 20:40]
    observed_noise_std.append(np.std(corner))

# Theoretical: σ ≈ 1/√I_0 (normalized to image space)
theoretical_noise = [1.0 / np.sqrt(I0) for I0 in doses]

# Compare in output: return both for plotting
```

---

#### 5.7 Scenario 2: Add Optimal Lambda Detection

**Add auto-detection of best λ:**
```python
# In WorkerScenario2
for lda in lambdas:
    # ... existing reconstruction ...
    psnr_results.append(...)
    cnr_results.append(...)

# Find optimal
best_idx = np.argmax(psnr_results)
optimal_lambda = lambdas[best_idx]
```

---

#### 5.8 Scenario 3: Add Critical Angle Threshold

**Add automatic threshold detection:**
```python
# Find angle where PSNR drops below threshold (e.g., 25 dB)
threshold = 25  # dB
for i, psnr in enumerate(psnr_fbp):
    if psnr < threshold:
        critical_angle_fbp = angles_list[i]
        break
```

---

### Phase 4: UI Enhancements (Priority: LOW)

#### 5.9 Add Display Controls

**In UI components:**
- Filter type dropdown
- ART iterations slider
- TV lambda slider
- Display options: HU toggle, 2D NPS toggle

---

### 5.10 Logging System

**Purpose:** Comprehensive logging at every step for future debugging and analysis.

**Implementation:** Added Python's `logging` module with structured log levels across all core modules:

| File | Log Levels | Information Logged |
|------|------------|-------------------|
| `core/physics.py` | INFO, DEBUG | I0 dose, scale factors, photon counts at each step |
| `core/reconstruction.py` | INFO, DEBUG | Function calls, iteration progress, output ranges |
| `core/metrics.py` | DEBUG | Each metric value computed |
| `gui/workers.py` | INFO, DEBUG | Scenario start/end, dose/lambda processing, metrics per step |

**Log Format:**
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

**Usage:**
- Default level: INFO (shows major steps)
- Enable DEBUG: Set `logging.basicConfig(level=logging.DEBUG)` for detailed tracing

---

### Implementation Checklist

| Task | File | Priority | Status |
|------|------|----------|--------|
| Add HU conversion | `core/physics.py` | HIGH | ☐ |
| Add 2D NPS | `core/metrics.py` | HIGH | ☐ |
| Add edge preservation | `core/metrics.py` | HIGH | ☐ |
| Expose filter parameter | `core/reconstruction.py` | HIGH | ☐ |
| Configurable ART | `core/reconstruction.py` | HIGH | ☐ |
| Scenario 1 noise theory | `gui/workers.py` | MEDIUM | ☐ |
| Scenario 2 optimal λ | `gui/workers.py` | MEDIUM | ☐ |
| Scenario 3 threshold | `gui/workers.py` | MEDIUM | ☐ |
| UI controls update | `gui/components.py` | LOW | ☐ |
| **Add logging** | **All core modules** | **HIGH** | **✓** |
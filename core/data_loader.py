import pydicom
import numpy as np
from skimage.transform import resize

def load_clinical_phantom(dicom_path, output_size=(256, 256), window_center=47, window_width=69):
    """
    Reads a DICOM file, applies HU conversion, clips to clinical window,
    normalizes to [0.0, 1.0], and resizes.
    """
    # 1. Read the DICOM file
    dataset = pydicom.dcmread(dicom_path)
    
    # 2. Extract pixel array and apply RescaleIntercept and RescaleSlope
    image = dataset.pixel_array.astype(np.float32)
    
    intercept = getattr(dataset, 'RescaleIntercept', 0)
    slope = getattr(dataset, 'RescaleSlope', 1)
    
    hu_image = image * slope + intercept
    
    # 3. Clip to specific window
    min_hu = window_center - (window_width / 2.0)
    max_hu = window_center + (window_width / 2.0)
    clipped_hu = np.clip(hu_image, min_hu, max_hu)
    
    # 4. Normalize to strictly [0.0, 1.0] range
    normalized_image = (clipped_hu - min_hu) / (max_hu - min_hu)
    
    # 5. Resize with anti-aliasing
    final_image = resize(normalized_image, output_size, anti_aliasing=True)
    
    return final_image.astype(np.float64)

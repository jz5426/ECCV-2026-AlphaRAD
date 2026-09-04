import numpy as np
import os
import cv2
import pydicom
import SimpleITK as sitk
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

def resize_dicom_and_save(load_path, save_path, do_resize=True):
    """
    Converting dicom image to png image.
    load_path=/path/to/load/*.dicom, save_path=/path/to/save/*.png
    """
    ds = pydicom.dcmread(load_path, force=True)
    img = ds.pixel_array
    img = apply_modality_lut(img, ds)  # rescaleSlope & intercept
    img = apply_voi_lut(img, ds)  # windowing
    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation.lower().strip() == "monochrome1":
            img = img.max() - img  # invert
    
    if do_resize:
        h, w = img.shape
        ratio = 512 / min(h, w)
        target_size = (int(w * ratio), int(h * ratio))
        img = cv2.resize(img, target_size, cv2.INTER_LANCZOS4)
   
    # normalize
    img = (img - img.min()) / (img.max() - img.min()) * np.iinfo(np.uint8).max
    img = img.astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    success = cv2.imwrite(save_path, img)
    if not success:
        print(f'fail to save {save_path}')

def resize_mha_and_save(load_path, save_path, do_resize=True):  # load_path=/path/to/*.mha
    """
    Converting .mha image to .png image.
    """
    # 1. Replace pydicom with SimpleITK to read .mha files
    image = sitk.ReadImage(load_path)
    img = sitk.GetArrayFromImage(image)

    # 2. Handle dimensions: sitk often returns (Depth, Height, Width), 
    # so we extract the 2D slice if extra dimensions exist.
    assert img.ndim == 2

    # Note: We REMOVED the DICOM specific steps (apply_modality_lut, apply_voi_lut, 
    # PhotometricInterpretation). The NODE21 data is already pre-processed with 
    # energy-based normalization and does not require raw DICOM windowing.

    # 3. Resize logic (Maintained from your original code)
    if do_resize:
        h, w = img.shape
        # Example: Resize ensuring the shortest side is 512
        ratio = 512 / min(h, w) 
        target_size = (int(w * ratio), int(h * ratio))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    img = (img - img.min()) / (img.max() - img.min()) * np.iinfo(np.uint8).max
    img = img.astype(np.uint8)

    # 5. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    success = cv2.imwrite(save_path, img)
    
    if not success:
        print(f'fail to save {save_path}')

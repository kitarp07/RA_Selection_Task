import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

def load_nifti(path_to_data):
    """
    Load a NIfTI file and return the volume data and affine.
    
    Parameters:
        path (str): Path to the .nii or .nii.gz file

    Returns:
        tuple: (3D numpy array, affine matrix)
    """
    data = nib.load(path_to_data)
    
    return data

def get_img_array(nifti_img):
    """
    Extract the image data array from a NIfTI image.

    Parameters:
        nifti_img (nib.Nifti1Image): Loaded NIfTI image

    Returns:
        np.ndarray: 3D volume data as a NumPy array
    """
    vol_data = np.array(nifti_img.get_fdata())
    return vol_data

def plot_slices(data):
    """
    Plot axial, sagittal, and coronal slices of a 3D volume.

    Parameters:
        data (np.ndarray): 3D volume data (e.g., CT or mask)

    Notes:
        - Slice indices are hardcoded for demonstration.
        - Axial:     data[:, :, z]
        - Sagittal:  data[x, :, :]
        - Coronal:   data[:, y, :]
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice 
    axes[0].imshow(data[:, :, 108], cmap='gray')
    axes[0].set_title('Axial Slice (Index 108)')
    axes[0].axis('off')

    # Coronal slice
    axes[1].imshow(data[375, :, :], cmap='gray')
    axes[1].set_title('Saggital Slice (Index 375)')
    axes[1].axis('off')
    
    # Sagittal slice
    axes[2].imshow(data[:, 256, :], cmap='gray')
    axes[2].set_title('Coronal Slice (Index 256)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    


def save_file(final_mask, original_image_vol, file_name):
    labelled_mask = final_mask.astype(np.uint8)  # NIfTI format expects int types
    nii_mask = nib.Nifti1Image(labelled_mask, affine=original_image_vol.affine)
    nib.save(nii_mask, Path("results")/file_name)

    
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt


def get_medial_lateral_lowest_points(path_to_tibia_data):
    """
    Given a file path to a tibia segmentation mask (.nii), 
    returns the voxel coordinates of the medial and lateral lowest points 
    on the tibial plateau surface.

    Parameters:
    - tibia_nii_path: str, file path to the tibia segmentation mask (.nii)

    Returns:
    - medial_lowest_point: np.ndarray, shape (3,), voxel (x,y,z)
    - lateral_lowest_point: np.ndarray, shape (3,), voxel (x,y,z)
    """
    # Load mask and convert to boolean
    img = nib.load(path_to_tibia_data)
    mask = img.get_fdata().astype(bool)

    # Step 1: Extract top surface voxels (max Z per (X,Y))
    top_surface_z = np.full(mask.shape[:2], -1, dtype=int)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            z_indices = np.where(mask[x, y, :])[0]
            if z_indices.size > 0:
                top_surface_z[x, y] = z_indices.max()

    valid_coords = np.where(top_surface_z != -1)
    top_surface_points = np.vstack((valid_coords[0], valid_coords[1], top_surface_z[valid_coords])).T

    # Step 2: Split into medial and lateral groups 
    median_x = np.median(top_surface_points[:, 0])
    medial_points = top_surface_points[top_surface_points[:, 0] < median_x]
    lateral_points = top_surface_points[top_surface_points[:, 0] >= median_x]

    # Step 3: Find lowest point (min Z) in each group
    vertical_axis = 2
    medial_lowest_point = medial_points[np.argmin(medial_points[:, vertical_axis])]
    lateral_lowest_point = lateral_points[np.argmin(lateral_points[:, vertical_axis])]
    
     # Convert voxel coordinates to world coordinates
    medial_lowest_point_world = nib.affines.apply_affine(img.affine, medial_lowest_point)
    lateral_lowest_point_world = nib.affines.apply_affine(img.affine, lateral_lowest_point)

    return medial_lowest_point_world, lateral_lowest_point_world

def expand_mask_tibia(mask, spacing, expansion_mm):
    """
    Expands a binary mask outward by a given parameter in millimeters.

    Parameters:
    ----------
    mask : np.ndarray (bool or binary int)
        3D binary mask (True/1 = foreground, False/0 = background).
    spacing : tuple of float
        Voxel spacing along each axis (z, y, x), typically from NIfTI header.
    expansion_mm : float
        Distance to expand the mask by, in millimeters.

    Returns:
    -------
    expanded : np.ndarray (bool)
        A new mask where the original region has been expanded outward
        by the specified number of millimeters.
    """
    # Compute distance transform of the background (~mask)
    distance = distance_transform_edt(~mask, sampling=spacing)
    # Include original mask plus points within expansion_mm
    expanded = mask | (distance <= expansion_mm)
    return expanded

def get_voxel_spacing(affine):
    """
    Calculate the voxel spacing (physical size of each voxel) along the x, y, and z axes
    from a given 4x4 affine transformation matrix.

    Parameters:
    affine (numpy.ndarray): A 4x4 affine matrix that maps voxel coordinates to world coordinates.

    Returns:
    tuple: Voxel spacing along the x, y, and z axes (spacing_x, spacing_y, spacing_z).
    """
    spacing_x = np.linalg.norm(affine[:3, 0])
    spacing_y = np.linalg.norm(affine[:3, 1])
    spacing_z = np.linalg.norm(affine[:3, 2])
    return spacing_x, spacing_y, spacing_z

def get_expanded_mask_tibia(mask, affine, expansion_mm):
    """
    Expand the femur and tibia regions in the input segmentation mask by a specified margin.
    Parameters:
    mask (numpy.ndarray): 3D segmentation mask with integer labels where
                          1 represents femur, 2 represents tibia, and 0 background.
    affine (numpy.ndarray): 4x4 affine matrix to extract voxel spacing for physical scaling.

    Returns:
    numpy.ndarray: Expanded mask with the same shape as input mask,
                   where femur and tibia regions have been grown,
                   and any overlap is resolved by giving priority to the femur.
    """

    
    spacing = get_voxel_spacing(affine)
    
    tibia_expanded = expand_mask_tibia(mask, spacing, expansion_mm)
    
    expanded_mask = np.zeros_like(mask)
    expanded_mask[tibia_expanded] = 2
    
    return expanded_mask

def randomized_contour_adjustment(original_mask, spacing, expansion_limit, seed):
    """
    Randomly expand the contour of a binary mask by varying amounts per voxel,
    introducing random variation in the mask boundary.

    Parameters:
    original_mask (numpy.ndarray): Boolean array where True indicates the object mask.
    spacing (tuple): Physical voxel spacing along each axis (e.g., in mm).
    expansion_limit (float): Maximum expansion distance in physical units (mm).
    seed (int or None): Seed for random number generator to ensure reproducibility.

    Returns:
    numpy.ndarray: New boolean mask expanded randomly at the boundaries.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # distance (in mm) from each background voxel to original mask
    # distance[i,j,k] = min distance (mm) to a True voxel in original_mask
    distance = distance_transform_edt(
        ~original_mask,
        sampling=spacing
    )
    
    # draw a random expansion radius for *every* voxel in the volume
    #  range [0, expansion_mm]
    random_offsets = np.random.rand(*original_mask.shape) * expansion_limit
    
    # build the new mask:
    #    - keep all original voxels
    #    - add background voxels only if distance <= that voxel’s random offset
    randomized = original_mask | (
        (distance <= random_offsets) &
        (~original_mask)
    )
    
    return randomized

def get_randomized_mask_tibia(mask, affine, expansion_limit, seed):
    """
    Generate randomized expanded masks for femur and tibia regions by applying
    random contour expansions independently to each label.

    Parameters:
    mask (numpy.ndarray): Integer-labeled 3D mask where
                          0 = background, 1 = femur, 2 = tibia.
    affine (numpy.ndarray): 4x4 affine matrix to extract voxel spacing.
    expansion_limit (float): Maximum expansion distance in millimeters.
    seed (int or None): Seed for random number generator for reproducibility.

    Returns:
    numpy.ndarray: New integer mask with randomized expansions applied to femur and tibia.
    """
   
    spacing = get_voxel_spacing(affine)
    
    # assume you already have:
    #   mask          : int32 ndarray with labels {0,1,2}
    #   spacing       : (0.89, 0.89, 2.0)
    #   expansion_mm  : 2.0

    
    # First randomized adjustment (seed=42)
    tibia_rand1 = randomized_contour_adjustment(mask, spacing, expansion_limit, seed)
    
    # Combine into labeled masks
    rand_mask = np.zeros_like(mask, dtype=np.int32)
    rand_mask[tibia_rand1] = 2
    
    return rand_mask

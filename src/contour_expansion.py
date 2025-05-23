import numpy as np
from scipy.ndimage import distance_transform_edt

def expand_mask(mask, spacing, expansion_mm):
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

def get_expanded_mask(mask, affine, expansion_mm):
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
    # Separate femur and tibia masks
    femur_mask = (mask == 1)
    tibia_mask = (mask == 2)
    
    spacing = get_voxel_spacing(affine)
    
    femur_expanded = expand_mask(femur_mask, spacing, expansion_mm)
    tibia_expanded = expand_mask(tibia_mask, spacing, expansion_mm)

    overlap = femur_expanded & tibia_expanded
    # Remove overlap from tibia or assign priority
    tibia_expanded[overlap] = False

    expanded_mask = np.zeros_like(mask)
    expanded_mask[femur_expanded] = 1
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

def get_randomized_masks(mask, affine, expansion_limit, seed):
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
    femur_mask = (mask == 1)
    tibia_mask = (mask == 2)
    
    spacing = get_voxel_spacing(affine)
    
    # assume you already have:
    #   mask          : int32 ndarray with labels {0,1,2}
    #   spacing       : (0.89, 0.89, 2.0)
    #   expansion_mm  : 2.0

    
    # First randomized adjustment (seed=42)
    femur_rand1 = randomized_contour_adjustment(femur_mask, spacing, expansion_limit, seed)
    tibia_rand1 = randomized_contour_adjustment(tibia_mask, spacing, expansion_limit, seed)
    
    # Combine into labeled masks
    rand_mask = np.zeros_like(mask, dtype=np.int32)
    rand_mask[femur_rand1] = 1
    rand_mask[tibia_rand1] = 2
    
    return rand_mask

        
        
    



    

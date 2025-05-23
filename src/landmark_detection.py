import numpy as np
import nibabel as nib


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
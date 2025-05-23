from src.segmentation import (preprocess_mask, threshold_ct, fill_hole_component_1, connect_edges_component_2, 
                              fill_and_smooth, get_largest_components, get_labelled_mask, combine_mask)
from src.utils import load_nifti, save_file, plot_slices, get_img_array
from src.contour_expansion import expand_mask, get_expanded_mask, get_voxel_spacing, get_randomized_masks
from src.landmark_detection import get_medial_lateral_lowest_points, get_expanded_mask_tibia, get_randomized_mask_tibia


def main():
    path_to_data = "./3702_left_knee.nii/3702_left_knee.nii"
    
    #load data
    ct_vol = load_nifti(path_to_data)
    ct_images_vol_data = get_img_array(ct_vol)
    
    #TASK 1
    #thresholding to get bone mask
    bone_mask = threshold_ct(ct_images_vol_data)
        
    #preprocess bone mask to fill holes, remove noise and disconnect femur and tibia
    bone_mask_preprocessed = preprocess_mask(bone_mask)
    
    # extract two largest components from mask and separate them for further preprocessing
    component_1, component_2 = get_largest_components(bone_mask_preprocessed)
    

    # fill internal holes in the femur
    component_1_filled = fill_hole_component_1(component_1)
    
    #connect edges in broken rim of tibia
    component_2_filled = connect_edges_component_2(component_2)

    
    # fill holes and smooth the rough edges
    smoothed_component_2 = fill_and_smooth(component_2_filled)

    # combine the preprocessed separate components into 1 mask
    combined_mask = combine_mask(component_1_filled, smoothed_component_2)

    # label the components and get final mask
    final_mask = get_labelled_mask(combined_mask)
    # plot_slices(final_mask)
    #save mask to results folder
    save_file(final_mask, ct_vol, "task_1_femur_tibia_mask.nii.gz")
    print("Task 1 complete: Segmentation mask saved to results/task_1_femur_tibia_mask.nii.gz")
    
    #TASK 1 COMPLETE
    #------------------------------------------------------------------
    #TASK 2 - Mask Expansion
    
    #load femur_tibia_mask
    mask_vol = load_nifti("./results/task_1_femur_tibia_mask.nii.gz")
    mask_vol_data = get_img_array(mask_vol)
    affine = mask_vol.affine
    
    task_2_expanded_mask = get_expanded_mask(mask_vol_data, affine,expansion_mm= 2)
    
    # plot_slices(task_2_expanded_mask)
    save_file(task_2_expanded_mask, mask_vol, "task_2_tibia_mask.nii.gz")
    print("Task 2 complete: Segmentation mask saved to results/task_2_tibia_mask.nii.gz")
    
    #---------------------------------------------------------------------------
    
    # Task 3 - Randomized Contour Adjustment
    randomized_mask = get_randomized_masks(mask_vol_data, affine,expansion_limit = 2, seed=42)
    # plot_slices(randomized_mask)
    save_file(randomized_mask, mask_vol, "task_3_randomized_mask_expansion.nii.gz")
    print("Task 3 complete: Segmentation mask saved to results/task_3_randomized_mask_expansion.nii.gz")
    
    #-----------------------------------------------------------------------------
    
    # Task 4 
    
    # data already loaded in task 2
    
    #generate tibia mask
    tibia_mask = (mask_vol_data ==2)
    # plot_slices(tibia_mask)
    save_file(tibia_mask, mask_vol, "task_4_tibia_mask.nii.gz")
    print("Tibia Mask Saved: Segmentation mask saved to results/task_4_tibia_mask.nii.gz")
    
    #generate tibia mask expanded 2mm
    task_4_expanded_mask_2mm = get_expanded_mask_tibia(tibia_mask, affine,expansion_mm= 2)
    # plot_slices(task_4_expanded_mask_2mm)
    save_file(task_4_expanded_mask_2mm, mask_vol, "task_4_tibia_mask_expanded_2mm.nii.gz")
    print("Tibia Mask Expanded 2mm Saved: Segmentation mask saved to results/task_4_tibia_mask_expanded_2mm.nii.gz")
    
    #generate tibia mask expanded 4mm
    task_4_expanded_mask_4mm = get_expanded_mask_tibia(tibia_mask, affine,expansion_mm= 4)
    # plot_slices(task_4_expanded_mask_4mm)
    save_file(task_4_expanded_mask_4mm, mask_vol, "task_4_tibia_mask_expanded_4mm.nii.gz")
    print("Tibia Mask Expanded 4mm Saved: Segmentation mask saved to results/task_4_tibia_mask_expanded_4mm.nii.gz")
    
    #generate tibia mask randomized 1
    task_4_randomized_mask_1 = get_randomized_mask_tibia(tibia_mask, affine,expansion_limit = 2, seed=42)
    # plot_slices(task_4_randomized_mask_1)
    save_file(task_4_randomized_mask_1, mask_vol, "task_4_randomized_mask_1.nii.gz")
    print("Randomized Mask 1 Saved: Segmentation mask saved to results/task_4_randomized_mask_1.nii.gz")
    
    #generate tibia mask randomized 2
    task_4_randomized_mask_2 = get_randomized_mask_tibia(tibia_mask, affine,expansion_limit = 2, seed=99)
    # plot_slices(task_4_randomized_mask_2)
    save_file(task_4_randomized_mask_2, mask_vol, "task_4_randomized_mask_2.nii.gz")
    print("Randomized Mask 2 Saved: Segmentation mask saved to results/task_4_randomized_mask_2.nii.gz")
    
    #Landmark Detection
    medial_lowest_1, lateral_lowest_1 = get_medial_lateral_lowest_points("./results/task_4_tibia_mask.nii.gz")
    print("Medial lowest (Tibia Mask):", medial_lowest_1)
    print("Lateral lowest (Tibia Mask):", lateral_lowest_1)
    
    medial_lowest_2, lateral_lowest_2 = get_medial_lateral_lowest_points("./results/task_4_tibia_mask_expanded_2mm.nii.gz")
    print("Medial lowest  (Tibia Mask 2mm expanded):", medial_lowest_2)
    print("Lateral lowest  (Tibia Mask 2mm expanded):", lateral_lowest_2)
    
    medial_lowest_3, lateral_lowest_3 = get_medial_lateral_lowest_points("./results/task_4_tibia_mask_expanded_4mm.nii.gz")
    print("Medial lowest (Tibia Mask 4mm expanded):", medial_lowest_3)
    print("Lateral lowest (Tibia Mask 4mm expanded):", lateral_lowest_3)
    
    medial_lowest_4, lateral_lowest_4 = get_medial_lateral_lowest_points("./results/task_4_randomized_mask_1.nii.gz")
    print("Medial lowest (Tibia Mask Randomized mask 1):", medial_lowest_4)
    print("Lateral lowest (Tibia Mask Randomized mask 1):", lateral_lowest_4)
    
    medial_lowest_5, lateral_lowest_5 = get_medial_lateral_lowest_points("./results/task_4_randomized_mask_2.nii.gz")
    print("Medial lowest (Tibia Mask Randomized mask 2):", medial_lowest_5)
    print("Lateral lowest (Tibia Mask Randomized mask 2):", lateral_lowest_5)
    
    
    
if __name__ == "__main__":
    main()
    
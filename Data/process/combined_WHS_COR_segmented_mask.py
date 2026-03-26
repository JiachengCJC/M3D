import nibabel as nib
import numpy as np
import os

# Define the root directory
base_dir = 'output'

# Iterate through every 'case' folder
for case_name in os.listdir(base_dir):
    case_path = os.path.join(base_dir, case_name)
    
    if os.path.isdir(case_path):
        # Iterate through every 'cube_index' folder
        for cube_name in os.listdir(case_path):
            cube_path = os.path.join(case_path, cube_name)
            
            if os.path.isdir(cube_path):
                # Define file paths
                mask1_path = os.path.join(cube_path, 'WHS.nii.gz')
                mask2_path = os.path.join(cube_path, 'cor_seg.nii.gz')
                output_path = os.path.join(cube_path, 'combined_mask.nii.gz')

                # Check if both files exist before processing
                if os.path.exists(mask1_path) and os.path.exists(mask2_path):
                    print(f"Processing: {case_name} -> {cube_name}")
                    
                    # 1. Load the original masks
                    img1 = nib.load(mask1_path)
                    img2 = nib.load(mask2_path)

                    data1 = img1.get_fdata()
                    data2 = img2.get_fdata()

                    # 2. Shift labels for mask2 (cor_seg becomes Label 2)
                    data2_shifted = np.where(data2 > 0, data2 + np.max(data1), 0)
                    
                    # 3. Combine them (Overlaps will become Label 3)
                    combined_data = data1 + data2_shifted

                    # 4. Save back to the same folder
                    combined_img = nib.Nifti1Image(
                        combined_data.astype(np.int16), 
                        img1.affine, 
                        img1.header
                    )
                    nib.save(combined_img, output_path)
                else:
                    print(f"Skipping {cube_path}: Missing one or both .nii.gz files")
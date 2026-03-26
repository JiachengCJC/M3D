import nibabel as nib
import numpy as np

mask1 = ''
mask2 = ''

# 1. Load the original masks
img1 = nib.load(mask1)
img2 = nib.load(mask2)

data1 = img1.get_fdata()
data2 = img2.get_fdata()

# 2. Shift labels for mask2 (assuming mask1 uses label 1)
# This makes mask2 use label 2 instead of 1
data2_shifted = np.where(data2 > 0, 2, 0)

# 3. Combine them
# Note: If they overlap, this will create a label '3' (1+2)
combined_data = data1 + data2_shifted

# 4. Save back to .nii.gz
# CRITICAL: We use img1.affine to keep the spatial orientation correct
combined_img = nib.Nifti1Image(combined_data.astype(np.int16), img1.affine, img1.header)
nib.save(combined_img, 'combined_mask.nii.gz')
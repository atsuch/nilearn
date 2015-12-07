"""
Simple example of NiftiMasker and HemisphereMasker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.
"""

### Load nyu_rest dataset #####################################################

from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.input_data.hemisphere_masker import HemisphereMasker
nyu_dataset = datasets.fetch_nyu_rest(n_subjects=1)

# print basic information on the dataset
print('First anatomical nifti image (3D) is at: %s' % nyu_dataset.anat_anon[0])
print('First functional nifti image (4D) is at: %s' %
      nyu_dataset.func[0])  # 4D data

### Compute the mask ##########################################################

# As this is raw resting-state EPI, the background is noisy and we cannot
# rely on the 'background' masking strategy. We need to use the 'epi' one
nifti_masker = NiftiMasker(standardize=False, mask_strategy='epi',
                           memory="nilearn_cache", memory_level=2)
func_filename = nyu_dataset.func[0]
nifti_masker.fit(func_filename)
mask_img = nifti_masker.mask_img_

# Test HemisphereMasker
r_masker = HemisphereMasker(standardize=False, mask_strategy='epi',
                           memory="nilearn_cache", memory_level=2, hemisphere ="R")
r_masker.fit(func_filename)
r_mask = r_masker.mask_img_ 

l_masker = HemisphereMasker(standardize=False, mask_strategy='epi',
                           memory="nilearn_cache", memory_level=2)
l_masker.fit(func_filename)
l_mask = l_masker.mask_img_ 



### Visualize the mask ########################################################
from nilearn.plotting import plot_roi, show
from nilearn.image.image import mean_img

# calculate mean image for the background
mean_func_img = mean_img(func_filename)

plot_roi(mask_img, mean_func_img, display_mode='y', cut_coords=4, title="Mask")

plot_roi(r_mask, mean_func_img, display_mode='y', cut_coords=4, title="R Mask")
plot_roi(l_mask, mean_func_img, display_mode='y', cut_coords=4, title="L Mask")


### Preprocess data ###########################################################
nifti_masker.fit(func_filename)
fmri_masked = nifti_masker.transform(func_filename)

fmri_r_masked = r_masker.transform(func_filename)
fmri_l_masked = l_masker.transform(func_filename)
### Run an algorithm ##########################################################
from sklearn.decomposition import FastICA
n_components = 20
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(fmri_masked.T).T

r_components_masked = ica.fit_transform(fmri_r_masked.T).T
l_components_masked = ica.fit_transform(fmri_l_masked.T).T

### Reverse masking ###########################################################
components = nifti_masker.inverse_transform(components_masked)

r_components = r_masker.inverse_transform(r_components_masked)
l_components = l_masker.inverse_transform(l_components_masked)

### Show results ##############################################################
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

plot_stat_map(index_img(components, 0), mean_func_img,
              display_mode='y', cut_coords=4, title="Component 0")
              
plot_stat_map(index_img(r_components, 0), mean_func_img,
              display_mode='y', cut_coords=4, title="R Component 0")
              
plot_stat_map(index_img(l_components, 0), mean_func_img,
              display_mode='y', cut_coords=4, title="L Component 0")

show()

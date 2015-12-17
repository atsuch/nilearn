"""
NeuroVault cross-study ICA maps.
================================

This example shows how to download statistical maps from
NeuroVault, label them with NeuroSynth terms,
and compute ICA components across all the maps.

See :func:`nilearn.datasets.fetch_neurovault` documentation for more details.
"""
# Author: Ben Cipollini
# License: BSD
# Ported from code authored by Chris Filo Gorgolewski, Gael Varoquaux
# https://github.com/NeuroVault/neurovault_analysis

import os.path as op
import warnings
warnings.simplefilter('error', RuntimeWarning)  # Catch numeric issues in imgs

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.decomposition import FastICA

from nilearn import datasets
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir
from nilearn.image import new_img_like, resample_img, 
from nilearn.input_data import NiftiMasker
from nilearn.input_data.hemisphere_masker import HemisphereMasker
from nilearn._utils import check_niimg
from nilearn.plotting import plot_stat_map


def clean_img(img):
    """ Remove nan/inf entries."""
    img = check_niimg(img)
    img_data = img.get_data()
    img_data[np.isnan(img_data)] = 0
    img_data[np.isinf(img_data)] = 0
    return new_img_like(img, img_data)

def cast_img(img, dtype=np.float32):
    """ Cast image to the specified dtype"""
    img = check_niimg(img)
    img_data = img.get_data().astype(dtype)
    return new_img_like(img, img_data)

### Get image and term data ###################################################
# Download 100 matching images
ss_all = datasets.fetch_neurovault(max_images=100,  # Use np.inf for all imgs.
                                   map_types=['F map', 'T map', 'Z map'],
                                   fetch_terms=True)
images, collections = ss_all['images'], ss_all['collections']
term_scores = ss_all['terms']

# Clean & report term scores
terms = np.array(term_scores.keys())
term_matrix = np.asarray(term_scores.values())
term_matrix[term_matrix < 0] = 0
total_scores = np.mean(term_matrix, axis=1)

print("Top 10 neurosynth terms from downloaded images:")
for term_idx in np.argsort(total_scores)[-10:][::-1]:
    print('\t%-25s: %.2f' % (terms[term_idx], total_scores[term_idx]))

print("Creating a grey matter mask.")
target_img = datasets.load_mni152_template()
grey_voxels = (target_img.get_data() > 0).astype(int)
mask_img = new_img_like(target_img, grey_voxels)

### Reshape & mask images #####################################################
print("Reshaping and masking images.")
masker = NiftiMasker(mask_img=mask_img, target_affine=target_img.affine,
                     target_shape=target_img.shape, memory='nilearn_cache')
masker = masker.fit()

# R and L maskers
r_masker = HemisphereMasker(mask_img=mask_img, target_affine=target_img.affine,
                           target_shape=target_img.shape, memory='nilearn_cache',
                           hemisphere = 'R')
l_masker = HemisphereMasker(mask_img=mask_img, target_affine=target_img.affine,
                           target_shape=target_img.shape, memory='nilearn_cache',
                           hemisphere = 'L')
r_masker = r_masker.fit()
l_masker = l_masker.fit()


# Images may fail to be transformed, and are of different shapes,
# so we need to trasnform one-by-one and keep track of failures.
X, r_X, l_X = [], [], []
xformable_idx = np.ones((len(images),), dtype=bool)
for ii, im in enumerate(images):
    img = cast_img(im['local_path'], dtype=np.float32)
    img = clean_img(img)
    try:
        X.append(masker.transform(img))
        r_X.append(r_masker.transform(img))
        l_X.append(l_masker.transform(img))
    except Exception as e:
        print("Failed to mask/reshape image %d/%s: %s" % (
            im.get('collection_id', 0), op.basename(im['local_path']), e))
        xformable_idx[ii] = False

# Now reshape list into 2D matrix, and remove failed images from terms
X = np.vstack(X)
term_matrix = term_matrix[:, xformable_idx]

# Reshape R and L masked lists to matrices
r_X, l_X = np.vstack(r_X), np.vstack(l_X)

### Run ICA and map components to terms #######################################
print("Running ICA; may take time...")
fast_ica = FastICA(n_components=20, random_state=42)
ica_maps = fast_ica.fit_transform(X.T).T

# Don't use the transform method as it centers the data
ica_terms = np.dot(term_matrix, fast_ica.components_.T).T

# Repeat ICA for R and L hemispheres
r_fast_ica = FastICA(n_components=20, random_state=42)
r_ica_maps = r_fast_ica.fit_transform(r_X.T).T

l_fast_ica = FastICA(n_components=20, random_state=42)
l_ica_maps = l_fast_ica.fit_transform(l_X.T).T

# Create separate terms for R and L
r_ica_terms = np.dot(term_matrix, r_fast_ica.components_.T).T
l_ica_terms = np.dot(term_matrix, l_fast_ica.components_.T).T


### Generate figures ##########################################################

hemi_dict = {"both":[ica_maps, ica_terms, masker],
            "R":[r_ica_maps, r_ica_terms, r_masker],
            "L":[l_ica_maps, l_ica_terms, l_masker]}
            
for hemi in ["both","R","L"]:
    
    hemi_masker = hemi_dict[hemi][2]
    ica_images = hemi_masker.inverse_transform(hemi_dict[hemi][0]) 
    ica_images.to_filename('ica_neurovault_RL/%s_ica_components.nii.gz'% hemi)
            
    for idx, (ic, ic_terms) in enumerate(zip(hemi_dict[hemi][0], hemi_dict[hemi][1])):
        if -ic.min() > ic.max():
            # Flip the map's sign for prettiness
            ic = -ic
            ic_terms = -ic_terms

        ic_thr = stats.scoreatpercentile(np.abs(ic), 90)
        ic_img = hemi_masker.inverse_transform(ic)
        display = plot_stat_map(ic_img, threshold=ic_thr, colorbar=False,
                            bg_img=target_img)

        # Use the 4 terms weighted most as a title
        important_terms = terms[np.argsort(ic_terms)[-4:]]
        title = '%d: %s' % (idx, ', '.join(important_terms[::-1]))
        display.title(title, size=16)
        
        # Save images instead of displaying
        plt.savefig('ica_neurovault_RL/ica_maps/%s_component_%i.png'%(hemi, idx))
        plt.close()



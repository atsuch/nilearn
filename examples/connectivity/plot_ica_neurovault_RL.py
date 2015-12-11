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

import json
import io
import os
import warnings
warnings.simplefilter('error', RuntimeWarning)

import numpy as np
from joblib import Memory
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import DictVectorizer

import nibabel as nib
from nilearn import datasets
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir
from nilearn.image import resample_img
from nilearn.input_data import NiftiMasker
from nilearn.input_data.hemisphere_masker import HemisphereMasker
from nilearn.masking import compute_background_mask, _extrapolate_out_mask
from nilearn.plotting import plot_stat_map


# Use for caching results to disk, so running twice will be *fast*
mem = Memory('cache', verbose=0)


@mem.cache
def splitext_multi(p):
    """From my_img.nii.gz, return .nii.gz"""
    dirname, basename = os.path.dirname(p), os.path.basename(p)
    stem, ext = basename.split('.', 1)
    return os.path.join(dirname, stem), '.' + ext if ext else ''


@mem.cache
def resample_and_expand_image(src_img, target_img, out_path=None):
    """Resample src_img to target_img affine, turn 4D into list of 3D.

    Optionally save to out_path.
    """
    data = src_img.get_data().squeeze()
    data[np.isnan(data)] = 0
    assert np.abs(data).max() > 0

    if out_path is not None:
        # Create the proper output filenames (if 3D, just one;
        #   if 4D then one per time point)
        out_stem, out_ext = splitext_multi(out_path)
        out_paths = [out_path] if len(data.shape) == 3 else \
                    ['%s-%d%s' % (out_stem, ii, out_ext)
                     for ii in range(data.shape[-1])]
        # If all output images exist, use them cached from disk.
        if np.all([os.path.exists(p) for p in out_paths]):
            return [nib.load(p) for p in out_paths]

    # Copy the image and make a background mask.
    src_img = nib.Nifti1Image(data, src_img.get_affine(),
                              header=src_img.get_header())
    bg_mask = compute_background_mask(src_img).get_data()

    # Test if the image has been masked:
    out_of_mask = data[np.logical_not(bg_mask)]
    if np.all(np.isnan(out_of_mask)) or len(np.unique(out_of_mask)) == 1:
        # Need to extrapolate
        with warnings.catch_warnings(record=False):
            warnings.simplefilter('ignore', RuntimeWarning)
            data, _ = _extrapolate_out_mask(data.astype(np.float), bg_mask,
                                            iterations=3)
    src_img = nib.Nifti1Image(data, src_img.get_affine(),
                              header=src_img.get_header())
    del out_of_mask, bg_mask

    # Resampling the file to target and saving the output in the "resampled"
    # folder
    resampled_nii = resample_img(src_img, target_img.get_affine(),
                                 target_img.shape)
    resampled_nii = nib.Nifti1Image(resampled_nii.get_data().squeeze(),
                                    resampled_nii.get_affine(),
                                    header=src_img.get_header())

    # Decimate 4D files.
    if len(resampled_nii.shape) == 3:
        resampled_niis = [resampled_nii]
    else:
        resampled_niis = []
        resampled_data = resampled_nii.get_data()
        for index in range(resampled_nii.shape[3]):
            # First save the files separately
            this_nii = nib.Nifti1Image(resampled_data[..., index],
                                       resampled_nii.get_affine())
            resampled_niis.append(this_nii)

    # Save the result
    if out_path:
        for nii, nii_path in zip(resampled_niis, out_paths):
            nib.save(nii, nii_path)

    return resampled_niis


@mem.cache
def get_neurosynth_terms(images, data_dir, print_frequency=100):
    """ Grab terms for each image, decoded with neurosynth"""

    terms = list()
    vectorizer = DictVectorizer()
    for ii, img in enumerate(images):
        if ii % print_frequency == 0:
            print("Fetching terms for images %d-%d of %d" % (
                ii + 1, min(ii + print_frequency, len(images)), len(images)))

        url = 'http://neurosynth.org/api/v2/decode/?neurovault=%d' % img['id']
        fil = 'terms-for-image-%d.json' % img['id']
        elevations = _fetch_files(data_dir, ((fil, url, {'move': fil}),),
                                  verbose=2)[0]

        try:
            with io.open(elevations, 'r', encoding='utf8') as fp:
                data = json.load(fp)['data']
        except Exception as e:
            if os.path.exists(elevations):
                os.remove(elevations)
            terms.append({})
        else:
            data = data['values']
            terms.append(data)
    X = vectorizer.fit_transform(terms).toarray()
    return dict([(name, X[:, idx])
                 for name, idx in vectorizer.vocabulary_.items()])



# -------------------------------------------
# Define pre-download filters
bad_collects = [16]
bad_image_ids = [
    96, 97, 98,                    # The following maps are not brain maps
    338, 339,                      # And the following are crap
    335,                           # 335 is a duplicate of 336
    3360, 3362, 3364,              # These are mean images, and not Z maps
    1202, 1163, 1931, 1101, 1099]  # Ugly / obviously not Z maps
imfilts = [lambda im: im.get('perc_bad_voxels', 0.) < 100.]

# Download 100 matching images
ss_all = datasets.fetch_neurovault(max_images=300,  # Use np.inf for all imgs.
                                   collection_ids=[-bid for bid in bad_collects],
                                   image_ids=[-bid for bid in bad_image_ids],
                                   map_types=['F map', 'T map', 'Z map'],
                                   image_filters=imfilts)
images, collections = ss_all['images'], ss_all['collections']

# -------------------------------------------
# Resample the images
target_img = datasets.load_mni152_template()
resampled_niis = []
for ii, image in enumerate(images):
    if ii % 100 == 0:
        print("Resampling image %d-%d of %d..." % (
            ii + 1, min(ii + 100, len(images)), len(images)))
    src_img = nib.load(image['local_path'])
    file_path, ext = splitext_multi(image['local_path'])
    resample_path = '%s-resampled%s' % (file_path, ext)
    resampled_niis += resample_and_expand_image(src_img, target_img,
                                                resample_path)
if len(images) != len(resampled_niis):
    print("After resampling, %d images => %d "
          "(each time point became an image)" % (
              len(images), len(resampled_niis)))

# -------------------------------------------
# Get the neurosynth terms; returns a dict of terms, with
#   a vector of values (1 per image)
terms = get_neurosynth_terms(images, _get_dataset_dir('neurosynth'))
good_terms = dict([(t, v) for t, v in terms.items()
                   if np.sum(v[v > 0]) > 0.])

print("Top 100 neurosynth terms:")
sort_idx = np.argsort([np.sum(v[v > 0]) for v in good_terms.values()])
for term_idx in sort_idx[-100:]:
    # Eliminate negative values
    term = list(good_terms.keys())[term_idx]
    vec = good_terms[term]
    vec[vec < 0] = 0
    print('\t%-25s: %.4e' % (term, np.sum(vec)))

# -------------------------------------------
# Get the grey matter mask. Cheat by pulling from github :)
print("Downloading and resampling grey matter mask.")
url = 'https://github.com/NeuroVault/neurovault_analysis/raw/master/gm_mask.nii.gz'
mask = _fetch_files(_get_dataset_dir('neurovault'),
                    (('gm_mask.nii.gz', url, {}),))[0]
mask = resample_and_expand_image(nib.load(mask), target_img)[0]
mask = nib.Nifti1Image((mask.get_data() >= 0.5).astype(int),
                       mask.get_affine(), mask.get_header())

# -------------------------------------------
# Do the ICA transform.
b_masker = NiftiMasker(mask_img=mask, memory=mem)
X = b_masker.fit_transform(resampled_niis)

# R and L maskers
r_masker = HemisphereMasker(mask_img = mask, memory = mem, hemisphere = "R")
l_masker = HemisphereMasker(mask_img = mask, memory = mem, hemisphere = "L")
r_X = r_masker.fit_transform(resampled_niis)
l_X = l_masker.fit_transform(resampled_niis)

print("Running ICA; may take time...")
fast_ica = FastICA(n_components=20, random_state=42)
ica_maps = fast_ica.fit_transform(X.T).T

r_fast_ica = FastICA(n_components=20, random_state=42)
r_ica_maps = r_fast_ica.fit_transform(r_X.T).T

l_fast_ica = FastICA(n_components=20, random_state=42)
l_ica_maps = l_fast_ica.fit_transform(l_X.T).T

# -------------------------------------------
# Relate ICA to terms
term_matrix = np.asarray(list(good_terms.values()))
# Don't use the transform method as it centers the data
ica_terms = np.dot(term_matrix, fast_ica.components_.T).T

# Create separate terms for R and L
r_ica_terms = np.dot(term_matrix, r_fast_ica.components_.T).T
l_ica_terms = np.dot(term_matrix, l_fast_ica.components_.T).T

col_names = list(good_terms.keys())

# -------------------------------------------
# Generate figures and save files

hemi_dict = {"both":[ica_maps, ica_terms, b_masker],
            "R":[r_ica_maps, r_ica_terms, r_masker],
            "L":[l_ica_maps, l_ica_terms, l_masker]}
            
for hemi in ["both","R","L"]:
    
    masker = hemi_dict[hemi][2]
    ica_images = masker.inverse_transform(hemi_dict[hemi][0]) 
    ica_images.to_filename('ica_neurovault_RL/%s_ica_components'% hemi)
    
    for idx, (ic, ic_terms) in enumerate(zip(hemi_dict[hemi][0], hemi_dict[hemi][1])):
        if -ic.min() > ic.max():
            # Flip the map's sign for prettiness
            ic = -ic
            ic_terms = -ic_terms

        ic_thr = stats.scoreatpercentile(np.abs(ic), 90)
        ic_img = masker.inverse_transform(ic)
        # Use the 4 terms weighted most as a title
        important_terms = np.array(col_names)[np.argsort(ic_terms)[-4:]]
        display = plot_stat_map(ic_img, threshold=ic_thr, colorbar=False,
                               bg_img=target_img)
        display.title(', '.join(important_terms[::-1]), size=16)
        
        # Save images instead of displaying
        plt.savefig('ica_neurovault_RL/ica_maps/%s_component_%i.png'%(hemi, idx))
        plt.close()
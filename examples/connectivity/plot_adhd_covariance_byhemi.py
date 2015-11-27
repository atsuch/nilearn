"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a covariance matrix based on these signals.
"""
n_subjects = 20  # Number of subjects to consider for group-sparse covariance
plotted_subject = 1  # subject to plot
n_jobs = 1

import numpy as np

import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory

import nilearn
from nilearn import plotting
from nilearn.input_data.hemisphere_masker import HemisphereMasker, split_bilateral_rois
from nilearn.plotting import cm
from nilearn.image import index_img, iter_img


def compute_lh_idx(atlas_maps):
    idx = []
    lh_masker = HemisphereMasker(hemisphere='L')
    lh_masker.fit(atlas_maps)
    for img in iter_img(atlas_maps):
        idx.append(lh_masker.transform(img).sum() > 0)
    return np.asarray(idx, dtype=np.bool)

def plot_connectome(cov, atlas_maps, plot_type='all', random_seed=42, **kwargs):
    """Plot connectome given a covariance matrix and atlas maps

    plot_type in ['all', 'lh', 'rh', 'intra', 'inter']

    assumptions: in atlas_maps, even #s are lh, odd # masks are rh"""

    # Select sub-region of covariance matrix and image
    lh_idx = compute_lh_idx(atlas_maps)
    rh_idx = np.logical_not(lh_idx)

    # Choose colors
    np.random.seed(random_seed)
    hemi_node_colors = np.random.rand(len(lh_idx) // 2, 3)
    node_colors = np.empty((len(lh_idx), 3))
    node_colors[lh_idx, :] = hemi_node_colors
    node_colors[rh_idx, :] = hemi_node_colors

    if plot_type == 'intra':
        cov = cov.copy()
        cov[np.ix_(lh_idx, rh_idx)] = 0
        cov[np.ix_(rh_idx, lh_idx)] = 0
    elif plot_type == 'inter':
        cov = cov.copy()
        cov[np.ix_(lh_idx, lh_idx)] = 0
        cov[np.ix_(rh_idx, rh_idx)] = 0

    imgs = iter_img(atlas_maps)
    regions_coords = np.array([
        map(np.asscalar, plotting.find_xyz_cut_coords(img)) for img in imgs])

    # Plot
    plotting.plot_connectome(cov, regions_coords,
                             nodes_kwargs={'s': 50, 'c': node_colors},
                             **kwargs)


def plot_matrices(cov, prec, title):
    """Plot covariance and precision matrices, for a given processing. """

    # Compute sparsity pattern
    sparsity = (prec == 0)

    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[range(size), range(size)] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plt.figure()
    plt.imshow(cov, interpolation="nearest",
               vmin=-1, vmax=1, cmap=cm.bwr)
    plt.colorbar()
    plt.title("%s / covariance" % title)

    # Display sparsity pattern
    plt.figure()
    plt.imshow(sparsity, interpolation="nearest")
    plt.title("%s / sparsity" % title)

    # Display precision matrix
    plt.figure()
    plt.imshow(prec, interpolation="nearest",
               vmin=-span, vmax=span,
               cmap=cm.bwr)
    plt.colorbar()
    plt.title("%s / precision" % title)


# Fetching datasets ###########################################################
print("-- Fetching datasets ...")
from nilearn import datasets
msdl_atlas_dataset = datasets.fetch_msdl_atlas()
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)

# Extracting region signals ###################################################
import nibabel
import nilearn.image
import nilearn.input_data

mem = Memory('nilearn_cache')

maps_img = nibabel.load(msdl_atlas_dataset.maps)
maps_img = split_bilateral_rois(maps_img)

masker = nilearn.input_data.NiftiMapsMasker(
    maps_img, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=0.01, t_r=2.5, standardize=True,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

subjects = []
func_filenames = adhd_dataset.func
confound_filenames = adhd_dataset.confounds
for func_filename, confound_filename in zip(func_filenames,
                                            confound_filenames):
    print("Processing file %s" % func_filename)

    print("-- Computing confounds ...")
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)

    print("-- Computing region signals ...")
    region_ts = masker.transform(func_filename,
                                 confounds=[hv_confounds, confound_filename])
    subjects.append(region_ts)

# Computing group-sparse precision matrices ###################################
print("-- Computing graph-lasso precision matrices ...")
from sklearn import covariance
gl = covariance.GraphLassoCV(n_jobs=n_jobs, verbose=2)
gl.fit(subjects[plotted_subject])

# Displaying results ##########################################################
print("-- Displaying results")

cov = gl.covariance_

fh = plt.figure()
for ai, plot_type in enumerate(['intra', 'inter', 'all']):
    ax = fh.add_subplot(3, 1, ai)
    display = plot_connectome(cov, maps_img,
                              plot_type=plot_type,
                              edges_threshold='70%',
                              title='threshold=70%% [%s]' % plot_type,
                              axes=ax)
plt.show()

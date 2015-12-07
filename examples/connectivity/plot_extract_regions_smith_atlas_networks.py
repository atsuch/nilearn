"""
Regions Extraction of Default Mode Networks using Smith Atlas
=============================================================

This simple example shows how to extract regions from Smith atlas
resting state networks.
"""

################################################################################
# Fetching the smith ICA 10 RSN by importing datasets utilities
from nilearn import datasets

smith_atlas = datasets.fetch_atlas_smith_2009()
atlas_networks = smith_atlas.rsn10

################################################################################
# Import region extractor to extract atlas networks
from nilearn.regions import RegionExtractor

# min_region_size in voxel volume mm^3
extraction = RegionExtractor(atlas_networks, min_region_size=800,
                             threshold=98, thresholding_strategy='percentile')

# Just call fit() to execute region extraction procedure
extraction.fit()
regions_img = extraction.regions_img_

################################################################################
# Visualization
# Show region extraction results by importing image & plotting utilities

from nilearn import plotting
from nilearn.image import iter_img
from nilearn.plotting import find_xyz_cut_coords

# Showing region extraction results using 4D maps visualization tool
plotting.plot_prob_atlas(regions_img, display_mode='z', cut_coords=1,
                         view_type='contours', title="Regions extracted.")

for i, cur_img in zip(extraction.index_, iter_img(regions_img)):
    coords = find_xyz_cut_coords(cur_img)
    plotting.plot_stat_map(cur_img, display_mode='z', cut_coords=coords[2:3],
                           title="Blob of network %d " % i, colorbar=False)

plotting.show()

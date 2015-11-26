""" Test Region Extractor and its functions """

import numpy as np
import nibabel

from nose.tools import assert_raises, assert_equal, assert_true, assert_not_equal

from nilearn.regions import connected_regions, RegionExtractor
from nilearn.regions.region_extractor import _threshold_maps
from nilearn.image import iter_img

from nilearn._utils.testing import assert_raises_regex, generate_maps


def _make_random_data(shape):
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    data_rng = rng.normal(size=shape)
    img = nibabel.Nifti1Image(data_rng, affine)
    data = img.get_data()
    return img, data


def test_invalid_thresholds_in_threshold_maps():
    maps, _ = generate_maps((10, 11, 12), n_regions=2)

    for invalid_threshold in [10, '80%', 'auto', -1.0]:
        assert_raises_regex(ValueError,
                            "threshold given as ratio to the number of voxels must "
                            "be float value and should be positive and between 0 and "
                            "total number of maps i.e. n_maps={0}. "
                            "You provided {1}".format(maps.shape[-1], invalid_threshold),
                            _threshold_maps,
                            maps, threshold=invalid_threshold)


def test_threshold_maps():
    # smoke test for function _threshold_maps with randomly
    # generated maps

    # make sure that n_regions (4th dimension) are kept same even
    # in thresholded image
    maps, _ = generate_maps((6, 8, 10), n_regions=3)
    thr_maps = _threshold_maps(maps, threshold=1.0)
    assert_true(thr_maps.shape[-1] == maps.shape[-1])

    # check that the size should be same for 3D image
    # before and after thresholding
    img = np.zeros((30, 30, 30)) + 0.1 * np.random.randn(30, 30, 30)
    img = nibabel.Nifti1Image(img, affine=np.eye(4))
    thr_maps_3d = _threshold_maps(img, threshold=0.5)
    assert_true(img.shape == thr_maps_3d.shape)


def test_invalids_extract_types_in_connected_regions():
    maps, _ = generate_maps((10, 11, 12), n_regions=2)
    valid_names = ['connected_components', 'local_regions']

    # test whether same error raises as expected when invalid inputs
    # are given to extract_type in connected_regions function
    message = ("'extract_type' should be {0}")
    for invalid_extract_type in ['connect_region', 'local_regios']:
        assert_raises_regex(ValueError,
                            message.format(valid_names),
                            connected_regions,
                            maps, extract_type=invalid_extract_type)


def test_connected_regions():
    # 4D maps
    n_regions = 4
    maps, _ = generate_maps((30, 30, 30), n_regions=n_regions)
    # 3D maps
    map_img = np.zeros((30, 30, 30)) + 0.1 * np.random.randn(30, 30, 30)
    map_img = nibabel.Nifti1Image(map_img, affine=np.eye(4))

    valid_extract_types = ['connected_components', 'local_regions']
    # smoke test for function connected_regions and also to check
    # if the regions extracted should be equal or more than already present.
    # 4D image case
    for extract_type in ['connected_components', 'local_regions']:
        connected_extraction, index = connected_regions(maps, min_region_size=10,
                                                        extract_type=extract_type)
        assert_true(connected_extraction.shape[-1] >= n_regions)
        assert_true(index, np.ndarray)
        # For 3D images regions extracted should be more than equal to one
        connected_extraction_3d, _ = connected_regions(map_img, min_region_size=10,
                                                       extract_type=extract_type)
        assert_true(connected_extraction_3d.shape[-1] >= 1)


def test_invalid_threshold_value_in_regionextractor():
    maps, _ = generate_maps((10, 11, 12), n_regions=1)
    threshold = 10
    extractor = RegionExtractor(maps, threshold=threshold,
                                thresholding_strategy='ratio_n_voxels')
    message = ("threshold should be given as float value "
               "for thresholding_strategy='ratio_n_voxels'. "
               "You provided a value of threshold={0}")
    assert_raises_regex(ValueError,
                        message.format(threshold),
                        extractor.fit)


def test_region_extractor_fit_and_transform():
    n_regions = 9
    n_subjects = 5
    maps, mask_img = generate_maps((40, 40, 40), n_regions=n_regions)

    # smoke test to RegionExtractor with thresholding_strategy='ratio_n_voxels'
    extract_ratio = RegionExtractor(maps, threshold=0.2,
                                    thresholding_strategy='ratio_n_voxels')
    extract_ratio.fit()
    assert_not_equal(extract_ratio.regions_, '')
    assert_true(extract_ratio.regions_.shape[-1] >= 9)

    # smoke test with threshold=string and strategy=percentile
    extractor = RegionExtractor(maps, threshold='30%',
                                thresholding_strategy='percentile',
                                mask_img=mask_img)
    extractor.fit()
    assert_true(extractor.index_, np.ndarray)
    assert_not_equal(extractor.regions_, '')
    assert_true(extractor.regions_.shape[-1] >= 9)

    n_regions_extracted = extractor.regions_.shape[-1]
    imgs = []
    signals = []
    shape = (91, 109, 91, 7)
    expected_signal_shape = (7, n_regions_extracted)
    for id_ in range(n_subjects):
        img, data = _make_random_data(shape)
        # smoke test NiftiMapsMasker transform inherited in Region Extractor
        signal = extractor.transform(img)
        assert_equal(expected_signal_shape, signal.shape)

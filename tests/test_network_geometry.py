import warnings

import numpy as np
import pytest
from gtfs_fixtures import densify, ll

from microsegments.network.geometry import (
    cumlen,
    cut,
    hausdorff_m,
    interpolate,
    length,
    project_monotone,
    shape_slices,
)


def test_out_and_back_loop():
    """Shape goes east then back west on the same street: return stops must land on the return leg."""
    shape = densify([(0, 0), (600, 0), (600, 3), (0, 3)], step=20)
    stops = np.array([ll(x, 1) for x in (0, 300, 600, 300, 0)])
    s, off = project_monotone(stops, shape)
    assert np.allclose(s, [0, 300, 601.5, 903, 1203], atol=2)
    assert off.max() < 3


def test_far_stop_stays_in_order():
    shape = densify([(0, 0), (1000, 0)])
    stops = np.array([ll(0), ll(200), ll(900, 400), ll(500), ll(1000)])   # third stop far off and "ahead"
    s, _ = project_monotone(stops, shape)
    assert np.all(np.diff(s) >= 0)
    assert s[3] == pytest.approx(500, abs=1)


def test_slices_lengths_and_cut():
    shape = densify([(0, 0), (400, 0), (400, 300)], step=50)
    stops = np.array([ll(10, 2), ll(390, -2), ll(398, 150), ll(400, 300)])
    sl = shape_slices(stops, shape)
    assert np.allclose(sl.lengths, [380, 10 + 150, 150], atol=1)
    for c, L in zip(sl.coords, sl.lengths):
        assert length(c) == pytest.approx(L, abs=1e-6)
    c = cut(shape, 100, 450)
    assert length(c) == pytest.approx(350, abs=1e-6)
    p = interpolate(shape, [0, 400, 1e9])
    assert np.allclose(p[1], ll(400, 0)) and np.allclose(p[2], ll(400, 300))
    assert cumlen(shape)[-1] == pytest.approx(700, abs=0.05)


def test_no_shape_or_bad_shape():
    stops = np.array([ll(0), ll(300), ll(300, 400)])
    sl = shape_slices(stops, None)
    assert not sl.from_shape and np.allclose(sl.lengths, [300, 400], atol=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sl = shape_slices(stops, densify([(5000, 5000), (6000, 5000)]))
    assert not sl.from_shape


def test_hausdorff():
    a = densify([(0, 0), (300, 0)])
    b = densify([(0, 10), (300, 10)])
    assert hausdorff_m(a, b) == pytest.approx(10, abs=0.1)

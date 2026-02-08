"""
Regression tests of [cmip_nominal_resolution.nominal_resolution][].
"""

import pytest

from cmip_nominal_resolution.nominal_resolution import (
    calculate_nominal_resolution,
    calculate_nominal_resolution_regular_lat_lon_grid,
)
from cmip_nominal_resolution.testing import generate_regular_lat_lon_grid_test_info


@pytest.mark.parametrize(
    "nlon, nlat, exp",
    (
        (3, 4, "10000 km"),
        (4, 3, "10000 km"),
        (10, 9, "5000 km"),
        (18, 90, "2500 km"),
        (180, 10, "2500 km"),
        (24, 30, "1000 km"),
        (80, 40, "500 km"),
        (180, 90, "250 km"),
        (360, 180, "100 km"),
        (int(180 / 0.25), int(360 / 0.5), "50 km"),
        (int(180 / 0.125), int(360 / 0.25), "25 km"),
        # Higher resolution than this is very slow to calculate
        # and requires lots of memory just to hold the cell vertices.
    ),
)
def test_regular_lat_lon_grids(nlon, nlat, exp):
    info = generate_regular_lat_lon_grid_test_info(nlon=nlon, nlat=nlat)

    res = calculate_nominal_resolution(info.cell_vertices, info.cell_areas)

    assert res == exp


@pytest.mark.parametrize(
    "lat_delta, lon_delta, exp",
    (
        (60.0, 90.0, "10000 km"),
        (45.0, 120.0, "10000 km"),
        (18.0, 40.0, "5000 km"),
        (20.0, 8.0, "2500 km"),
        (1.0, 36.0, "2500 km"),
        (7.5, 12.0, "1000 km"),
        (2.0, 5.0, "500 km"),
        (1.0, 2.0, "250 km"),
        (0.5, 1.0, "100 km"),
        (0.25, 0.5, "50 km"),
        (0.25, 0.25, "25 km"),
        (0.05, 0.1, "10 km"),
        (0.025, 0.05, "5 km"),
        (0.01, 0.025, "2.5 km"),
        (0.01, 0.01, "1 km"),
        (0.005, 0.005, "0.5 km"),
    ),
)
def test_regular_lat_lon_grids_analytical(lat_delta, lon_delta, exp):
    res = calculate_nominal_resolution_regular_lat_lon_grid(
        lat_spacing=lat_delta, lon_spacing=lon_delta
    )

    assert res == exp

"""
Test determination of nominal resolution

These are high level integration tests.
"""

import pytest

from cmip_nominal_resolution.nominal_resolution import (
    calculate_nominal_resolution,
    calculate_nominal_resolution_regular_lat_lon_grid,
)
from cmip_nominal_resolution.testing import generate_regular_lat_lon_grid_test_info


@pytest.mark.parametrize(
    "nlon, nlat",
    (
        (3, 4),
        (4, 3),
        (10, 9),
        (18, 90),
        (180, 10),
        (24, 30),
        (80, 40),
        (180, 90),
        (360, 180),
    ),
)
def test_regular_lat_lon_grids_compared_to_analytical(nlon, nlat):
    info = generate_regular_lat_lon_grid_test_info(nlon=nlon, nlat=nlat)

    res_numerical = calculate_nominal_resolution(info.cell_vertices, info.cell_areas)
    res_analytical = calculate_nominal_resolution_regular_lat_lon_grid(
        lat_spacing=info.lat_delta, lon_spacing=info.lon_delta
    )

    assert res_numerical == res_analytical

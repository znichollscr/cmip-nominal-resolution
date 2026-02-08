"""
Test calculation of mean resolution

These are high level integration tests.
"""

import numpy as np
import pytest

from cmip_nominal_resolution.mean_resolution import (
    calculate_mean_resolution,
    calculate_mean_resolution_regular_lat_lon_grid,
)
from cmip_nominal_resolution.testing import generate_regular_lat_lon_grid_test_info


@pytest.mark.parametrize(
    "nlon, nlat",
    (
        (3, 4),
        (18, 90),
        (180, 10),
        (360, 180),
    ),
)
def test_regular_lat_lon_grids_compared_to_analytical_mean_resolution(nlon, nlat):
    info = generate_regular_lat_lon_grid_test_info(nlon=nlon, nlat=nlat)

    res_numerical = calculate_mean_resolution(info.cell_vertices, info.cell_areas)
    res_analytical = calculate_mean_resolution_regular_lat_lon_grid(
        lat_spacing=info.lat_delta, lon_spacing=info.lon_delta
    )

    np.testing.assert_allclose(
        res_numerical, res_analytical, rtol=0.02 * res_numerical / 10_000
    )

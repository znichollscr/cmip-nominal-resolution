"""
Test derivation of nominal resolution

These are high level integration tests.
"""

from typing import Any

import numpy as np
import pytest
from attrs import define

from cmip_nominal_resolution import calculate_nominal_resolution

# - regular lat lon grid
#    - test against analytic solution
# - irregular grid (e.g. triangles, hexagons)
# - some funky arrangement of cells
# - entry points: pint, numpy, xarray and warning handling


@define
class RegularLatLonGridTestInfo:
    cell_vertices: np.typing.NDArray[np.number[Any]]
    # Location of cell vertices in deg north and east
    # Shape: [ncells, nvertices, 2 (lat, lon)]
    cell_areas: np.typing.NDArray[np.number[Any]]
    # Shape: [ncells]
    lat_delta: float
    # Degrees
    lon_delta: float
    # Degrees


def generate_regular_lat_lon_grid_test_info(
    nlat: int,
    nlon: int,
    earth_radius: float = 6371.0,  # km
) -> RegularLatLonGridTestInfo:
    lat_delta = 180.0 / nlat
    lon_delta = 360.0 / nlon

    ncells = nlat * nlon

    breakpoint()

    res = RegularLatLonGridTestInfo(
        lat_delta=lat_delta,
        lon_delta=lon_delta,
    )

    return res


@pytest.mark.parametrize("nlon, nlat, exp", ((3, 4, "10000 km"),))
def test_regular_lat_lon_grids(nlon, nlat, exp):
    info = generate_regular_lat_lon_grid_test_info(nlon, nlat)

    res = calculate_nominal_resolution(info.cell_vertices, info.cell_areas)

    assert res == exp

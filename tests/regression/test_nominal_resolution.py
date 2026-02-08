"""
Regression tests of [cmip_nominal_resolution.nominal_resolution][].
"""

import numpy as np
import pytest
from attrs import define

from cmip_nominal_resolution.nominal_resolution import calculate_nominal_resolution


@define
class RegularLatLonGridTestInfo:
    cell_vertices: np.typing.NDArray[np.number[float]]
    # Location of cell vertices in deg north and east
    # Shape: [ncells, nvertices, 2 (lat, lon)]
    cell_areas: np.typing.NDArray[np.number[float]]
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

    lat_edges = np.linspace(-90.0, 90.0, nlat + 1)
    lat_upper = lat_edges[1:]
    lat_lower = lat_edges[:-1]

    lon_edges = np.linspace(0.0, 360.0, nlon + 1)
    lon_upper = lon_edges[1:]
    lon_lower = lon_edges[:-1]

    def get_cell_locs(
        lons: np.typing.NDArray[np.number[float]],
        lats: np.typing.NDArray[np.number[float]],
    ) -> np.typing.NDArray[np.number[float]]:
        return np.flip(
            np.reshape(np.dstack(np.meshgrid(lons, lats)), (ncells, 2)), axis=1
        )

    lower_lefts = get_cell_locs(lons=lon_lower, lats=lat_lower)
    lower_rights = get_cell_locs(lons=lon_upper, lats=lat_lower)
    upper_lefts = get_cell_locs(lons=lon_lower, lats=lat_upper)
    upper_rights = get_cell_locs(lons=lon_upper, lats=lat_upper)

    cell_vertices = np.stack(
        [lower_lefts, lower_rights, upper_rights, upper_lefts], axis=1
    )

    cell_max_lon = lower_rights[:, 1]
    cell_min_lon = lower_lefts[:, 1]
    cell_max_lat = upper_lefts[:, 0]
    cell_min_lat = lower_lefts[:, 0]

    cell_areas = (
        earth_radius**2
        * ((cell_max_lon - cell_min_lon) * np.pi / 180.0)
        * (np.sin(np.pi * cell_max_lat / 180.0) - np.sin(np.pi * cell_min_lat / 180.0))
    )

    res = RegularLatLonGridTestInfo(
        cell_vertices=cell_vertices,
        cell_areas=cell_areas,
        lat_delta=lat_delta,
        lon_delta=lon_delta,
    )

    return res


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

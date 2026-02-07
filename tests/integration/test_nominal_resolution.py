"""
Test derivation of nominal resolution

These are high level integration tests.
"""

import numpy as np
import pytest
import xarray as xr
from attrs import define

from cmip_nominal_resolution.nominal_resolution import (
    calculate_mean_resolution_regular_lat_lon_grid_fast_unitless,
    calculate_mean_resolution_unitless,
    calculate_nominal_resolution_regular_lat_lon_grid_fast_unitless,
    calculate_nominal_resolution_unitless,
)


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


# TODO: move to regression test
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

    res = calculate_nominal_resolution_unitless(info.cell_vertices, info.cell_areas)

    assert res == exp


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

    res_numerical = calculate_nominal_resolution_unitless(
        info.cell_vertices, info.cell_areas
    )
    res_analytical = calculate_nominal_resolution_regular_lat_lon_grid_fast_unitless(
        lat_spacing=info.lat_delta, lon_spacing=info.lon_delta
    )

    assert res_numerical == res_analytical


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
    res = calculate_nominal_resolution_regular_lat_lon_grid_fast_unitless(
        lat_spacing=lat_delta, lon_spacing=lon_delta
    )

    assert res == exp


# TODO: move this to a different test file
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

    res_numerical = calculate_mean_resolution_unitless(
        info.cell_vertices, info.cell_areas
    )
    res_analytical = calculate_mean_resolution_regular_lat_lon_grid_fast_unitless(
        lat_spacing=info.lat_delta, lon_spacing=info.lon_delta
    )

    np.testing.assert_allclose(
        res_numerical, res_analytical, rtol=0.02 * res_numerical / 10_000
    )


# TODO: change all to tests of mean resolution
@pytest.mark.parametrize(
    "side_length, exp",
    (
        (30.0, "2500 km"),
        (3.0, "250 km"),
        (1.0, "100 km"),
        (0.25, "25 km"),
        (0.005, "0.5 km"),
    ),
)
def test_triangular_grid(side_length, exp):
    dlat = side_length * np.sin(60.0 * np.pi / 180)
    origin_cell = np.array(
        [
            [0.0, 0.0],
            [0.0, side_length],
            [dlat, side_length / 2.0],
        ]
    )

    lon_shift_cell_a = np.copy(origin_cell)
    lon_shift_cell_a[:, 1] += side_length

    lon_shift_cell_b = np.copy(origin_cell)
    lon_shift_cell_b[:, 1] += 2 * side_length

    inverted_cell = np.array(
        [
            [0.0, side_length],
            [dlat, 3 * side_length / 2.0],
            [dlat, side_length / 2.0],
        ]
    )

    inverted_lon_shift_cell_a = np.copy(inverted_cell)
    inverted_lon_shift_cell_a[:, 1] += side_length

    inverted_lon_shift_cell_b = np.copy(inverted_cell)
    inverted_lon_shift_cell_b[:, 1] += 2 * side_length

    cell_vertices = np.stack(
        [
            origin_cell,
            lon_shift_cell_a,
            lon_shift_cell_b,
            inverted_cell,
            inverted_lon_shift_cell_b,
            inverted_lon_shift_cell_b,
        ]
    )
    cell_areas = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    res = calculate_nominal_resolution_unitless(cell_vertices, cell_areas)

    assert res == exp


# TODO: change all to tests of mean resolution
@pytest.mark.parametrize(
    "side_length, exp",
    (
        (20.0, "5000 km"),
        (2.0, "500 km"),
        (0.2, "50 km"),
    ),
)
def test_hexagonal_grid(side_length, exp):
    dlat = side_length * np.cos(30.0 * np.pi / 180)
    dlon = side_length * np.sin(30.0 * np.pi / 180)

    origin_cell = np.array(
        [
            [0.0, 0.0],
            [0.0, side_length],
            [dlat, side_length + dlon],
            [2 * dlat, side_length],
            [2 * dlat, 0.0],
            [dlat, -dlon],
        ]
    )

    middle_cell = np.copy(origin_cell)
    middle_cell[:, 1] += side_length + dlon
    middle_cell[:, 0] += dlat

    lon_shift_cell = np.copy(origin_cell)
    lon_shift_cell[:, 1] += 2 * side_length + 2 * dlon

    cell_vertices = np.stack([origin_cell, middle_cell, lon_shift_cell])
    cell_areas = np.array([1.0, 1.0, 1.0])

    res = calculate_nominal_resolution_unitless(cell_vertices, cell_areas)

    assert res == exp


# TODO: change all to tests of mean resolution
@pytest.mark.parametrize(
    "side_length, exp",
    (
        (20.0, "5000 km"),
        (2.0, "500 km"),
        (0.2, "50 km"),
    ),
)
def test_triangle_hexagonal_grid(side_length, exp):
    dlat = side_length * np.cos(30.0 * np.pi / 180)
    dlon = side_length * np.sin(30.0 * np.pi / 180)

    origin_cell_hexagon = np.array(
        [
            [0.0, 0.0],
            [0.0, side_length],
            [dlat, side_length + dlon],
            [2 * dlat, side_length],
            [2 * dlat, 0.0],
            [dlat, -dlon],
        ]
    )

    lon_shift_cell_hexagon = np.copy(origin_cell_hexagon)
    lon_shift_cell_hexagon[:, 1] += side_length + 2 * dlon

    upright_triangle = np.ma.masked_invalid(
        [
            [0.0, side_length],
            [0.0, 2 * side_length],
            [dlat, 3 * side_length / 2.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    upside_down_triangle = np.ma.masked_invalid(
        [
            [2 * dlat, side_length],
            [2 * dlat, 2 * side_length],
            [dlat, 3 * side_length / 2.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    cell_vertices = np.stack(
        [
            origin_cell_hexagon,
            lon_shift_cell_hexagon,
            upright_triangle,
            upside_down_triangle,
        ]
    )
    cell_areas = np.array([6.0, 6.0, 1.0, 1.0])

    res = calculate_nominal_resolution_unitless(cell_vertices, cell_areas)

    assert res == exp


def test_irregular_grid(test_data_path):
    exp = "50 km"

    fp = test_data_path / "areacello_Ofx_AWI-CM-1-1-MR_ssp370_r4i1p1f1_gn.nc"

    ds = xr.open_dataset(fp)

    cell_vertices = np.dstack([ds["lat_bnds"].values, ds["lon_bnds"].values])

    # I don't think this is meant to be needed according to CF-conventions,
    # but there is no checking of the data so it's not surprising there are issues.
    for i in range(cell_vertices.shape[1] - 1)[::-1]:
        cell_vertices[
            np.where(np.equal(cell_vertices[:, i + 1, :], cell_vertices[:, i, :]))
        ] = np.nan

    cell_areas = ds["areacello"].values

    res = calculate_nominal_resolution_unitless(cell_vertices, cell_areas)

    assert res == exp


# - entry points: pint, numpy, xarray and warning handling

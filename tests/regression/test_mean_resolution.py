"""
Regression tests of [cmip_nominal_resolution.mean_resolution][].
"""

import numpy as np
import pytest
import xarray as xr

from cmip_nominal_resolution.mean_resolution import (
    calculate_mean_resolution,
)


@pytest.mark.parametrize(
    "side_length, exp",
    (
        (30.0, 3321.0),
        (3.0, 333.5),
        (1.0, 111.2),
        (0.25, 27.8),
        (0.005, 0.556),
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

    res = calculate_mean_resolution(cell_vertices, cell_areas)

    np.testing.assert_allclose(res, exp, rtol=1e-3)


@pytest.mark.parametrize(
    "side_length, exp",
    (
        (20.0, 4329.0),
        (2.0, 444.7),
        (0.2, 44.48),
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

    res = calculate_mean_resolution(cell_vertices, cell_areas)

    np.testing.assert_allclose(res, exp, rtol=1e-3)


@pytest.mark.parametrize(
    "side_length, exp",
    (
        (20.0, 4066.0),
        (2.0, 413.0),
        (0.2, 41.3),
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

    res = calculate_mean_resolution(cell_vertices, cell_areas)

    np.testing.assert_allclose(res, exp, rtol=1e-3)


def test_irregular_grid(test_data_path):
    exp = 42.44

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

    res = calculate_mean_resolution(cell_vertices, cell_areas)

    np.testing.assert_allclose(res, exp, rtol=1e-3)

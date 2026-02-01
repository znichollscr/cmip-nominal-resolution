"""
Determination of nominal resolution
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt

DEFAULT_NOMINAL_RESOLUTION_THRESHOLDS = {
    0.72: "0.5 km",
    1.6: "1 km",
    3.6: "2.5 km",
    7.2: "5 km",
    16.0: "10 km",
    36.0: "25 km",
    72.0: "50 km",
    160.0: "100 km",
    360.0: "250 km",
    720.0: "500 km",
    1600.0: "1000 km",
    3600.0: "2500 km",
    7200.0: "5000 km",
    np.inf: "10000 km",
}


def calculate_central_angle_unitless(
    lat_lon_a: npt.NDArray[np.number[float]], lat_lon_b: npt.NDArray[np.number[float]]
) -> float:
    lat_a_r = lat_lon_a[0] * np.pi / 180
    lat_b_r = lat_lon_b[0] * np.pi / 180
    delta_lon_r = (lat_lon_a[1] - lat_lon_b[1]) * np.pi / 180

    central_angle = np.arccos(
        np.sin(lat_a_r) * np.sin(lat_b_r)
        + np.cos(lat_a_r) * np.cos(lat_b_r) * np.cos(delta_lon_r)
    )

    return central_angle


def calculate_central_angles_for_cell_unitless(
    single_cell_vertices: npt.NDArray[np.number[float]],
) -> npt.NDArray[np.number[float]]:
    ncells = single_cell_vertices.shape[0]
    central_angles = np.zeros(int(ncells * (ncells - 1) / 2))
    for i, (lat_lon_a, lat_lon_b) in enumerate(
        itertools.combinations(single_cell_vertices, 2)
    ):
        central_angles[i] = calculate_central_angle_unitless(lat_lon_a, lat_lon_b)

    return central_angles


def calculate_central_angles_unitless(
    cell_vertices: npt.NDArray[np.number[float]],
) -> npt.NDArray[np.number[float]]:
    central_angles = np.zeros(
        (
            cell_vertices.shape[0],
            int(cell_vertices.shape[1] * (cell_vertices.shape[1] - 1) / 2),
        )
    )
    # TODO: vectorise or speed up, I think this is the slow bit
    for i, single_cell_vertices in enumerate(cell_vertices):
        central_angles[i] = calculate_central_angles_for_cell_unitless(
            single_cell_vertices=single_cell_vertices
        )

    return central_angles


def calculate_mean_resolution_unitless(
    max_distances: npt.NDArray[np.number[float]],
    cell_areas: npt.NDArray[np.number[float]],
) -> float:
    mean_resolution = np.sum(max_distances * cell_areas) / np.sum(cell_areas)

    return mean_resolution


def determine_nominal_resolution(
    mean_resolution: float,
    nominal_resolution_thresholds: dict[float, str],
) -> str:
    for threshold in sorted(nominal_resolution_thresholds.keys()):
        if mean_resolution < threshold:
            return nominal_resolution_thresholds[threshold]

    msg = (
        f"{mean_resolution=} is greater than all of the thresholds "
        f"in {nominal_resolution_thresholds=}"
    )
    raise ValueError(msg)


def calculate_nominal_resolution_unitless(
    cell_vertices: npt.NDArray[np.number[float]],
    cell_areas: npt.NDArray[np.number[float]],
    nominal_resolution_thresholds: dict[float, str] | None = None,
    earth_radius: float = 6371.0,  # km
) -> str:
    # Location of cell vertices in deg north and east
    # Shape: [ncells, nvertices, 2 (lat, lon)]
    if nominal_resolution_thresholds is None:
        nominal_resolution_thresholds = DEFAULT_NOMINAL_RESOLUTION_THRESHOLDS

    central_angles = calculate_central_angles_unitless(cell_vertices=cell_vertices)
    max_distances = earth_radius * np.max(central_angles, axis=1)
    mean_resolution = calculate_mean_resolution_unitless(max_distances, cell_areas)
    nominal_resolution = determine_nominal_resolution(
        mean_resolution, nominal_resolution_thresholds
    )

    return nominal_resolution


def calculate_nominal_resolution_regular_lat_lon_grid_fast_unitless(
    lat_spacing: float,
    lon_spacing: float,
    nominal_resolution_thresholds: dict[float, str] | None = None,
    earth_radius: float = 6371.0,  # km
) -> str:
    if nominal_resolution_thresholds is None:
        nominal_resolution_thresholds = DEFAULT_NOMINAL_RESOLUTION_THRESHOLDS

    lat_spacing_r = lat_spacing * np.pi / 180
    lon_spacing_r = lon_spacing * np.pi / 180

    mean_resolution = (
        earth_radius
        * lat_spacing_r
        / 2
        * (
            1
            + (lat_spacing_r**2 + lon_spacing_r**2)
            / (lat_spacing_r * lon_spacing_r)
            * np.arctan(lon_spacing_r / lat_spacing_r)
        )
    )

    nominal_resolution = determine_nominal_resolution(
        mean_resolution, nominal_resolution_thresholds
    )

    return nominal_resolution

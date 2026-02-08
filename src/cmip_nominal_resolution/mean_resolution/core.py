"""
Core algorithm

If the link is still live, the steps are the first two steps described here
https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6_MrDz1q8n5XUk/edit?tab=t.0#bookmark=id.ibeh7ad2gpdi
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt


def calculate_central_angle(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float
) -> float:
    """
    Calculate central angle between two points

    Implemented following wikipedia:
    https://en.wikipedia.org/wiki/Great-circle_distance

    Parameters
    ----------
    lat_a
        Latitude of first point in degrees north

    lon_a
        Longitude of first point in degrees east

    lat_b
        Latitude of second point in degrees north

    lon_b
        Longitude of second point in degrees east

    Returns
    -------
    :
        Central angle between the points
    """
    lat_a_r = lat_a * np.pi / 180
    lat_b_r = lat_b * np.pi / 180
    delta_lon_r = (lon_a - lon_b) * np.pi / 180

    central_angle = np.arccos(
        np.sin(lat_a_r) * np.sin(lat_b_r)
        + np.cos(lat_a_r) * np.cos(lat_b_r) * np.cos(delta_lon_r)
    )

    return central_angle


def calculate_central_angles(
    cell_vertices: npt.NDArray[np.number[float]],
) -> npt.NDArray[np.number[float]]:
    """
    Calculate central angles between all pairs of vertices in the given cells

    Parameters
    ----------
    cell_vertices
        Cell vertices

        This should be an array of shape (ncells, nvertices, 2)
        i.e. the number of cells, the (maximum) number of vertices in each cell
        then the latitude and longitude of each vertex
        in degrees north and degrees east respectively.

    Returns
    -------
    :
        Central angles between all pairs of vertices in each cell

        This is an array of shape (ncells, npairs)
        i.e. the number of cells and the number of pairs of vertices in each cell.
    """
    central_angles = np.zeros(
        (
            cell_vertices.shape[0],
            int(cell_vertices.shape[1] * (cell_vertices.shape[1] - 1) / 2),
        )
    )
    for i, (cell_index_a, cell_index_b) in enumerate(
        itertools.combinations(np.arange(cell_vertices.shape[1]), 2)
    ):
        cell_vertices_a = cell_vertices[:, cell_index_a, :]
        cell_vertices_b = cell_vertices[:, cell_index_b, :]

        central_angles[:, i] = calculate_central_angle(
            lat_a=cell_vertices_a[:, 0],
            lon_a=cell_vertices_a[:, 1],
            lat_b=cell_vertices_b[:, 0],
            lon_b=cell_vertices_b[:, 1],
        )

    return central_angles


def calculate_mean_resolution(
    cell_vertices: npt.NDArray[np.number[float]],
    cell_areas: npt.NDArray[np.number[float]],
    earth_radius: float = 6371.0,
) -> float:
    """
    Calculate mean resolution for a given set of cells

    Parameters
    ----------
    cell_vertices
        Cell vertices

        This should be an array of shape (ncells, nvertices, 2)
        i.e. the number of cells, the (maximum) number of vertices in each cell
        then the latitude and longitude of each vertex
        in degrees north and degrees east respectively.

    cell_areas
        Area of each cell

        This should be an array of shape (ncells,) i.e. the number of cells.

    earth_radius
        Radius of the earth to use in calculations

        This should be in km.

    Returns
    -------
    :
        Mean resolution for the given set of cells
    """
    central_angles = calculate_central_angles(cell_vertices=cell_vertices)
    max_distances = earth_radius * np.nanmax(central_angles, axis=1)
    mean_resolution = np.sum(max_distances * cell_areas) / np.sum(cell_areas)

    return mean_resolution

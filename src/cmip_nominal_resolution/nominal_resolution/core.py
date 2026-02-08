"""
Core algorithm
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import numpy.typing as npt
from attrs import field, frozen

from cmip_nominal_resolution.mean_resolution import calculate_mean_resolution

ASSUMED_DISTANCE_UNIT: str = "km"
"""
Unit assumed for all distances in this module
"""


@frozen
class NominalResolutionThresholdDefinition:
    """
    Definition of thresholds for determining the nominal resolution
    """

    thresholds: dict[float, str] = field()
    """
    Thresholds for assigning nominal resolution

    The keys (thresholds) are checked from smallest to largest.
    If the resolution is less than the threshold,
    its value (label) is assigned as the nominal resolution.
    If the resolution is bigger than all thresholds,
    then `fallback` is used.
    """

    fallback: str = field()
    """
    Nominal resolution to assign to data that is not below any of the thresholds
    """

    def __attrs_post_init__(self):
        """
        Sort `self.thresholds` to allow for faster usage
        """
        thresholds_sorted = OrderedDict(
            {k: self.thresholds[k] for k in sorted(self.thresholds.keys())}
        )
        object.__setattr__(self, "thresholds", thresholds_sorted)

    def to_pretty(self) -> str:
        """
        To pretty representation

        Returns
        -------
        :
            Pretty representation that shows the thresholds and corresponding labels
            in an arguably more human-readable way
        """
        keys = list(self.thresholds.keys())
        unit = ASSUMED_DISTANCE_UNIT
        mk = keys[0]
        pl = [
            f"r <= {mk} {unit}: {self.thresholds[mk]!r}",
            *(
                f"{kp} {unit} < r <= {k} {unit}: {self.thresholds[k]!r}"
                for kp, k in zip(keys[:-1], keys[1:])
            ),
            f"Otherwise: {self.fallback!r}",
        ]

        return "\n".join(pl)

    def translate(self, mean_resolution: float) -> str:
        """
        Translate mean resolution to nominal resolution

        Parameters
        ----------
        mean_resolution
            Mean resolution (in km)

        Returns
        -------
        :
            Nominal resolution
        """
        for threshold, nominal_resolution in self.thresholds.items():
            if mean_resolution <= threshold:
                return nominal_resolution

        return self.fallback


DEFAULT_NOMINAL_RESOLUTION_THRESHOLD_DEFINITION = NominalResolutionThresholdDefinition(
    thresholds={
        0.72: f"0.5 {ASSUMED_DISTANCE_UNIT}",
        1.6: f"1 {ASSUMED_DISTANCE_UNIT}",
        3.6: f"2.5 {ASSUMED_DISTANCE_UNIT}",
        7.2: f"5 {ASSUMED_DISTANCE_UNIT}",
        16.0: f"10 {ASSUMED_DISTANCE_UNIT}",
        36.0: f"25 {ASSUMED_DISTANCE_UNIT}",
        72.0: f"50 {ASSUMED_DISTANCE_UNIT}",
        160.0: f"100 {ASSUMED_DISTANCE_UNIT}",
        360.0: f"250 {ASSUMED_DISTANCE_UNIT}",
        720.0: f"500 {ASSUMED_DISTANCE_UNIT}",
        1600.0: f"1000 {ASSUMED_DISTANCE_UNIT}",
        3600.0: f"2500 {ASSUMED_DISTANCE_UNIT}",
        7200.0: f"5000 {ASSUMED_DISTANCE_UNIT}",
    },
    fallback=f"10000 {ASSUMED_DISTANCE_UNIT}",
)


def calculate_nominal_resolution(
    cell_vertices: npt.NDArray[np.number[float]],
    cell_areas: npt.NDArray[np.number[float]],
    earth_radius: float = 6371.0,  # km
    nominal_resolution_thresholds: NominalResolutionThresholdDefinition = DEFAULT_NOMINAL_RESOLUTION_THRESHOLD_DEFINITION,
) -> str:
    """
    Calculate nominal resolution for a given set of cells

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

    nominal_resolution_thresholds
        Thresholds to use when converting mean resolution to nominal resolution

    Returns
    -------
    :
        Nominal resolution for the given set of cells
    """
    mean_resolution = calculate_mean_resolution(
        cell_vertices=cell_vertices,
        cell_areas=cell_areas,
        earth_radius=earth_radius,
    )

    nominal_resolution = nominal_resolution_thresholds.translate(mean_resolution)

    return nominal_resolution

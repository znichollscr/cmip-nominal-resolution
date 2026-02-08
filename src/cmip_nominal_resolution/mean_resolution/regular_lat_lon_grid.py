"""
Calculation of mean resolution for a regular lat-lon grid

This is much faster than the full calculations done in
[cmip_nominal_resolution.mean_resolution.core][].

If the link is still live, the formula is given in step 3 of
https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6_MrDz1q8n5XUk/edit?tab=t.0#bookmark=id.ibeh7ad2gpdi
"""

from __future__ import annotations

import numpy as np


def calculate_mean_resolution_regular_lat_lon_grid(
    lat_spacing: float,
    lon_spacing: float,
    earth_radius: float = 6371.0,
) -> float:
    """
    Calculate mean resolution for a regular latitude longitude grid

    Parameters
    ----------
    lat_spacing
        Latitudinal spacing (in degrees)

    lon_spacing
        Lonitudinal spacing (in degrees)

    earth_radius
        Radius of the earth to use in calculations

        This should be in km.

    Returns
    -------
    :
        Mean resolution for the given grid information
    """
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

    return mean_resolution

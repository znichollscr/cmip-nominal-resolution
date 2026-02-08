"""
Calculation of mean resolution
"""

from cmip_nominal_resolution.mean_resolution.core import calculate_mean_resolution
from cmip_nominal_resolution.mean_resolution.regular_lat_lon_grid import (
    calculate_mean_resolution_regular_lat_lon_grid,
)

__all__ = [
    "calculate_mean_resolution",
    "calculate_mean_resolution_regular_lat_lon_grid",
]

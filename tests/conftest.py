"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from pathlib import Path

import pytest

HERE = Path(__file__).parents[0]


@pytest.fixture(scope="session")
def test_data_path():
    return HERE / "test-data"

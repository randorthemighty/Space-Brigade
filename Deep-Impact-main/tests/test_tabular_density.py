import pytest
import numpy as np
from deepimpact.solver import Planet  # Replace with your actual module name


@pytest.fixture
def earth():
    # Create an instance of the Planet class with tabular atmospheric function
    return Planet(atmos_func="tabular")


def test_known_data_points(earth):
    # Test at known data points from the CSV
    assert np.isclose(earth.rhoa(0), 1.225, atol=1e-3)
    assert np.isclose(earth.rhoa(2000), 1.00649, atol=1e-3)
    assert np.isclose(earth.rhoa(10000), 0.4127, atol=1e-3)
    # Test higher altitude data point from the CSV
    assert np.isclose(earth.rhoa(86000), 5.6411400000000003e-06, atol=1e-5)


def test_interpolation(earth):
    # Test the interpolation at points not directly in your dataset.
    # Testing against standard atmosphere values
    expected_interpolated_density_3000 = 0.9091
    assert np.isclose(earth.rhoa(3000), expected_interpolated_density_3000, atol=1e-3)
    expected_interpolated_density_25000 = 0.0395
    assert np.isclose(earth.rhoa(25000), expected_interpolated_density_25000, atol=1e-3)


def test_extrapolation(earth):
    expected_extrapolated_density = 0.000003
    assert np.isclose(earth.rhoa(89700), expected_extrapolated_density, atol=1e-3)
    assert np.isclose(earth.rhoa(110000), 0.0000001, atol=1e-3)
    # Test negative altitude.
    assert np.isclose(earth.rhoa(-1000), 1.347, atol=1e-3)

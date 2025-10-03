import numpy as np
from deepimpact import great_circle_distance, GeospatialLocator
import pytest


def test_great_circle_distance_base():
    # Test with two different points
    distance = great_circle_distance([[54.0, 0.0]], [[55, 1.0]])
    assert distance.shape == (1, 1)
    assert distance[0, 0] > 0

    # Test with identical points
    distance = great_circle_distance([[54.0, 0.0]], [[54.0, 0.0]])
    assert distance[0, 0] == 0


def test_great_circle_distance():
    # check for test case in base file
    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])

    data = np.array(
        [
            [1.28580537e05, 2.59579735e05, 2.25409117e02],
            [0.00000000e00, 2.24656571e05, 1.28581437e05],
            [2.72529953e05, 2.08175028e05, 1.96640630e05],
        ]
    )

    dist = great_circle_distance(pnts1, pnts2)

    assert np.allclose(data, dist, rtol=1.0e-4)


def test_longitude_edge():
    # checking for edge case for lognitude
    point1 = [0, -179]
    point2 = [0, 179]
    distance = great_circle_distance(point1, point2)
    assert np.isclose(distance, great_circle_distance([0, -1], [0, 1]))


def test_invalid_latitude():
    # Test with latitude out of range
    latlon1 = [[-91, 0]]
    latlon2 = [[10, 10]]
    with pytest.raises(ValueError):
        great_circle_distance(latlon1, latlon2)

    latlon1 = [[91, 0]]
    with pytest.raises(ValueError):
        great_circle_distance(latlon1, latlon2)


def test_invalid_longitude():
    # Test with longitude out of range
    latlon1 = [[0, -181]]
    latlon2 = [[0, 0]]
    with pytest.raises(ValueError):
        great_circle_distance(latlon1, latlon2)

    latlon1 = [[0, 181]]
    with pytest.raises(ValueError):
        great_circle_distance(latlon1, latlon2)


def test_invalid_lat_and_lon():
    # Test with both latitude and longitude out of range
    latlon1 = [[-91, -181]]
    latlon2 = [[91, 181]]
    with pytest.raises(ValueError):
        great_circle_distance(latlon1, latlon2)


def test_get_postcodes_by_radius():
    # check for postcodes for a radius
    locator = GeospatialLocator()
    result = locator.get_postcodes_by_radius((51.4981, -0.1773), [1500])
    assert isinstance(result, list)
    assert all(isinstance(sublist, list) for sublist in result)


def test_find_nearest_coordinates():
    # check for the outputs for helper function
    locator = GeospatialLocator()
    coords, distances = locator.find_nearest_coordinates((51.4981, -0.1773), 4)
    assert len(coords) == 4
    assert len(distances) == 4
    assert all(d >= 0 for d in distances)


def test_get_population_by_radius():
    # test for the edge cases for population function
    locator = GeospatialLocator()
    populations = locator.get_population_by_radius((51.4981, -0.1773), [10, 500, 1000])
    assert len(populations) == 3
    assert all(isinstance(p, int) for p in populations)
    assert all(p >= 0 for p in populations)

    # Test with negative radius
    populations = locator.get_population_by_radius((51.4981, -0.1773), [-100])
    assert populations == [0]

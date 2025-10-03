import pandas as pd
import numpy as np
import os

from pytest import fixture


# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly


@fixture(scope="module")
def deepimpact():
    import deepimpact

    return deepimpact


@fixture(scope="module")
def planet(deepimpact):
    return deepimpact.Planet()


@fixture(scope="module")
def loc(deepimpact):
    return deepimpact.GeospatialLocator()


def test_damage_zones(deepimpact):
    outcome = {
        "burst_peak_dedz": 1000.0,
        "burst_altitude": 9000.0,
        "burst_distance": 90000.0,
        "burst_energy": 6000.0,
        "outcome": "Airburst",
    }

    blat, blon, damrad = deepimpact.damage_zones(
        outcome, 55.0, 0.0, 135.0, [27e3, 43e3]
    )

    assert type(blat) is float
    assert type(blon) is float
    assert type(damrad) is list
    assert len(damrad) == 2


def test_longlist_pressure(deepimpact):
    outcome = {
        "burst_peak_dedz": 1000.0,
        "burst_altitude": 9000.0,
        "burst_distance": 90000.0,
        "burst_energy": 6000.0,
        "outcome": "Airburst",
    }

    blat, blon, damrad = deepimpact.damage_zones(
        outcome, 55.0, 0.0, 135.0, [1, 10, 50, 100, 500, 1e3, 1e4]
    )

    assert type(blat) is float
    assert type(blon) is float
    assert type(damrad) is list
    assert len(damrad) == 7


def test_empty_list(deepimpact):
    outcome = {
        "burst_peak_dedz": 1000.0,
        "burst_altitude": 9000.0,
        "burst_distance": 90000.0,
        "burst_energy": 6000.0,
        "outcome": "Airburst",
    }

    blat, blon, damrad = deepimpact.damage_zones(outcome, 55.0, 0.0, 135.0, [])

    assert type(blat) is float
    assert type(blon) is float
    assert type(damrad) is list
    assert len(damrad) == 0


# Test is the pressure is extremely large
def test_extreme_pressure(deepimpact):
    outcome = {
        "burst_peak_dedz": 1000.0,
        "burst_altitude": 9000.0,
        "burst_distance": 90000.0,
        "burst_energy": 6000.0,
        "outcome": "Airburst",
    }

    blat, blon, damrad = deepimpact.damage_zones(outcome, 55.0, 0.0, 135.0, [80e4])

    assert type(blat) is float
    assert type(blon) is float
    assert type(damrad) is list
    assert len(damrad) == 0


def test_great_circle_distance(deepimpact):
    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])

    data = np.array(
        [
            [1.28580537e05, 2.59579735e05, 2.25409117e02],
            [0.00000000e00, 2.24656571e05, 1.28581437e05],
            [2.72529953e05, 2.08175028e05, 1.96640630e05],
        ]
    )

    dist = deepimpact.great_circle_distance(pnts1, pnts2)

    assert np.allclose(data, dist, rtol=1.0e-4)


# Fixture for common outcome dictionary
@fixture
def example_outcome(scope="module"):
    return {
        "burst_altitude": 8e3,
        "burst_energy": 7e3,
        "burst_distance": 90e3,
        "burst_peak_dedz": 1e3,
        "outcome": "Airburst",
    }


# Zero Location test
# Test Normal Case
def test_normal_case(deepimpact, example_outcome):
    blat, blon, damrad = deepimpact.damage_zones(
        example_outcome, 52.79, -2.95, 135, pressures=[1e3, 3.5e3, 27e3, 43e3]
    )
    assert isinstance(blat, float) and isinstance(blon, float)
    assert all(isinstance(r, float) for r in damrad)


# Test Zero Burst Distance
def test_zero_burst_distance(deepimpact, example_outcome):
    outcome = {
        "burst_peak_dedz": 1000.0,
        "burst_altitude": 9000.0,
        "burst_distance": 0.0,
        "burst_energy": 6000.0,
        "outcome": "Airburst",
    }
    blat, blon, _ = deepimpact.damage_zones(outcome, 52.79, -2.95, 135, pressures=[1e3])
    assert blat == 52.79 and blon == -2.95

    example_outcome["burst_distance"] = 0
    blat, blon, _ = deepimpact.damage_zones(
        example_outcome, 52.79, -2.95, 135, pressures=[1e3]
    )
    assert blat == 52.79 and blon == -2.95


def test_extreme_lat_lon(deepimpact, example_outcome):
    # Maximum latitude and longitude
    blat, blon, _ = deepimpact.damage_zones(
        example_outcome, 90, 180, 135, pressures=[1e3]
    )
    assert -90 <= blat <= 90 and -180 <= blon <= 180

    # Minimum latitude and longitude
    blat, blon, _ = deepimpact.damage_zones(
        example_outcome, -90, -180, 135, pressures=[1e3]
    )
    assert -90 <= blat <= 90 and -180 <= blon <= 180


def test_bearing_edge_cases(deepimpact, example_outcome):
    # Bearing 0 degrees
    blat, blon, _ = deepimpact.damage_zones(
        example_outcome, 52.79, -2.95, 0, pressures=[1e3]
    )
    assert isinstance(blat, float) and isinstance(blon, float)

    # Bearing 360 degrees
    blat, blon, _ = deepimpact.damage_zones(
        example_outcome, 52.79, -2.95, 360, pressures=[1e3]
    )
    assert isinstance(blat, float) and isinstance(blon, float)


# !!! check this test case for extreme value
def test_extremely_large_inputs(deepimpact, example_outcome):
    example_outcome["burst_distance"] = 1e12  # Extremely large burst distance
    blat, blon, _ = deepimpact.damage_zones(
        example_outcome, 52.79, -2.95, 135, pressures=[1e12]
    )
    assert isinstance(blat, float) and isinstance(blon, float)


def test_impact_risk(deepimpact, planet):
    probability, population = deepimpact.impact_risk(planet)

    assert type(probability) is pd.DataFrame
    assert "probability" in probability.columns
    assert type(population) is dict
    assert "mean" in population.keys()
    assert "stdev" in population.keys()
    assert type(population["mean"]) is float
    assert type(population["stdev"]) is float

    # common test
    risk_file = os.sep.join(
        (os.path.dirname(__file__), "..", "resources", "impact_parameter_list.csv")
    )
    assert os.path.isfile(risk_file)

    # sepcial case: pressure
    probability, population = deepimpact.impact_risk(
        planet, pressure=[1e3, 4e3, 30e3, 50e3]
    )
    assert not probability

    # special case: nsampel = 1
    probability, population = deepimpact.impact_risk(planet, nsamples=1)
    assert all([element == 1 for element in probability["probability"]])

    # special case: nsampel = None
    probability, population = deepimpact.impact_risk(planet)
    assert all([element <= 1 for element in probability["probability"]]) and all(
        [element >= 0 for element in probability["probability"]]
    )

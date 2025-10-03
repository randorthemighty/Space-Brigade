from collections import OrderedDict
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


@fixture(scope="module")
def result(planet):
    input = {
        "radius": 1.0,
        "velocity": 2.0e4,
        "density": 3000.0,
        "strength": 1e5,
        "angle": 30.0,
        "init_altitude": 0.0,
    }

    result = planet.solve_atmospheric_entry(**input)

    return result


@fixture(scope="module")
def outcome(planet, result):
    outcome = planet.analyse_outcome(result=result)
    return outcome


def test_import(deepimpact):
    assert deepimpact


def test_planet_signature(deepimpact):
    inputs = OrderedDict(
        atmos_func="exponential",
        atmos_filename=None,
        Cd=1.0,
        Ch=0.1,
        Q=1e7,
        Cl=1e-3,
        alpha=0.3,
        Rp=6371e3,
        g=9.81,
        H=8000.0,
        rho0=1.2,
    )

    # call by keyword
    _ = deepimpact.Planet(**inputs)

    # call by position
    _ = deepimpact.Planet(*inputs.values())


def test_attributes(planet):
    for key in ("Cd", "Ch", "Q", "Cl", "alpha", "Rp", "g", "H", "rho0"):
        assert hasattr(planet, key)


def test_atmos_filename(planet):
    assert os.path.isfile(planet.atmos_filename)


def test_solve_atmospheric_entry(result):
    assert type(result) is pd.DataFrame

    for key in ("velocity", "mass", "angle", "altitude", "distance", "radius", "time"):
        assert key in result.columns


def test_calculate_energy(planet, result):
    energy = planet.calculate_energy(result=result)

    assert type(energy) is pd.DataFrame

    for key in (
        "velocity",
        "mass",
        "angle",
        "altitude",
        "distance",
        "radius",
        "time",
        "dedz",
    ):
        assert key in energy.columns


def test_analyse_outcome(outcome):
    assert type(outcome) is dict

    for key in (
        "outcome",
        "burst_peak_dedz",
        "burst_altitude",
        "burst_distance",
        "burst_energy",
    ):
        assert key in outcome.keys()


def test_scenario(planet):
    inputs = {
        "radius": 35.0,
        "angle": 45.0,
        "strength": 1e7,
        "density": 3000.0,
        "velocity": 19e3,
    }

    result = planet.solve_atmospheric_entry(**inputs)

    # For now, we just check the result is a DataFrame
    # and the columns are as expected.

    # You should add more tests here to check the values
    # are correct and match the expected output
    # given in the tests/scenario.npz file

    assert type(result) is pd.DataFrame

    for key in ("velocity", "mass", "angle", "altitude", "distance", "radius", "time"):
        assert key in result.columns


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


def test_locator_postcodes(loc):
    latlon = (52.2074, 0.1170)

    result = loc.get_postcodes_by_radius(latlon, [0.2e3, 0.1e3])

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is list


def test_population_by_radius(loc):
    latlon = (52.2074, 0.1170)

    result = loc.get_population_by_radius(latlon, [5e2, 1e3])

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is int


def test_impact_risk(deepimpact, planet):
    probability, population = deepimpact.impact_risk(planet)

    assert type(probability) is pd.DataFrame
    assert "probability" in probability.columns
    assert type(population) is dict
    assert "mean" in population.keys()
    assert "stdev" in population.keys()
    assert type(population["mean"]) is float
    assert type(population["stdev"]) is float

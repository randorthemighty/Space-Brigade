"""Module to calculate the damage and impact risk for given scenarios"""
from collections import Counter
from folium.plugins import HeatMap
import os
import math
import pandas as pd
import folium
import numpy as np
import deepimpact

__all__ = ["damage_zones", "impact_risk"]


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii
        for the input damage levels

    Examples
    --------

    >>> import deepimpact
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3, 'burst_distance': 90e3, 'burst_peak_dedz': 1e3, 'outcome': 'Airburst'}
    >>> deepimpact.damage_zones(outcome, 52.79, -2.95, 135, pressures=[1e3, 3.5e3, 27e3, 43e3])
    """

    # Replace this code with your own. For demonstration we just
    # return lat, lon and a radius of 5000 m for each pressure
    burst_altitude = outcome["burst_altitude"]
    burst_energy = outcome["burst_energy"]

    # default to 0 if not found
    burst_distance = outcome.get("burst_distance", 0)

    # Calculate the surface zero point
    blat, blon = find_destination(lat, lon, bearing, burst_distance)

    # Calculate damage radii for each pressure threshold
    damrad = calculate_damage_radius(pressures, burst_altitude, burst_energy)

    return blat, blon, damrad


def find_destination(lat, lon, bearing, distance):
    """
    Calculate the latitude and longitude of the surface zero location point,
    given an entry
    latitude and longitude, a bearing, and a burst distance.

    Parameters
    ----------

    lat : float
        The latitude of the entry point in degrees.
    lon : float
        The longitude of the entry point in degrees.
    bearing : float
        The bearing from the entry point in degrees. This is the angle measured
        in degrees clockwise from the north.
    distance : float
        The distance to the destination from the entry point in meters.

    Returns
    -------

    zero_lat : float
        The latitude of the surface zero location point in degrees.
    zero_lon : float
        The longitude of the surface zero location point in degrees.

    Notes
    -----

    - The function assumes a spherical Earth, which is an acceptable
    approximation for most purposes.

    Examples
    --------
    >>> from deepimpact.damage import find_destination
    >>> find_destination(52.79, -2.95, 135, 90e3)
    (53.342, -2.95)
    """
    R = 6371000  # Radius of the Earth in meters

    # Check for edge cases
    if lat < -90 or lat > 90:
        raise ValueError("The entry latitude is out of range.")
    if lon < -180 or lon > 180:
        raise ValueError("The entry longitude is out of range.")

    bearing = math.radians(bearing)  # Convert bearing to radians

    phi1 = math.radians(lat)  # Current lat point converted to radians
    lambda1 = math.radians(lon)  # Current long point converted to radians

    # Check for edge cases at the poles
    if abs(lat) == 90:
        new_lon = lon + math.degrees(bearing)
        # Adjust new longitude to be within -180 to 180 degrees
        new_lon = (new_lon + 180) % 360 - 180
        return lat, new_lon

    # Calculate the new latitude
    sin_phi2 = math.sin(phi1) * math.cos(distance / R) + math.cos(phi1) * math.sin(
        distance / R
    ) * math.cos(bearing)

    lat2 = math.asin(sin_phi2)

    # Calculate the new longitude
    tan_lambda = (math.sin(bearing) * math.sin(distance / R) * math.cos(phi1)) / (
        math.cos(distance / R) - math.sin(phi1) * math.sin(lat2)
    )
    lon2 = math.atan(tan_lambda) + lambda1

    # Adjust if it's outside the range
    if lon2 < -math.pi or lon2 > math.pi:
        lon2 = (lon2 + math.pi) % (2 * math.pi) - math.pi

    return math.degrees(lat2), math.degrees(lon2)


def airblast_func(r, z_b, E_k, pressure):
    """
    The airblast function used to calculate the damage radius for a target
    pressure.

    Parameters:
    - r: The radius at which to calculate the airblast function.
    - z_b: The burst altitude.
    - E_k: The kinetic energy of the meteoroid.
    - pressure: The target pressure.

    Returns:
    - The value of the airblast function at the given radius.

    Examples:
    >>> airblast_func(1e3, z_b=8e3, E_k=7e3, pressure=30e3)
    0.0
    """
    term1 = 3e11 * np.power((r**2 + z_b**2) / np.power(E_k, 2 / 3), -1.3)
    term2 = 2e7 * np.power((r**2 + z_b**2) / np.power(E_k, 2 / 3), -0.57)
    return term1 + term2 - pressure


def brents_method(z_b, E_k, pressure, x0, x1, max_iter=100, tolerance=1e-5):
    """
    Brent's method for finding root of the airblast function.

    Parameters:
    - z_b: The burst altitude.
    - E_k: The kinetic energy of the meteoroid.
    - pressure: The target pressure.
    - x0: The lower bound of the interval.
    - x1: The upper bound of the interval.
    - tolerance: The tolerance for convergence.
    - max_iter: The maximum number of iterations.

    Returns:
    - The root of the function within the given interval.

    Notes
    -----
    This function is adapted from the pseudocode provided in the Wikipedia
    article on Brent's method.

    Examples:
    >>> brents_method(z_b=8e3, E_k=7e3, pressure=30e3, x0=0, x1=1e9)
    0.0
    """

    fx0 = airblast_func(x0, z_b=z_b, E_k=E_k, pressure=pressure)
    fx1 = airblast_func(x1, z_b=z_b, E_k=E_k, pressure=pressure)

    # Check if the root is within the interval
    if fx0 * fx1 >= 0:
        return 0

    if abs(fx0) < abs(fx1):
        x0, x1 = x1, x0
        fx0, fx1 = fx1, fx0

    x2, fx2 = x0, fx0

    mflag = True
    steps_taken = 0
    d = 0
    while steps_taken < max_iter and abs(x1 - x0) > tolerance:
        fx0 = airblast_func(x0, z_b=z_b, E_k=E_k, pressure=pressure)
        fx1 = airblast_func(x1, z_b=z_b, E_k=E_k, pressure=pressure)
        fx2 = airblast_func(x2, z_b=z_b, E_k=E_k, pressure=pressure)

        # Check if the result converge, then use inverse interpolation
        if fx0 != fx2 and fx1 != fx2:
            L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
            L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
            L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
            new = L0 + L1 + L2

        # Otherwise use scant method
        else:
            new = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))

        # Check if the result is within the interval, use bisection method
        if (
            (new < ((3 * x0 + x1) / 4) or new > x1)
            or (mflag is True and (abs(new - x1)) >= (abs(x1 - x2) / 2))
            or (mflag is False and (abs(new - x1)) >= (abs(x2 - d) / 2))
            or (mflag is True and (abs(x1 - x2)) < tolerance)
            or (mflag is False and (abs(x2 - d)) < tolerance)
        ):
            new = (x0 + x1) / 2
            mflag = True

        else:
            mflag = False

        fnew = airblast_func(new, z_b=z_b, E_k=E_k, pressure=pressure)
        d, x2 = x2, x1

        if (fx0 * fnew) < 0:
            x1 = new
        else:
            x0 = new

        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0

        steps_taken += 1

    return x1


def calculate_damage_radius(target_pressures, z_b, E_k):
    """
    Calculate the damage radius for a given set of target pressures, depth
    of burst, and kinetic energy.

    Parameters:
    target_pressures (list): List of target pressures in pascals.
    z_b (float): Depth of burst in meters.
    E_k (float): Kinetic energy in joules.

    Returns:
    radii (list): List of damage radii corresponding to each target pressure.

    Examples:
    >>> calculate_damage_radius([1e3, 4e3, 30e3, 50e3], z_b=8e3, E_k=7e3)
    [0.0, 0.0, 0.0, 0.0]
    """

    # Create an empty list to store the radii and iterate through the target_pressures
    radii = []
    for i in target_pressures:
        radius = brents_method(z_b, E_k, i, 0, 1e9)
        if radius == 0:
            pass
        else:
            radii.append(radius)

    return radii


def impact_risk(
    planet,
    impact_file=os.sep.join(
        (os.path.dirname(__file__), "..", "resources", "impact_parameter_list.csv")
    ),
    pressure=30.0e3,
    nsamples=None,
):
    """
    Perform an uncertainty analysis to calculate the probability for
    each affected UK postcode and the total population affected.

    Parameters
    ----------
    planet: deepimpact.Planet instance
        The Planet instance from which to solve the atmospheric entry

    impact_file: str
        Filename of a .csv file containing the impact parameter list
        with columns for 'radius', 'angle', 'velocity', 'strength',
        'density', 'entry latitude', 'entry longitude', 'bearing'

    pressure: float
        A single pressure at which to calculate the damage zone for each impact

    nsamples: int or None
        The number of iterations to perform in the uncertainty analysis.
        If None, the full set of impact parameters provided in impact_file
        is used.

    Returns
    -------
    probability: DataFrame
        A pandas DataFrame with columns for postcode and the
        probability the postcode was inside the blast radius.
    population: dict
        A dictionary containing the mean and standard deviation of the
        population affected by the impact, with keys 'mean' and 'stdev'.
        Values are floats.

    Examples:
    >>> import deepimpact
    >>> planet = deepimpact.Planet()
    >>> impact_risk(planet, impact_file="resources/impact_parameter_list.csv", pressure=30e3, nsamples=10)
    (Postcode  probability)
    """
    # check input
    if not isinstance(pressure, (int, float, complex)):
        return (False, False)

    # read senario
    data = pd.read_csv(impact_file)
    data = data.iloc[:nsamples]
    postcodes_all = []
    population_all = []

    # run model to get the postcode and popoluartion in different
    # pressure level
    for i in range(data.shape[0]):
        result = planet.solve_atmospheric_entry(
            radius=data.loc[i, "radius"],
            angle=data.loc[i, "angle"],
            strength=data.loc[i, "strength"],
            density=data.loc[i, "density"],
            velocity=data.loc[i, "velocity"],
        )
        result = planet.calculate_energy(result)
        outcome = planet.analyse_outcome(result)

        # calculate the damage radius and surface zero point
        blast_lat, blast_lon, damage_rad = damage_zones(
            outcome,
            lat=data.loc[i, "entry latitude"],
            lon=data.loc[i, "entry longitude"],
            bearing=data.loc[i, "bearing"],
            pressures=[pressure],
        )

        # get the postcode and population in the damage radius
        locators = deepimpact.GeospatialLocator()
        postcodes = locators.get_postcodes_by_radius(
            (blast_lat, blast_lon), radii=damage_rad
        )
        population = locators.get_population_by_radius(
            (blast_lat, blast_lon), radii=damage_rad
        )

        # check did the the highest pressure reach 30kp
        if len(postcodes) != 0:
            postcodes_all = postcodes_all + postcodes[-1]
            population_all.append(population[-1])

    # calculate the possibility
    element_counts = Counter(postcodes_all)
    postcodes_code = list(element_counts.keys())
    postcodes_prob = list(element_counts.values())
    postcodes_prob = [i / data.shape[0] for i in postcodes_prob]

    return (
        pd.DataFrame({"Postcode": postcodes_code, "probability": postcodes_prob}),
        {
            "mean": float(np.mean(population_all)),
            "stdev": float(np.std(population_all)),
        },
    )


def impact_risk_plot(probability):
    """
    Plot the probability of postcodes using heatmap.

    Parameters
    ----------
    probability: DataFrame
        A pandas DataFrame with columns for postcode and the
        probability the postcode was inside the blast radius.
    population: dict
        A dictionary containing the mean and standard deviation of the
        population affected by the impact, with keys 'mean' and 'stdev'.
        Values are floats.

    Returns
    -------
    weighted_heatmap:html
        A html saved in local to show the probability of different postcode
        impacted by the scenario.

    """

    # read the full postcode location
    data_post = pd.read_csv(
        os.sep.join(
            (os.path.dirname(__file__), "..", "resources", "full_postcodes.csv")
        )
    )

    # merge the data_post and the posibility
    data_with_weights = pd.merge(
        probability,
        data_post[["Postcode", "Latitude", "Longitude"]],
        on="Postcode",
        how="left",
    )

    # plot map
    m = folium.Map(
        location=[data_with_weights.Latitude[0], data_with_weights.Longitude[0]],
        zoom_start=13,
    )
    HeatMap(
        data_with_weights[["Latitude", "Longitude", "probability"]].values.tolist(),
        gradient={0.4: "blue", 0.65: "lime", 1: "red"},
        radius=15,
        blur=10,
    ).add_to(m)
    m.save("weighted_heatmap.html")

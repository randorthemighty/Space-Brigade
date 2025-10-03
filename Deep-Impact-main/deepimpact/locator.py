"""Module dealing with postcode information."""

import numpy as np
import pandas as pd
import os
from scipy.spatial import KDTree

__all__ = ["GeospatialLocator", "great_circle_distance"]


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)}
    >>> with numpy.printoptions(formatter={'all', fmt}):
        print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """
    # Ensure latlon1 and latlon2 are at least 2-dimensional
    latlon1 = np.atleast_2d(latlon1)
    latlon2 = np.atleast_2d(latlon2)

    # Validate latitude and longitude values
    for latlon in [latlon1, latlon2]:
        if np.any(latlon[:, 0] < -90) or np.any(latlon[:, 0] > 90):
            raise ValueError("Latitude must be in the range -90 to 90")
        if np.any(latlon[:, 1] < -180) or np.any(latlon[:, 1] > 180):
            raise ValueError("Longitude must be in the range -180 to 180")

    # Converting latitudes and longitudes from degrees to radians
    latlon1_rad = np.radians(latlon1)
    latlon2_rad = np.radians(latlon2)

    # Earth's radius in meters
    R = 6371000

    # Trigonometric values
    sin_lat1 = np.sin(latlon1_rad[:, 0])[:, np.newaxis]
    sin_lat2 = np.sin(latlon2_rad[:, 0])
    cos_lat1 = np.cos(latlon1_rad[:, 0])[:, np.newaxis]
    cos_lat2 = np.cos(latlon2_rad[:, 0])
    delta_lon = latlon2_rad[:, 1] - latlon1_rad[:, 1][:, np.newaxis]

    # Applying the great circle formula
    cos_d = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * np.cos(delta_lon)

    # Clipping to avoid precision issues
    cos_d = np.clip(cos_d, -1, 1)

    # Compute distance
    distance = R * np.arccos(cos_d)

    # Set distance to zero for identical points
    identical = np.all(latlon1_rad[:, np.newaxis] == latlon2_rad, axis=2)
    distance[identical] = 0

    return distance


class GeospatialLocator(object):
    """
    Class to interact with a postcode database file and a population grid file.
    """

    def __init__(
        self,
        postcode_file=os.sep.join(
            (os.path.dirname(__file__), "..", "resources", "full_postcodes.csv")
        ),
        census_file=os.sep.join(
            (
                os.path.dirname(__file__),
                "..",
                "resources",
                "UK_residential_population_2011_latlon.asc",
            )
        ),
        norm=great_circle_distance,
    ):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .asc file containing census data on a
            latitude-longitude grid.

        norm : function
            Python function defining the distance between points in
            latitude-longitude space.

        """

        self.postcode_file = postcode_file
        self.census_file = census_file
        self.norm = norm
        self.postcodes = self.load_postcode_data()
        self.census = self.load_census_data()

    def load_postcode_data(self):
        """
        Load postcode data from a CSV file. Filters out invalid latitude and longitude values.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the filtered postcode data with valid latitude and longitude values.

        Raises
        ------
        FileNotFoundError
            If the specified postcode file does not exist.

        Notes
        -----
        The method assumes the CSV file has columns 'Latitude' and 'Longitude' among others.
        Only rows with latitude values between -90 and 90 and longitude values between -180 and 180 are retained.
        """
        # Load postcode data from CSV
        if self.postcode_file:
            df = pd.read_csv(self.postcode_file)
            # filter invalid latitude and longitude values
            df = df[df["Latitude"].between(-90, 90)]
            df = df[df["Longitude"].between(-180, 180)]
            return df
        return pd.DataFrame()

    def get_postcodes_by_radius(self, X, radii):
        """
        Return postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements
            of radii to the location X.


        Examples
        --------

        >>> locator = GeospatialLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [1.5e3])
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773),
                                            [1.5e3, 4.0e3])
        """
        # Return an empty list for empty postcodes
        if self.postcodes.empty:
            return [[] for _ in radii]

        result = []
        for radius in radii:
            if radius <= 0:
                # Return an empty list for non-positive radius values
                result.append([])
                continue

            # Calculating distances to all postcodes
            distances = self.norm(self.postcodes[["Latitude", "Longitude"]].values, [X])

            # Filter postcodes within the radius
            within_radius = self.postcodes[distances[:, 0] <= radius]
            result.append(within_radius["Postcode"].tolist())

        return result

    def load_census_data(self):
        """
        Load census data from an .asc file. This file is expected to contain geographic and population data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing latitude, longitude, and population data from the census file.

        Raises
        ------
        FileNotFoundError
            If the specified census file does not exist.

        Notes
        -----
        The method reads the geographic and population data assuming a specific format for the .asc file.
        It filters out rows with 'no data' values, replacing them with a population of zero.
        """
        # Check if the file exists
        with open(self.census_file, "r") as file:
            # Headers
            ncols = int(file.readline().split()[1])
            nrows = int(file.readline().split()[1])
            nodata_value = float(file.readline().split()[1])
            # Skipping the array labels
            _ = file.readline()
            _ = file.readline()
            _ = file.readline()

            # Read the data
            data = np.loadtxt(file, dtype=float)

            # Reshape into 3 arrays as they are stacked one after the other
            data = data.reshape((-1, nrows, ncols))

            # Updating the population to 0 for missing values
            data[2][data[2] == nodata_value] = 0

            latitude = data[0].flatten()
            longitude = data[1].flatten()
            population = data[2].flatten()

            df = pd.DataFrame(
                {"Latitude": latitude, "Longitude": longitude, "Population": population}
            )

        return df

    def find_nearest_coordinates(self, X, num_coords):
        """
        Find the nearest geographic coordinates from the census data to a given point.

        Parameters
        ----------
        X : arraylike
            A pair (tuple, list) of latitude and longitude for the reference point.
        num_coords : int
            The number of nearest coordinates to find.

        Returns
        -------
        tuple
            A tuple containing two elements:
            1. An array of the nearest coordinates (latitude, longitude).
            2. An array of distances from the reference point to each of these coordinates.

        Notes
        -----
        This method uses a KDTree for efficient nearest neighbor searching.
        The distances returned are calculated using the great_circle_distance function.
        """
        # Convert census data to a KDTree for efficient nearest neighbor search
        tree = KDTree(self.census[["Latitude", "Longitude"]].values)

        # Query the tree for the nearest neighbors
        distances, indices = tree.query(X, k=num_coords)

        # Retrieve the nearest coordinates and their distances
        nearest_coords = self.census.iloc[indices][["Latitude", "Longitude"]].values
        near_dist = [self.norm([X], [coord])[0, 0] for coord in nearest_coords]

        return nearest_coords, near_dist

    def calculate_impacted_population(self, radius, grid_centers, grid_distances):
        """
        Calculate the population impacted within a specified radius around a set of grid centers.

        Parameters
        ----------
        radius : float
            The radius within which to calculate the impacted population.
        grid_centers : arraylike
            An array of latitude-longitude pairs representing grid centers.
        grid_distances : arraylike
            An array of distances from a reference point to each grid center.

        Returns
        -------
        float
            The total population impacted within the specified radius around the given grid centers.

        Notes
        -----
        This method approximates the impacted population by considering the proportion of each grid cell's area
        that falls within the specified radius. The population data is sourced from the census data loaded earlier.
        """
        # Initialize the impacted population
        impacted_populations = 0
        # Calculate the half-diagonal of the square grid
        half_diagonal = np.sqrt(2 * (500**2))

        for grid_center, distance in zip(grid_centers, grid_distances):
            # Initialize the intersection percentage
            intersection_percentage = 0

            # Check if the entire circle is within a grid cell
            if distance + radius < half_diagonal:
                intersection_percentage = (np.pi * radius**2) / (1000 * 1000)
                # Retrieve the population for this grid center
                grid_population = self.census[
                    (self.census["Latitude"] == grid_center[0])
                    & (self.census["Longitude"] == grid_center[1])
                ]["Population"].values[0]

                # Entire population of this grid is impacted
                impacted_populations += intersection_percentage * grid_population
                break  # No need to check other neighbors

            # Check if the grid intersects with the circle
            if distance <= radius + half_diagonal:
                if distance <= radius - half_diagonal:
                    # The entire grid is inside the circle
                    intersection_percentage = 1
                else:
                    # Partial intersection - this is a rough approximation
                    intersection_percentage = max(
                        0,
                        1 - (distance - (radius - half_diagonal)) / (2 * half_diagonal),
                    )

            # Retrieve the population for the grid center from the census data
            grid_population = self.census[
                (self.census["Latitude"] == grid_center[0])
                & (self.census["Longitude"] == grid_center[1])
            ]["Population"].values[0]

            # Calculate the impacted population from intersection percentage
            impacted_pop = grid_population * intersection_percentage
            impacted_populations += impacted_pop

        return impacted_populations

    def get_population_by_radius(self, X, radii):
        """
        Return the population within specific distances of input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list
            Contains the population closer than the elements of radii to
            the location X. Output should be the same shape as the radii array.

        Examples
        --------
        >>> loc = GeospatialLocator()
        >>> loc.get_population_by_radius((51.4981, -0.1773), [1e2, 5e2, 1e3])

        """
        # Calculate distances from X to each point in the census data
        self.census["distance"] = self.norm(
            self.census[["Latitude", "Longitude"]].values, [X]
        )[:, 0]

        populations_by_radius = []
        for radius in radii:
            if radius <= 0:
                populations_by_radius.append(0)
                continue

            # Sum population for points within the radius

            if radius <= 500:
                # Find the 4 nearest coordinates
                nearest_coords, distance = self.find_nearest_coordinates(X, 4)

                total_population = self.calculate_impacted_population(
                    radius, nearest_coords, distance
                )

            elif radius > 500 and radius <= 1000:
                # Find the 4 nearest coordinates
                nearest_coords, distance = self.find_nearest_coordinates(X, 10)

                total_population = self.calculate_impacted_population(
                    radius, nearest_coords, distance
                )

            else:
                total_population = self.census[self.census["distance"] <= radius][
                    "Population"
                ].sum()

            populations_by_radius.append(int(total_population))

        return populations_by_radius

"""
This module contains the atmospheric entry solver class
for the Deep Impact project
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

__all__ = ["Planet"]


class Planet:
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(
        self,
        atmos_func="exponential",
        atmos_filename=os.sep.join(
            (os.path.dirname(__file__), "..", "resources", "AltitudeDensityTable.csv")
        ),
        Cd=1.0,
        Ch=0.1,
        Q=1e7,
        Cl=1e-3,
        alpha=0.3,
        Rp=6371e3,
        g=9.81,
        H=8000.0,
        rho0=1.2,
    ):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'

        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename

        try:
            # set function to define atmoshperic density
            if atmos_func == "exponential":
                self.rhoa = lambda z: rho0 * np.exp(-z / H)
            elif atmos_func == "tabular":
                self.read_csv()
                self.rhoa = lambda x: self.interpolate_density(x)
            elif atmos_func == "constant":
                self.rhoa = lambda x: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be 'exponential', 'tabular' or 'constant'"
                )
        except NotImplementedError:
            print("atmos_func {} not implemented yet.".format(atmos_func))
            print("Falling back to constant density atmosphere for now")
            self.rhoa = lambda x: rho0

    def rk4_step(self, f, y, t, dt):
        """
        Perform a single step of the RK4 integration method.

        Parameters
        ----------
        f : callable
            The derivative function of the ODE, f(t, y), which returns the rate of change at a time t.
        y : float or array_like
            The current value of the dependent variable of the ODE.
        t : float
            The current time value.
        dt : float
            The time step to advance the solution.

        Returns
        -------
        float or ndarray
            The estimated value of the dependent variable after one time step.

        Examples
        --------
        Consider a simple ODE dy/dt = -2y, starting at y(0) = 1.

        >>> def dydt(t, y):
        ...     return -2 * y
        >>> y_start = 1
        >>> t_start = 0
        >>> dt = 0.1
        >>> planet = Planet()
        >>> planet.rk4_step(dydt, y_start, t_start, dt)
        0.8187333333333333
        """
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 2, y + k1 / 2)
        k3 = dt * f(t + dt / 2, y + k2 / 2)
        k4 = dt * f(t + dt, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve_atmospheric_entry(
        self,
        radius,
        velocity,
        density,
        strength,
        angle,
        init_altitude=100e3,
        dt=0.25,
        radians=False,
    ):
        """
        Simulate the atmospheric entry of an object, considering factors like
        fragmentation, velocity change, and trajectory alteration.

        Parameters
        ----------
        radius : float
            Radius of the object (in meters).
        velocity : float
            Initial velocity of the object (in meters per second).
        density : float
            Density of the object (in kilograms per cubic meter).
        strength : float
            Material strength of the object (in Pascals).
        angle : float
            Entry angle of the object relative to the surface (in degrees, unless `radians` is True).
        init_altitude : float, optional
            Initial altitude of the object (in meters), by default 100,000 meters (100 km).
        dt : float, optional
            Time step for the simulation (in seconds), by default 0.25 seconds.
        radians : bool, optional
            If True, interprets the angle in radians; otherwise in degrees, by default False.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the simulation results over time. Columns include time,
            velocity, mass, angle, altitude, distance, and radius.

        Examples
        --------
        Simulate an object with a radius of 0.5 meters, velocity of 12,000 m/s, density of 3000 kg/m^3,
        strength of 1e7 Pascals, and an entry angle of 45 degrees:

        >>> planet = Planet()
        >>> result = planet.solve_atmospheric_entry(0.5, 12000, 3000, 1e7, 45)
        >>> print(result.head())
        time      velocity         mass      angle       altitude     distance  radius
        0  0.00  12000.000000  1570.796327  45.000000  100000.000000     0.000000     0.5
        1  0.25  12001.687925  1570.787637  44.989491   97878.724810  2089.219435     0.5
        2  0.50  12003.361518  1570.776305  44.978971   95757.541925  4179.800297     0.5
        3  0.75  12005.016521  1570.761525  44.968438   93636.454859  6271.741069     0.5
        4  1.00  12006.647378  1570.742251  44.957893   91515.467988  8365.039379     0.5
        """

        if not radians:
            angle = np.radians(angle)

        def equations_of_motion(t, y):
            """
            Calculate the derivatives of the state variables for atmospheric entry.

            Parameters
            ----------
            t : float
                Current time in seconds.
            y : ndarray
                Array of current state variables [velocity, mass, angle, altitude, distance, radius].

            Returns
            -------
            ndarray
                Array of derivatives [dv/dt, dm/dt, dtheta/dt, dz/dt, dx/dt, dr/dt].
            """
            v, m, theta, z, x, r = y
            rho_a = self.rhoa(z)
            A = np.pi * r**2

            dvdt = (-self.Cd * rho_a * A * v**2) / (2 * m) + self.g * np.sin(theta)
            dmdt = (-self.Ch * rho_a * A * v**3) / (2 * self.Q)
            dthetadt = (
                (self.g * np.cos(theta)) / v
                - (self.Cl * rho_a * A * v) / (2 * m)
                - (v * np.cos(theta)) / (self.Rp + z)
            )
            dzdt = -v * np.sin(theta)
            dxdt = (v * np.cos(theta)) / (1 + z / self.Rp)
            drdt = (
                np.sqrt((7 / 2) * self.alpha * (rho_a / density)) * v
                if rho_a * v**2 > strength
                else 0
            )

            return np.array([dvdt, dmdt, dthetadt, dzdt, dxdt, drdt])

        y0 = np.array(
            [
                velocity,
                density * (4 / 3) * np.pi * radius**3,
                angle,
                init_altitude,
                0,
                radius,
            ]
        )
        t = 0
        results = []
        fragmented = False
        user_time_elapsed = 0.0
        results.append([t] + list(y0))

        while True:
            # dt_actual = min(dt - user_time_elapsed, 0.01)
            dt_actual = min(dt, 0.01)

            y0 = self.rk4_step(equations_of_motion, y0, t, dt_actual)
            t += dt_actual
            # user_time_elapsed = dt
            user_time_elapsed += dt_actual

            if y0[1] <= 0 or y0[3] <= 0 or y0[0] < 0:
                break
            if len(results) > 0 and y0[3] > results[-1][4]:
                break

            # Check for height changes when the cumulative time meets or
            # exceeds theuser-defined dt
            if user_time_elapsed >= dt:
                # If the previous result exists and the height change
                # is less than 1, the simulation is stopped
                if len(results) > 0 and abs(y0[3] - results[-1][4]) < 1:
                    break
                results.append([t] + list(y0))
                user_time_elapsed = 0.0

            ram_pressure = self.rhoa(y0[3]) * y0[0] ** 2
            if ram_pressure > strength:
                fragmented = True
            elif fragmented and ram_pressure <= strength:
                fragmented = False

        result_df = pd.DataFrame(
            results,
            columns=[
                "time",
                "velocity",
                "mass",
                "angle",
                "altitude",
                "distance",
                "radius",
            ],
        )

        # Converts the angle column in the result from radians to degrees
        result_df["angle"] = np.degrees(result_df["angle"])

        return result_df

    def calculate_energy(self, result):
        """
        Calculate the kinetic energy and its variation per unit altitude of an object.

        This method computes the kinetic energy in kilotons of TNT and the rate of
        energy dissipation per kilometer of altitude change. It adds or updates the
        'dedz' column in the provided DataFrame, representing the energy dissipation rate.

        Parameters
        ----------
        result : DataFrame
            A pandas DataFrame with columns 'mass', 'velocity', and 'altitude', representing
            the mass in kilograms, velocity in meters per second, and altitude in meters,
            respectively, of an object at various time steps.

        Returns
        -------
        DataFrame
            The input DataFrame with an additional or updated column 'dedz', representing
            the rate of energy dissipation per kilometer.

        Examples
        --------
        >>> planet = Planet()
        >>> entry_result = planet.solve_atmospheric_entry(0.5, 12000, 3000, 1e7, 45)
        >>> energy_result = planet.calculate_energy(entry_result)
        >>> print(energy_result.head())
        time      velocity         mass      angle       altitude     distance  radius      dedz
        0  0.00  12000.000000  1570.796327  45.000000  100000.000000     0.000000     0.5  0.000000
        1  0.25  12001.687925  1570.787637  44.989491   97878.724810  2089.219435     0.5 -0.000004
        2  0.50  12003.361518  1570.776305  44.978971   95757.541925  4179.800297     0.5 -0.000003
        3  0.75  12005.016521  1570.761525  44.968438   93636.454859  6271.741069     0.5 -0.000003
        4  1.00  12006.647378  1570.742251  44.957893   91515.467988  8365.039379     0.5 -0.000003
        """

        # Calculate the kinetic energy
        kinetic_energy = 0.5 * result["mass"] * result["velocity"] ** 2

        # Convert kinetic energy from Joules to kilotons of TNT
        kinetic_energy_kt = kinetic_energy / 4.184e12

        # Calculate the energy difference between successive steps
        energy_diff = np.diff(kinetic_energy_kt, prepend=kinetic_energy_kt[0])

        # Calculate the altitude difference between successive steps
        altitude_diff = np.diff(result["altitude"], prepend=result["altitude"][0])

        small_value = 1e-6  # This can be adjusted as needed
        altitude_diff[altitude_diff == 0] = small_value

        # Calculate dedz, convert from per meter to per kilometer
        dedz = energy_diff / (altitude_diff / 1000)

        # Update or create the 'dedz' column
        if "dedz" in result.columns:
            result["dedz"] = dedz
        else:
            result.insert(len(result.columns), "dedz", dedz)

        return result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats.

        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius, and dedz as a function of time.

        Returns
        -------
        outcome : Dict
            Dictionary with details of the impact event. This should contain
            the key:

            ``outcome``
                which should contain one of the following strings:
                ``Airburst`` or ``Cratering``,

            and also the following 4 keys:

            ``burst_peak_dedz``, ``burst_altitude``,
            ``burst_distance``, ``burst_energy``.
        """

        outcome = {
            "outcome": "Unknown",
            "burst_peak_dedz": 0.0,
            "burst_altitude": 0.0,
            "burst_distance": 0.0,
            "burst_energy": 0.0,
        }
        # Check if the DataFrame is empty
        if result.empty:
            return outcome

        # Find the index of the maximum energy deposition rate
        max_dedz_idx = result["dedz"].idxmax()
        max_dedz = result.loc[max_dedz_idx, "dedz"]

        # Check if the max energy deposition occurs at an altitude above 0
        max_dedz_altitude = result.loc[max_dedz_idx, "altitude"]
        if max_dedz_altitude > 0:
            outcome["outcome"] = "Airburst"
            outcome["burst_peak_dedz"] = max_dedz
            outcome["burst_altitude"] = max_dedz_altitude
            outcome["burst_distance"] = result.loc[max_dedz_idx, "distance"]

            # Calculate the kE loss from initial altitude to burst altitude
            initial_kinetic_energy = (
                0.5 * result.loc[0, "mass"] * result.loc[0, "velocity"] ** 2
            )
            burst_kinetic_energy = (
                0.5
                * result.loc[max_dedz_idx, "mass"]
                * (result.loc[max_dedz_idx, "velocity"] ** 2)
            )
            energy_loss = initial_kinetic_energy - burst_kinetic_energy

            # Calculate burst energy
            residual_energy_at_burst = initial_kinetic_energy - energy_loss
            outcome["burst_energy"] = (
                max(energy_loss, residual_energy_at_burst) / 4.184e12
            )
        else:
            outcome["outcome"] = "Cratering"
            # For cratering, determine the specifics at point of ground impact
            impact_index = result[result["altitude"] <= 0].index[0]
            initial_kinetic_energy = (
                0.5 * result.loc[0, "mass"] * result.loc[0, "velocity"] ** 2
            )
            residual_kinetic_energy_at_impact = (
                0.5
                * result.loc[impact_index, "mass"]
                * (result.loc[impact_index, "velocity"] ** 2)
            )

            # Set burst_peak_dedz to the dedz value at impact
            outcome["burst_peak_dedz"] = result.loc[impact_index, "dedz"]
            # Set burst_altitude to 0, as the burst happens at ground level
            outcome["burst_altitude"] = 0
            # Set burst_distance to the horizontal distance at impact
            outcome["burst_distance"] = result.loc[impact_index, "distance"]
            # Calculate burst_energy
            outcome["burst_energy"] = (
                max(
                    (initial_kinetic_energy - residual_kinetic_energy_at_impact),
                    residual_kinetic_energy_at_impact,
                )
                / 4.184e12
            )

        return outcome

    def read_csv(self):
        """
        Read atmospheric data from a CSV file and initialize interpolation.

        This method reads altitude and density data from a CSV file specified by
        `self.atmos_filename`. It initializes an interpolator for density as a function
        of altitude using cubic interpolation.

        The CSV file is expected to have two columns: altitude and density, with a header row.
        """
        with open(self.atmos_filename, "r") as file:
            next(file)  # Skip the header line
            data = np.loadtxt(file)
            self.altitudes = data[:, 0]
            self.densities = data[:, 1]
            self.interpolator = interp1d(
                self.altitudes,
                self.densities,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

    def interpolate_density(self, x):
        """
        Interpolate the atmospheric density at a given altitude.

        Parameters
        ----------
        x : float or array_like
            The altitude(s) at which to interpolate density.

        Returns
        -------
        float or ndarray
            The interpolated density value(s) at the given altitude(s).

        Examples
        --------
        >>> planet = Planet()
        >>> planet.read_csv()
        >>> density_at_10km = planet.interpolate_density(10000)
        >>> print(density_at_10km)
        0.41270699999999994
        """
        return self.interpolator(x)

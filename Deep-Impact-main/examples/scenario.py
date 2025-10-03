import deepimpact
import os


#######################
#   Airburst Solver   #
#######################
def final_scenario(earth):
    # Solve the atmospheric entry problem for a given set of input parameters
    result = earth.solve_atmospheric_entry(
        radius=90, angle=30, strength=2e6, density=2500, velocity=21e3
    )

    # Calculate the kinetic energy lost per unit altitude and add it
    # as a column to the result dataframe
    result = earth.calculate_energy(result)

    # Determine the outcomes of the impact event
    outcome = earth.analyse_outcome(result)

    #####################
    #   Damage Mapper   #
    #####################

    # Calculate the blast location and damage radius
    pressures = [1e3, 4e3, 30e3, 50e3]

    blast_lat, blast_lon, damage_rad = deepimpact.damage_zones(
        outcome, lat=55.35, lon=-1.7, bearing=290.0, pressures=pressures
    )

    # Plot a circle to show the limit of the lowest damage level
    damage_map = deepimpact.plot_circle(blast_lat, blast_lon, damage_rad[0])
    damage_map.save("damage_map.html")

    # The GeospatialLocator tool
    locator = deepimpact.GeospatialLocator()

    # Find the postcodes in the damage radii
    postcodes = locator.get_postcodes_by_radius(
        (blast_lat, blast_lon), radii=damage_rad
    )

    # Find the population in each postcode
    population = locator.get_population_by_radius(
        (blast_lat, blast_lon), radii=damage_rad
    )

    # Print the number of people affected in each damage zone
    print()
    print("Pressure |      Damage | Population")
    print("   (kPa) | radius (km) |   affected")
    print("-----------------------------------")
    for pop, rad, zone in zip(population, damage_rad, pressures):
        print(f"{zone/1e3:8.0f} | {rad/1e3:11.1f} | {pop:10,.0f}")
    print()

    # Construct the file path
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "examples", "final_output.txt"
    )

    with open(output_path, "w") as file:
        file.write("\n")
        file.write("Pressure |      Damage | Population\n")
        file.write("   (kPa) | radius (km) |   affected\n")
        file.write("-----------------------------------\n")
        for pop, rad, zone in zip(population, damage_rad, pressures):
            file.write(f"{zone/1e3:8.0f} | {rad/1e3:11.1f} | {pop:10,.0f}\n")
        file.write("\n")

    # Write postcodes to a CSV file
    with open("postcodes.txt", "w", newline="") as file:
        # Writing the number of total postcodes impacted
        file.write("Total Postcodes impacted: " + str(len(postcodes[0])) + "\n")
        file.write("\n")

        # Writing the number of highest damaged postcodes
        file.write("Highest damaged Postcodes: " + str(len(postcodes[-1])) + "\n")
        file.write("\n")

        # Writing the header for highest damaged postcodes
        file.write("Highest damaged Postcodes:\n")

        # Writing each highest damaged postcode on a new line
        for postcode in postcodes[-1]:
            file.write(postcode + "\n")


# Initialise the Planet class
earth = deepimpact.Planet()
final_scenario(earth)

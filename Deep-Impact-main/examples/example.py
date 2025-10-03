import deepimpact

#######################
#   Airburst Solver   #
#######################

# Initialise the Planet class
earth = deepimpact.Planet()

# Solve the atmospheric entry problem for a given set of input parameters
result = earth.solve_atmospheric_entry(
    radius=35, angle=45, strength=1e7, density=3000, velocity=19e3
)

# Calculate the kinetic energy lost per unit altitude and add it
# as a column to the result dataframe
result = earth.calculate_energy(result)

# Determine the outcomes of the impact event
outcome = earth.analyse_outcome(result)

#####################
#   Damage Mapper   #
#####################

# Calculate the blast location and damage radius for several pressure levels
pressures = [1e3, 4e3, 30e3, 50e3]

blast_lat, blast_lon, damage_rad = deepimpact.damage_zones(
    outcome, lat=55.2, lon=-2.5, bearing=217.0, pressures=pressures
)

# Plot a circle to show the limit of the lowest damage level
damage_map = deepimpact.plot_circle(blast_lat, blast_lon, damage_rad[0])
damage_map.save("damage_map.html")

# The GeospatialLocator tool
locator = deepimpact.GeospatialLocator()

# Find the postcodes in the damage radii
postcodes = locator.get_postcodes_by_radius((blast_lat, blast_lon), radii=damage_rad)

# Find the population in each postcode
population = locator.get_population_by_radius((blast_lat, blast_lon), radii=damage_rad)

# Print the number of people affected in each damage zone
print()
print("Pressure |      Damage | Population")
print("   (kPa) | radius (km) |   affected")
print("-----------------------------------")
for pop, rad, zone in zip(population, damage_rad, pressures):
    print(f"{zone/1e3:8.0f} | {rad/1e3:11.1f} | {pop:10,.0f}")
print()

# Print the postcodes inside the highest damage zone
print("Postcodes in the highest damage zone:")
print(*postcodes[-1])
print()

# Example usage of impact_risk function
# Uses the default file impact_parameter_list.csv in the resources folder
probability, population = deepimpact.impact_risk(earth, pressure=30e3)

# Sort the probability DataFrame by the 'Probability' col in descending order
probability_sorted = probability.sort_values(by="probability", ascending=False)

print(probability_sorted.head())
print(probability.head())
print(
    "Total population affected: "
    + f"{population['mean']:,.0f} +/- {population['stdev']:,.0f}"
)

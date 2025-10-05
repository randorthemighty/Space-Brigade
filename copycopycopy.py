import deepimpact as di
import math
#### General idea: Will receive variables, needs to take them, 

### RECEIVE DATA FROM API HERE ###


### ACTUAL CALCULATION PART ###
### The Catalyst: Deep Impact requires 8 parameters: Radius (meters), Angle (Degrees), Strength (Pascals [Kg/(m*s^2)]), Density (Kg/m^3),
### Latitude, Longitude, and Bearing [the last three only affect meteor trajectory].

# We start by taking API data and converting it into units fit for Deep-Impact
radius = diameter*500 #API measures Diameter in km

volume = (4/3)*(math.pi)*radius^3

strength = megatons*4.184e15*(1/volume)

density = mass/volume
# So right now our EEIO model will look like this:
#
#               agriculture  industry  services
# agriculture       1.0        0.5       0.1
# industry          0.2        1.0       0.2
# services          0.0        0.1       1.0
#
# CO2               0.1        0.2      0.05
# CH4               0.2        0.1      0.01

import numpy as np
from eeio_demo.calc import inv, scale_diagonal_to_unit
from eeio_demo.model import IO_MATRIX, INTERVENTION_MATRIX

# This matrices represent the direct requirements of sector or emissions i
# to produce one euro of sector j.
#
# How can we know not only the direct requirements, but also the upstream requirements?
# We can use the Leontief inverse to compute the upstream requirements.
# The Leontief inverse is the inverse of the Leontief input-output matrix.
# The Leontief inverse is a matrix that represents the upstream requirements of sector i
# to produce one euro of sector j.
#
# We can compute the Leontief inverse using the inv function from the calc module
# and rescale the columns to 1 euro to better understand the results:

leontief_inverse = inv(IO_MATRIX)
scaled_leontief_inverse = scale_diagonal_to_unit(leontief_inverse)

# The result looks like this:
#
#               agriculture      industry      services
# agriculture       1.0           -0.49          0.0
# industry        -0.204           1.0          -0.2
# services        0.0204          -0.1           1.0
#
# This matrix represents the total requirements (direct and upstream) of sector i
# to produce 1 euro of sector j. Then, for producing 1 euro of agriculture,
# considering the whole value chain, we need an input of 0.0204 euros from services
# and an output of 0.204 euros to industry.

# How can we get the upstream emissions? Easy, we can get the Leontief inverse matrix
# and multiply it by the intervention matrix. What we get is the intensity matrix:

intensity_matrix = np.matmul(leontief_inverse, INTERVENTION_MATRIX.transpose())

# The result looks like this:
#
#               agriculture      industry      services
# CO2               0.0            0.19          0.03
# CH4              0.17            0.07         0.003
#
# This matrix represents the total emissions (direct and upstream) of flow i
# to produce 1 euro of sector j. Then, for producing 1 euro of agriculture,
# considering the whole value chain, we emit 0.17 kg of CH4.
#
# So now we have our life cycle inventory of our economy model, but we can also
# use it as a background system to calculate the inventory for a new activity, that
# for example has inputs of 0.1 from agriculture, 0.2 from industry and 0.3 from services.
# This would be our demand vector:

demand_vector = np.array([0.1, 0.2, 0.3])

# Therefore we can calculate our supply vector by multiplying the demand vector
# by the Leontief inverse matrix:

supply_vector = np.matmul(demand_vector, leontief_inverse)

# The result is:
#
#               agriculture      industry      services
# supply           0.07           0.14           0.26
#
# This means that to produce 1 euro of our new activity, and considering the whole value chain,
# we need 0.07 euros from agriculture, 0.14 euros from industry and 0.26 euros from services.
#
# So we can get the emissions of our new activity, aka life cycle inventory, by multiplying
# the supply vector by the intervention matrix:

lci = np.matmul(supply_vector, INTERVENTION_MATRIX.transpose())

# The result is:
#
#               CO2            CH4
# lci          0.05           0.03
#
# So we have there the total emission from our new modelled activity.

# How can we get the emissions factors of our economy model and our new activity?
# For that we need an impact assessment method, for example the IPCC method. Which
# is a method that calculates the global warming potential in kg of CO2 eq. For that
# it provides a set of characterization factors, one for each elementary flow. In our
# example we have two elementary flows, CO2 and CH4 and the characterization factors are:
#
#               CO2            CH4
# cf            1.0           30.5

cf = np.array([1.0, 30.5])

# So the emission factor for our new activity is computed by multiplying the lci vector
# by the characterization factors:

ef = np.matmul(lci, cf)

# The result is 0.98 kg CO2 eq per euro.
#
# And we can also calculate the emission factors of our economy model by multiplying
# the intensity matrix by the characterization factors:

ef_model = np.matmul(intensity_matrix, cf)

# The result is:
#
#               agriculture      industry      services
# ef               5.08           2.21           0.13

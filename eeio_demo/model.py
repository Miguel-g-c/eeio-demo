# Let's say that we have a simple economy with only three sectors:
# agriculture, industry, and services.
# We will assume that the economy is closed,
# so that all the interactions happen in the same country.
# We can represent the input output economy as a matrix like this one:
#
#               agriculture  industry  services
# agriculture       1.0        0.5       0.1
# industry          0.2        1.0       0.2
# services          0.0        0.1       1.0
#
# The rows represent the sectors that produce the goods and services,
# and the columns represent the sectors that consume them.

import numpy as np

IO_MATRIX = np.array(
    [
        [1.0, 0.5, 0.1],
        [0.2, 1.0, 0.2],
        [0.0, 0.1, 1.0],
    ]
)

# This matrix represents the direct requirements of sector i to produce one euro of sector j.
# This way, for producing 1 euro of agriculture,
# we need 0.5 euros of industry and 0.1 euros of services.

# So this is our economic IO model which we can extend to include environmental emissions.
# For that we need another matrix, the intervention matrix or satellite accounts.
# In our example, we assume that we only have two types of emissions, CO2 and CH4.
# Then the EEIO model will look like this:
#
#               agriculture  industry  services
# agriculture       1.0        0.5       0.1
# industry          0.2        1.0       0.2
# services          0.0        0.1       1.0
#
# CO2               0.1        0.2      0.05
# CH4               0.2        0.1      0.01
#
# The botton matrix represents the direct emissions (+) or resources (-)
# of flow i to produce 1 euro of sector j. So our agriculture sector
# emits 0.1 kg CO2 and 0.2 kg CH4 to produce 1 euro.

INTERVENTION_MATRIX = np.array(
    [
        [0.1, 0.2, 0.05],
        [0.2, 0.1, 0.01],
    ]
)

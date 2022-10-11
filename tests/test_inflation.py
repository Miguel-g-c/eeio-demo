import numpy as np
from eeio_demo.model import IO_MATRIX, INTERVENTION_MATRIX
from eeio_demo.calc import inv


def test_that_applying_inflation_rate_directly_to_emissions_factors_doesnt_matter():
    # Given our EEIO model for 2020:
    #
    #               agriculture  industry  services
    # agriculture       1.0        0.5       0.1
    # industry          0.2        1.0       0.2
    # services          0.0        0.1       1.0
    #
    # CO2               0.1        0.2      0.05
    # CH4               0.2        0.1      0.01
    #
    # And the following inflation rates for 2021 per sector:
    #
    #               agriculture  industry  services
    # inflation        0.15        0.2       0.1
    #
    # When updating the IO matrix with the inflation rates and calculating the
    # emissions factors, the results should be the same than updating
    # the emissions factors directly.

    cf = np.array([1.0, 30.5])

    leontief_inverse = inv(IO_MATRIX)
    intensity_matrix = leontief_inverse @ INTERVENTION_MATRIX.transpose()
    emissions_factors = intensity_matrix @ cf

    updated_model = IO_MATRIX * [1.15, 1.2, 1.1]
    updated_leontief_inverse = inv(updated_model)
    updated_intensity_matrix = (
        updated_leontief_inverse @ INTERVENTION_MATRIX.transpose()
    )
    updated_emissions_factors = updated_intensity_matrix @ cf

    directly_updated_emissions_factors = emissions_factors * np.array(
        [
            1 / 1.15,
            1 / 1.2,
            1 / 1.1,
        ]
    )

    assert np.allclose(updated_emissions_factors, directly_updated_emissions_factors)

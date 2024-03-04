import numpy as np

from model.RICE_model.model_limits import ModelLimits


class CarbonSubmodel:
    def __init__(self, years):
        self.years = years
        self.simulation_horizon = len(self.years)
        self.delta_t = self.years[1] - self.years[0]  # Assuming equally spaced intervals between the years
        self.limits = ModelLimits()

        self.mat = np.zeros((self.simulation_horizon,))
        self.mup = np.zeros((self.simulation_horizon,))
        self.ml = np.zeros((self.simulation_horizon,))
        self.forcoth = np.zeros((self.simulation_horizon,))
        self.forc = np.zeros((self.simulation_horizon,))

        # RICE2010 INPUTS
        # Initial concentration in atmosphere 2000 [GtC]
        self.mat0 = 787
        # Initial concentration in atmosphere 2010 [GtC]
        self.mat1 = 829
        # Initial concentration in upper strata [GtC]
        self.mu0 = 1600.0  # 1600 in excel
        # Initial concentration in lower strata [GtC]
        self.ml0 = 10010.0

        # Carbon pools
        self.mat[0] = self.mat0
        self.mat[1] = self.mat1

        if self.mat[0] < self.limits.mat_lo:
            self.mat[0] = self.limits.mat_lo

        self.mup[0] = self.mu0
        if self.mup[0] < self.limits.mup_lo:
            self.mup[0] = self.limits.mup_lo

        self.ml[0] = self.ml0
        if self.ml[0] < self.limits.ml_lo:
            self.ml[0] = self.limits.ml_lo

        # 2000 forcings of non-CO2 greenhouse gases (GHG) [Wm-2]
        self.fex0 = -0.06
        # 2100 forcings of non-CO2 GHG [Wm-2]
        self.fex1 = 0.30
        # Forcings of equilibrium CO2 doubling [Wm-2]
        self.fco22x = 3.8

        # Radiative forcing
        self.forcoth[0] = self.fex0
        self.forc[0] = (
                self.fco22x
                * (np.log(((self.mat[0] + self.mat[1]) / 2) / 596.40) / np.log(2.0))
                + self.forcoth[0]
        )

        # Exogenous forcings from other GHG
        # rises linearly from 2010 to 2100 from -0.060 to 0.3 then becomes stable in RICE -  UPDATE FOR DICE2016R

        self.exo_forcing_2000 = -0.060
        self.exo_forcing_2100 = 0.3000

        years_before_2100 = [i for i in self.years if i <= 2100]  # years
        years_after_2100 = [i for i in self.years if i > 2100]  # years
        # Compute linearly increasing exogenous forcing from other GHGs until 2100
        for i, year in enumerate(years_before_2100):
            self.forcoth[i] = self.fex0 + (self.delta_t / 100) * (self.exo_forcing_2100 - self.exo_forcing_2000) * i
        # Compute static exogenous forcing from other GHGs from 2100 until end of simulation period
        for j, year in enumerate(years_after_2100):
            self.forcoth[len(years_before_2100) + j] = self.exo_forcing_2100

        self.E_worldwide_per_year = np.zeros(self.simulation_horizon)

    def run(self, t, year, E):
        def calculate_worldwide_yearly_atmospheric_carbon_conc_increase(t, E):
            # self.E_worldwide_per_year[t] = E[:, t].sum(axis=0)
            self.E_worldwide_per_year = E.sum(axis=0)
            return self.E_worldwide_per_year

        def calculate_biospehere_and_upper_ocean_carbon_conc(t):
            self.mup[t] = (
                    12 / 100 * self.mat[t-1]
                    + 94.796 / 100 * self.mup[t-1]
                    + 0.075 / 100 * self.ml[t-1])

            # set lower constraint for shallow ocean concentration
            if self.mup[t] < self.limits.mup_lo:
                self.mup[t] = self.limits.mup_lo
            return self.mup

        def calculate_lower_oceans_carbon_conc(t):
            self.ml[t] = 99.925 / 100 * self.ml[t-1] + 0.5 / 100 * self.mup[t-1]

            # set lower constraint for shallow ocean concentration
            if self.ml[t] < self.limits.ml_lo:
                self.ml[t] = self.limits.ml_lo
            return self.ml

        def calculate_radiative_forcing(t, year):
            # calculate concentration in atmosphere for t + 1 (because of averaging in forcing formula
            if year < self.years[-2]:
                self.mat[t+1] = (
                        88 / 100 * self.mat[t]
                        + 4.704 / 100 * self.mup[t]
                        + self.E_worldwide_per_year[t] * self.delta_t
                )

                # set lower constraint for atmospheric concentration
                if self.mat[t+1] < self.limits.mat_lo:
                    self.mat[t+1] = self.limits.mat_lo

                # Radiative forcing
                # Increase in radiative forcing [Wm-2 from 1900]
                # forcing = constant * Log2( current concentration / concentration of forcing in 1900 at a
                # doubling of CO2 (η)[◦C/2xCO2] ) + external forcing
                self.forc[t] = (
                        self.fco22x * (np.log(((self.mat[t] + self.mat[t+1]) / 2) / (280 * 2.13)) / np.log(2.0)) + self.forcoth[t]
                )
            else:
                self.mat[t] = (
                        88 / 100 * self.mat[t-1]
                        + 4.704 / 100 * self.mup[t-1]
                        + self.E_worldwide_per_year[t-1] * self.delta_t
                )

                # set lower constraint for atmospheric concentration
                if self.mat[t] < self.limits.mat_lo:
                    self.mat[t] = self.limits.mat_lo

                self.forc[t] = (
                        self.fco22x * (np.log((self.mat[t]) / (280 * 2.13)) / np.log(2.0)) + self.forcoth[t]
                )

            return self.forc, self.mat

        # Carbon concentration increase in atmosphere [GtC from 1750]
        self.E_worldwide_per_year = calculate_worldwide_yearly_atmospheric_carbon_conc_increase(t, E)
        # Skip initialisation step
        if t >= 1:
            # calculate concentration in bioshpere and upper oceans
            self.mup = calculate_biospehere_and_upper_ocean_carbon_conc(t)
            # Carbon concentration increase in lower oceans [GtC from 1750]
            self.ml = calculate_lower_oceans_carbon_conc(t)
            # Calculate radiatve forcing
            self.forc, self.mat = calculate_radiative_forcing(t, year)
        return

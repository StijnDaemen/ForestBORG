import numpy as np
import os
import json
from scipy.stats import norm, cauchy, lognorm
from model.RICE_model.model_limits import ModelLimits


class ClimateSubmodel:
    def __init__(self, years, regions, uncertainty_dict=None):
        self.years = years
        self.simulation_horizon = len(self.years)
        self.delta_t = self.years[1] - self.years[0]  # Assuming equally spaced intervals between the years
        self.regions = regions
        self.n_regions = len(regions)
        self.limits = ModelLimits()
        self.uncertainty_dict = uncertainty_dict
        # Ensure this variable is set to the same value in the carboncycle module
        self.fco22x = 3.8

        # Increase temperature of atmosphere [dC from 1900]
        self.temp_atm = np.zeros(self.simulation_horizon)
        # Increase temperature of lower oceans [dC from 1900]
        self.temp_ocean = np.zeros(self.simulation_horizon)
        # SLR parameters
        self.THERMEQUIL = np.zeros(self.simulation_horizon)
        self.SLRTHERM = np.zeros(self.simulation_horizon)
        self.GSICREMAIN = np.zeros(self.simulation_horizon)
        self.GSICCUM = np.zeros(self.simulation_horizon)
        self.GSICMELTRATE = np.zeros(self.simulation_horizon)
        self.GISREMAIN = np.zeros(self.simulation_horizon)
        self.GISMELTRATE = np.zeros(self.simulation_horizon)
        self.GISEXPONENT = np.zeros(self.simulation_horizon)
        self.GISCUM = np.zeros(self.simulation_horizon)
        self.AISREMAIN = np.zeros(self.simulation_horizon)
        self.AISMELTRATE = np.zeros(self.simulation_horizon)
        self.AISCUM = np.zeros(self.simulation_horizon)
        self.TOTALSLR = np.zeros(self.simulation_horizon)
        self.SLRDAMAGES = np.zeros((len(self.regions), self.simulation_horizon))

        # Initial lower stratum temperature change [dC from 1900]
        self.tocean0 = 0.0068
        # Initial atmospheric temperature change [dC from 1900]
        self.tatm0 = 0.83

        # Atmospheric temperature
        self.temp_atm[0] = self.tatm0
        self.temp_atm[1] = 0.980

        if self.temp_atm[0] < self.limits.temp_atm_lo:
            self.temp_atm[0] = self.limits.temp_atm_lo
        if self.temp_atm[0] > self.limits.temp_atm_up:
            self.temp_atm[0] = self.limits.temp_atm_up

        # Oceanic temperature
        self.temp_ocean[0] = self.tocean0 #  0.007

        if self.temp_ocean[0] < self.limits.temp_ocean_lo:
            self.temp_ocean[0] = self.limits.temp_ocean_lo
        if self.temp_ocean[0] > self.limits.temp_ocean_up:
            self.temp_ocean[0] = self.limits.temp_ocean_up

        self.samples_t2xco2 = self.set_up_climate_sensitivity_distributions()
        if self.uncertainty_dict:
            self.t2xco2_index = self.uncertainty_dict['t2xco2_index'] # -1
            self.t2xco2_dist = self.uncertainty_dict['climate_sensitivity_distribution']  # 0
        else:
            self.t2xco2_index = -1
            self.t2xco2_dist = 0
        # Equilibrium temperature impact [dC per doubling CO2]/(3.2 RICE OPT)
        self.t2xco2 = self.samples_t2xco2[self.t2xco2_dist][self.t2xco2_index]

        # inputs
        self.therm0 = 0.092066694
        self.thermadj = 0.024076141
        self.thermeq = 0.5

        self.gsictotal = 0.26
        self.gsicmelt = 0.0008
        self.gsicexp = 1
        self.gsieq = -1

        self.gis0 = 7.3
        self.gismelt0 = 0.6

        self.aiswais = 5
        self.aisother = 51.6

        # Initial definitions series

        self.THERMEQUIL[0] = self.temp_atm[0] * self.thermeq
        self.SLRTHERM[0] = self.therm0 + self.thermadj * (self.THERMEQUIL[0] - self.therm0)

        self.GSICREMAIN[0] = self.gsictotal

        self.GSICMELTRATE[0] = (self.gsicmelt * self.delta_t * (self.GSICREMAIN[0] / self.gsictotal) ** self.gsicexp * (self.temp_atm[0] - self.gsieq))
        self.GSICCUM[0] = self.GSICMELTRATE[0]

        self.GISREMAIN[0] = self.gis0
        self.GISMELTRATE[0] = self.gismelt0
        self.GISCUM[0] = self.gismelt0 / 100
        self.GISEXPONENT[0] = 1

        self.AISREMAIN[0] = self.aiswais + self.aisother
        self.AISMELTRATE[0] = 0.1225
        self.AISCUM[0] = self.AISMELTRATE[0] / 100

        self.TOTALSLR[0] = (self.SLRTHERM[0] + self.GSICCUM[0] + self.GISCUM[0] + self.AISCUM[0])

        self.SLRDAMAGES[:, 0] = 0

    def run(self, t, year, forc, gross_output):
        def calculate_atmospheric_temperature(t, forc):
            # Climate equation coefficient for upper level
            c1 = 0.208
            # Transfer coefficient upper to lower stratum
            c3 = 0.310

            if t > 1:
                self.temp_atm[t] = self.temp_atm[t-1] + c1 * (
                        (forc[t] - ((self.fco22x / self.t2xco2) * self.temp_atm[t-1]))
                        - (c3 * (self.temp_atm[t-1] - self.temp_ocean[t-1]))
                )


            # setting up lower and upper bound for temperatures
            if self.temp_atm[t] < self.limits.temp_atm_lo:
                self.temp_atm[t] = self.limits.temp_atm_lo

            if self.temp_atm[t] > self.limits.temp_atm_up:
                self.temp_atm[t] = self.limits.temp_atm_up

            return self.temp_atm

        def calculate_ocean_temp(t):
            # Transfer coefficient for lower level
            c4 = 0.05

            self.temp_ocean[t] = self.temp_ocean[t-1] + c4 * (
                    self.temp_atm[t-1] - self.temp_ocean[t-1]
            )

            # setting up lower and upper bound for temperatures
            if self.temp_ocean[t] < self.limits.temp_ocean_lo:
                self.temp_ocean[t] = self.limits.temp_ocean_lo

            if self.temp_ocean[t] > self.limits.temp_ocean_up:
                self.temp_ocean[t] = self.limits.temp_ocean_up
            return self.temp_ocean

        # def calculate_thermal_expension(SLRTHERM, temp_atm):
        #     thermadj = 0.024076141
        #     thermeq = 0.5
        #
        #     THERMEQUIL = temp_atm[1] * thermeq
        #
        #     SLRTHERM[1] = SLRTHERM[0] + thermadj * (THERMEQUIL - SLRTHERM[0])
        #     return SLRTHERM

        def calculate_total_SLR(t, year):
            # SLR thermal
            thermadj = 0.024076141
            thermeq = 0.5

            self.THERMEQUIL[t] = self.temp_atm[t] * thermeq
            self.SLRTHERM[t] = self.SLRTHERM[t-1] + thermadj * (self.THERMEQUIL[t] - self.SLRTHERM[t-1])

            # GSICCUM
            # glacier ice cap
            gsictotal = 0.26
            gsicmelt = 0.0008
            gsicexp = 1

            self.GSICREMAIN[t] = gsictotal - self.GSICCUM[t-1]
            self.GSICMELTRATE[t] = (gsicmelt * self.delta_t * (self.GSICREMAIN[t] / gsictotal) ** gsicexp * self.temp_atm[t])
            self.GSICCUM[t] = self.GSICCUM[t-1] + self.GSICMELTRATE[t]

            # GISCUM
            # greenland
            gis0 = 7.3
            gismelt0 = 0.6
            gismeltabove = 1.118600816
            gismineq = 0
            gisexp = 1

            self.GISREMAIN[t] = self.GISREMAIN[t-1] - (self.GISMELTRATE[t-1] / 100)
            if t == 1:
                self.GISMELTRATE[t] = 0.60
            else:
                self.GISMELTRATE[t] = (gismeltabove * (self.temp_atm[t] - gismineq) + gismelt0) * self.GISEXPONENT[t-1]
            self.GISCUM[t] = self.GISCUM[t - 1] + self.GISMELTRATE[t] / 100
            if t == 1:
                self.GISEXPONENT[t] = 1
            else:
                self.GISEXPONENT[t] = 1 - (self.GISCUM[t] / gis0) ** gisexp

            # AISCUM
            # antartica ice cap
            aismelt0 = 0.21
            aismeltlow = -0.600407185
            aismeltup = 2.225420209
            aisratio = 1.3
            aisinflection = 0
            aisintercept = 0.770332789

            if year <= 2115:
                if self.temp_atm[t] < 3.0:
                    self.AISMELTRATE[t] = (aismeltlow * self.temp_atm[t] * aisratio + aisintercept)
                else:
                    self.AISMELTRATE[t] = (aisinflection * aismeltlow + aismeltup * (self.temp_atm[t] - 3.0) + aisintercept)
            else:
                if self.temp_atm[t] < 3.0:
                    self.AISMELTRATE[t] = (aismeltlow * self.temp_atm[t] * aisratio + aismelt0)
                else:
                    self.AISMELTRATE[t] = (aisinflection * aismeltlow + aismeltup * (self.temp_atm[t] - 3.0) + aismelt0)

            self.AISCUM[t] = self.AISCUM[t-1] + self.AISMELTRATE[t] / 100

            # What purpose does this code have? Am I missing something?
            # AISREMAIN_0 = np.zeros(self.simulation_horizon)
            # AISREMAIN_0[0] = aiswais + aisother
            #
            # AISREMAIN[1] = AISREMAIN_0[0] - AISCUM[1]

            # Total
            self.TOTALSLR[t] = (self.SLRTHERM[t] + self.GSICCUM[t] + self.GISCUM[t] + self.AISCUM[t])
            return self.TOTALSLR, self.SLRTHERM, self.GSICCUM, self.GISREMAIN, self.GISMELTRATE, self.GISEXPONENT, self.GISCUM, self.AISCUM

        def calculate_SLRDAMAGES(t, gross_output):
            slrmultiplier = 2
            slrelasticity = 4

            slrdamlinear = np.array([0, 0.00452, 0.00053, 0, 0.00011, 0.01172, 0, 0.00138, 0.00351, 0, 0.00616, 0])
            slrdamquadratic = np.array([0.000255, 0, 0.000053, 0.000042, 0, 0.000001, 0.000255, 0, 0, 0.000071, 0, 0.001239])

            self.SLRDAMAGES[:, t] = (100 * slrmultiplier * (
                    self.TOTALSLR[t-1] * slrdamlinear
                    + (self.TOTALSLR[t-1] ** 2) * slrdamquadratic) * (gross_output[:, t-1] / gross_output[:, 0]) ** (
                                        1 / slrelasticity))
            return self.SLRDAMAGES

        # Skip initialisation step
        if t >= 1:
            # Calculate heating of oceans and atmospheric according to matrix equations
            self.temp_atm = calculate_atmospheric_temperature(t, forc)
            self.temp_ocean = calculate_ocean_temp(t)
            # Calculate total SLR
            self.TOTALSLR, self.SLRTHERM, self.GSICCUM, self.GISREMAIN, self.GISMELTRATE, self.GISEXPONENT, self.GISCUM, self.AISCUM =\
                calculate_total_SLR(t, year)
            # Calculate SLR damages
            self.SLRDAMAGES = calculate_SLRDAMAGES(t, gross_output)
        return

    @staticmethod
    def set_up_climate_sensitivity_distributions():
        """
        Setting up three distributions for the climate sensitivity; normal lognormal and cauchy
        @return:
            samples_t2xco2: list with three arrays
        """
        minb = 0
        maxb = 20
        nsamples = 1000

        directory = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(directory, 'input_data', 'ecs_dist_v5.json')) as f:
            d = json.load(f)

        np.random.seed(10)

        samples_norm = np.zeros((0,))

        while samples_norm.shape[0] < nsamples:
            samples = norm.rvs(d["norm"][0], d["norm"][1], nsamples)
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_norm = np.concatenate((samples_norm, accepted), axis=0)
        samples_norm = samples_norm[:nsamples]

        samples_lognorm = np.zeros((0,))

        while samples_lognorm.shape[0] < nsamples:
            samples = lognorm.rvs(
                d["lognorm"][0], d["lognorm"][1], d["lognorm"][2], nsamples
            )
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_lognorm = np.concatenate((samples_lognorm, accepted), axis=0)
        samples_lognorm = samples_lognorm[:nsamples]

        samples_cauchy = np.zeros((0,))

        while samples_cauchy.shape[0] < nsamples:
            samples = cauchy.rvs(d["cauchy"][0], d["cauchy"][1], nsamples)
            accepted = samples[(samples >= minb) & (samples <= maxb)]
            samples_cauchy = np.concatenate((samples_cauchy, accepted), axis=0)
        samples_cauchy = samples_cauchy[:nsamples]

        # extend array with the deterministic value of the nordhaus
        samples_norm = np.append(samples_norm, 3.2)
        samples_lognorm = np.append(samples_lognorm, 3.2)
        samples_cauchy = np.append(samples_cauchy, 3.2)

        samples_t2xco2 = [samples_norm, samples_lognorm, samples_cauchy]

        return samples_t2xco2

import numpy as np
import pandas as pd
import os

package_directory = os.path.dirname(os.path.abspath(__file__))


class EconomicSubmodel:
    def __init__(self, years, regions, uncertainty_dict=None):
        self.years = years
        self.simulation_horizon = len(self.years)
        self.n_regions = len(regions)
        self.delta_t = self.years[1] - self.years[0]  # Assuming equally spaced intervals between the years
        input_path = os.path.join(package_directory)
        self.input_data = pd.read_excel(input_path+"/input_data/RICE_input_data.xlsx", sheet_name="Stijn_RICE_input", usecols=range(0, 36), nrows=12)
        # self.must_change_10yr_only_input_data_tfp_growth = pd.read_excel("RICE_model/input_data/RICE_input_data.xlsx", sheet_name="Stijn_MUST_change_input", usecols=range(1, 32), nrows=12)
        self.must_change_10yr_only_input_data_tfp_growth = pd.read_excel(input_path+"/input_data/RICE_input_data.xlsx",
                                                                         sheet_name="Stijn_MUST_change_input",
                                                                         usecols=range(1, 32), nrows=12)
        self.must_change_10yr_only_input_data_pop_growth = pd.read_excel(input_path+"/input_data/RICE_input_data.xlsx", sheet_name="Stijn_MUST_change_input", usecols=range(34, 65), nrows=12)
        self.income_shares = pd.read_excel(input_path+"/input_data/RICE_input_data.xlsx", sheet_name='Ivar_RICE_income_shares',usecols=range(1, 6)).to_numpy()
        self.uncertainty_dict = uncertainty_dict

        # experimental
        if self.uncertainty_dict:
            self.labour_force = uncertainty_dict['pop']
            # self.gross_output = uncertainty_dict['GDP']
            self.total_factor_productivity = uncertainty_dict['tfp']
            self.sigma_gr_mapping = uncertainty_dict['sigma_gr_mapping']
            self.fosslim = uncertainty_dict['fosslim']
            self.cback = uncertainty_dict['price_backstop_tech']
            self.elasticity_of_damages = self.uncertainty_dict['elasticity_climate_impact']
            self.limmu = self.uncertainty_dict['negative_emissions_possible']
        # Uncertainties: SSP scenarios
        # if SSP_projections:
        #     self.SSP_dict = SSP_projections.SSP_dict
        # self.SSP_scenario = SSP_scenario
        # if self.SSP_scenario:
        #     self.labour_force = self.SSP_dict[f'SSP_{self.SSP_scenario}_pop']
        #     self.gross_output = self.SSP_dict[f'SSP_{self.SSP_scenario}_GDP']
        else:
            # Gross economy variables ---------------------------------------------------------------------------
            # Create dynamic variables matrices -----------------------------------------------------------------
            # self.gross_output = np.zeros((self.n_regions, self.simulation_horizon))
            self.labour_force = np.zeros((self.n_regions, self.simulation_horizon))

            # Place initial data (from 2005) -------------------------------------------------------------
            # self.gross_output[:, 0] = self.input_data['Output_2005 (US International $, trillions)']
            self.labour_force[:, 0] = self.input_data['Population_2005']

            # The growth series (tfp and population) are now taken from an excel which only supplies them at 10 year intervals
            # so if we want to change the timestep we MUST estimate them ourselves!
            self.total_factor_productivity_growth = np.array(self.must_change_10yr_only_input_data_tfp_growth)
            self.population_growth = np.array(self.must_change_10yr_only_input_data_pop_growth)

            self.sigma_gr_mapping = 1
            self.fosslim = 6000

            self.total_factor_productivity = np.zeros((self.n_regions, self.simulation_horizon))
            self.total_factor_productivity[:, 0] = self.input_data['Total_factor_productivity_2005']

            self.cback = 1.260  # This is for the SSP low scenario, it is 1.260 * 1.5 for the SSP high scenario
            self.elasticity_of_damages = 0  # This is again a scenario option: options are -1, 0, 1
            self.limmu = 1

        self.gross_output = np.zeros((self.n_regions, self.simulation_horizon))
        self.capital_stock = np.zeros((self.n_regions, self.simulation_horizon))

        self.mu = np.zeros((self.n_regions, self.simulation_horizon))
        self.S = np.zeros((self.n_regions, self.simulation_horizon))

        self.sigma_ratio = np.zeros((self.n_regions, self.simulation_horizon))
        self.Eind = np.zeros((self.n_regions, self.simulation_horizon))
        self.Eind_cum = np.zeros((self.n_regions, self.simulation_horizon))
        self.Etree = np.zeros((self.n_regions, self.simulation_horizon))
        self.Etree_cum = np.zeros((self.n_regions, self.simulation_horizon))
        self.E = np.zeros((self.n_regions, self.simulation_horizon))
        self.E_cum = np.zeros((self.n_regions, self.simulation_horizon))

        # Parameter definitions --------------------------------------------------------------------------------
        self.dk = 0.1
        # output_elasticity_of_capital
        self.gamma = 0.3

        # Place initial data (from 2005) -------------------------------------------------------------
        # self.total_factor_productivity[:, 0] = self.input_data['Total_factor_productivity_2005']
        self.gross_output[:, 0] = self.input_data['Output_2005 (US International $, trillions)']
        self.capital_stock[:, 0] = self.input_data['Capital_stock_2005']

        # # Initial atmospheric temperature values
        # # Initial atmospheric temperature change [dC from 1900]
        # self.temp_atm[0:2] = [0.83, 0.980]

        # set savings rate and control rate as optimized RICE 2010 for the first two timesteps
        self.mu[:, 0] = self.input_data['emission control rate [miu] 2005']
        self.S[:, 0] = self.input_data['savings rate [sr] 2005']
        self.mu[:, 1] = self.input_data['emission control rate [miu] 2015']
        self.S[:, 1] = self.input_data['savings rate [sr] 2015']

        # Emissions
        # Emissions from land change use
        self.Etree[:, 0] = self.input_data['Initial carbon emissions from land use change (GTC per year)']
        self.Etree_cum[:, 0] = self.input_data['Initial carbon emissions from land use change (GTC per year)']
        self.decline_rate_Etree = self.input_data['Decline rate of land emissions per decade']
        # # industrial emissions 2005
        self.sigma_ratio[:, 0] = self.input_data['Initial sigma (tC per $1000 GDP US $, 2005 prices) 2005']
        # self.Eind[:, 0] = self.sigma_ratio[:, 0] * self.gross_output[:, 0] * (1 - self.mu[:, 0])
        # self.Eind_cum[:, 0] = self.Eind[:, 0]
        # # initialize initial emissions
        # self.E[:, 0] = self.Eind[:, 0] + self.Etree[:, 0]
        # self.E_cum[:, 0] = self.Eind_cum[:, 0] + self.Etree_cum[:, 0]

        # Net economy variables ----------------------------------------------------------------------------
        # Create dynamic variables matrices -----------------------------------------------------------------
        self.damage_fraction = np.zeros((self.n_regions, self.simulation_horizon))
        # self.temp_atm = np.zeros(self.simulation_horizon)
        self.abatement_fraction = np.zeros((self.n_regions, self.simulation_horizon))
        self.beta1 = np.zeros((self.n_regions, self.simulation_horizon))
        self.paybacktime = np.zeros((self.n_regions, self.simulation_horizon))
        # self.sigma_ratio = np.zeros((self.n_regions, self.simulation_horizon))
        self.sigma_gr = np.zeros((self.n_regions, self.simulation_horizon))

        self.damages = np.zeros((self.n_regions, self.simulation_horizon))
        self.abatement_cost = np.zeros((self.n_regions, self.simulation_horizon))
        self.net_output = np.zeros((self.n_regions, self.simulation_horizon))
        self.marginal_abatement_cost = np.zeros((self.n_regions, self.simulation_horizon))
        self.carbon_price = np.zeros((self.n_regions, self.simulation_horizon))
        self.I = np.zeros((self.n_regions, self.simulation_horizon))
        self.consumption = np.zeros((self.n_regions, self.simulation_horizon))
        self.CPC = np.zeros((self.n_regions, self.simulation_horizon))

        # Parameter definitions -----------------------------------------------------------------------------
        self.beta2 = 2.8
        self.backstop_competitive_year = 2250
        # set uncertainties that drive MIU
        # self.limmu = self.uncertainty_dict['negative_emissions_possible']  # 1
        # self.irstp = 0.015  # Initial rate of social time preference  (RICE2010 OPT))

        # Place initial data (from 2005) --------------------------------------------------------------------
        # damage parameters (on atmospheric temperature)
        self.lambda1 = self.input_data['damage coefficient on temperature_MAX']
        self.lambda2 = self.input_data['damage coefficient on temperature squared_MAX']
        self.lambda3 = self.input_data['Exponent on damages_MAX']
        # abatement cost parameters (on emission control rate)
        self.cback_region = self.cback * self.input_data['Ratio of  backstop to World']
        self.decline_of_backstop_price_decade = self.input_data['Decline of backstop price (per decade)']
        self.paybacktime[:, 0] = self.cback_region

        self.total_growth_rate_sigma = self.input_data['Total growth rate sigma']  # sigma_growth_data[:, 2]
        self.decline_rate_sigma_growth = self.input_data['Decline rate sigma growth']  # sigma_growth_data[:, 3]
        self.trend_sigma_growth = self.input_data['Trend sigma growth']  # sigma_growth_data[:, 4]
        self.emission_factor = self.input_data['emission factor 2015']
        self.sigma_gr[:, 0] = self.input_data['sigma growth average 95-06']
        # self.sigma_gr[:, 1] = self.total_growth_rate_sigma  # (self.sigma_growth_data[:,4] + (self.sigma_growth_data[:,2] - self.sigma_growth_data[:,4] ))
        self.sigma_gr[:, 1] = self.trend_sigma_growth + (self.total_growth_rate_sigma - self.trend_sigma_growth)
        # self.sigma_ratio[:, 0] = self.input_data['Initial sigma (tC per $1000 GDP US $, 2005 prices) 2005']
        self.sigma_ratio[:, 1] = self.sigma_ratio[:, 0] * (
                2.71828 ** (self.sigma_gr[:, 1] * self.delta_t)) * self.emission_factor

        self.beta1[:, 0] = (self.paybacktime[:, 0] * self.sigma_ratio[:, 0]) / self.beta2

        # Miscellaneous ---------------------------------------------------------------
        # Disaggregated consumption tallys
        self.CPC_post_damage = {}
        self.CPC_pre_damage = {}
        self.climate_impact_relative_to_capita = {}

        self.pre_damage_total__region_consumption = np.zeros((self.n_regions, self.simulation_horizon))

        # damage share elasticity function derived from Denig et al 2015
        self.damage_share = self.income_shares ** self.elasticity_of_damages
        self.sum_damage = np.sum(self.damage_share, axis=1)

    def run_gross(self, t, year, mu_target, sr):
        # Gross output functions ----------------------------------------
        def calculate_total_factor_productivity(t):
            if self.uncertainty_dict:
                # calculate tfp based on GDP projections by SSP's
                self.total_factor_productivity[:, t] = self.gross_output[:, t] / ((self.capital_stock[:, t]**self.gamma)*(self.labour_force[:, t]/1000)**(1-self.gamma))
            else:
                # Below based on bad (bad because then you cant change the delta t) but verified 10yr only data tfp growth data
                self.total_factor_productivity[:, t] = self.total_factor_productivity[:, t-1] * 2.71828**(self.total_factor_productivity_growth[:, t]*self.delta_t)
            return self.total_factor_productivity

        def calculate_labour_force(t):
            self.labour_force[:, t] = self.labour_force[:, t-1] * 2.71828 ** (self.population_growth[:, t]*self.delta_t)
            return self.labour_force

        def calculate_capital_stock(t):
            # Deviating from code Ivar, going from DICE manual instead: "capital stock dynamics
            # follows a perpetual inventory method with an exponential depreciation rate"
            # self.capital_stock[:, t] = (self.I[:, t] * self.delta_t) - ((self.dk**self.delta_t)*self.capital_stock[:, t-1])
            self.capital_stock[:, t] = (self.capital_stock[:, t-1] * ((1-self.dk)**self.delta_t) + (self.I[:, t-1] * self.delta_t))
            # lower bound capital stock
            self.capital_stock[:, t] = np.where(self.capital_stock[:, t] > 1, self.capital_stock[:, t], 1)
            return self.capital_stock

        def gross_production_function(t):
            self.gross_output[:, t] = self.total_factor_productivity[:, t] * ((self.labour_force[:, t]/1000)**(1-self.gamma)) * (self.capital_stock[:, t]**self.gamma)
            return self.gross_output

        def calculate_S(t, sr):
            # Must do away with t>12 structure, no idea why it is there in the first place
            if t > 1:
                self.S[:, t] = (sr - self.S[:, 1]) * t / 12 + self.S[:, 1]
            if t > 12:
                self.S[:, t] = sr

            # self.S[:, t] = sr
            return self.S

        def calculate_mu(t, mu_target):
            self.mu_period = (mu_target - self.years[0]) / self.delta_t  # self.input_data['miu period']
            # The first two values (for 2005 and 2015) are already filled in, given by the optimized GAMS run of DICE2010R
            if t > 1:
                # Because every region has a different mu_period, iterate over the periods
                for region_idx in range(0, self.n_regions):
                    self.mu[region_idx, t] = min((self.mu[region_idx, t-1] + (self.limmu - self.mu[region_idx, 1])/self.mu_period), self.limmu)
            return self.mu

        # Emissions functions --------------------------------------------------

        def calculate_paybacktime(t, year):
            if year > self.backstop_competitive_year:
                # The line underneath is very odd because in the excel file it says it should be zero
                self.paybacktime[:, t] = self.paybacktime[:, t] = self.paybacktime[:, t - 1] * 0.5
            else:
                self.paybacktime[:, t] = 0.10 * self.cback_region + (
                            self.paybacktime[:, t - 1] - 0.1 * self.cback_region) * (
                                                     1 - self.decline_of_backstop_price_decade)
            return self.paybacktime

        def calculate_sigma_ratio(t):
            if t > 1:
                self.sigma_gr[:, t] = self.trend_sigma_growth + (self.sigma_gr[:, t - 1] - self.trend_sigma_growth) * (
                            1 - self.decline_rate_sigma_growth)
                self.sigma_gr[:, t] = self.sigma_gr[:, t] * self.sigma_gr_mapping
                self.sigma_ratio[:, t] = self.sigma_ratio[:, t - 1] * (2.71828 ** (self.sigma_gr[:, t] * self.delta_t))
            return self.sigma_gr, self.sigma_ratio

        def calculate_yearly_emissions(t):
            # Yearly emissions from industry
            self.Eind[:, t] = self.sigma_ratio[:, t]*(1-self.mu[:, t])*self.gross_output[:, t]
            # Yearly emissions from land use
            if t >= 1:
                self.Etree[:, t] = (1-self.decline_rate_Etree)*self.Etree[:, t-1]
            self.E[:, t] = self.Eind[:, t] + self.Etree[:, t]
            return self.Eind, self.Etree, self.E

        def calculate_cumulative_emissions(t):
            self.Eind_cum[:, t] = self.Eind[:, t-1] + (self.Eind[:, t] * self.delta_t)
            self.Eind_cum[:, t] = np.where(self.Eind_cum[:, t] < self.fosslim, self.Eind_cum[:, t], self.fosslim)
            if t >= 1:
                self.Etree_cum[:, t] = self.Etree[:, t-1] + (self.Etree[:, t] * self.delta_t)
            self.E_cum = self.Eind_cum + self.Etree_cum
            return self.Eind_cum, self.Etree_cum, self.E_cum

        # Variables for gross_output --------------------
        if self.uncertainty_dict:
            self.capital_stock = calculate_capital_stock(t)
            self.gross_output = gross_production_function(t)
            # self.total_factor_productivity = calculate_total_factor_productivity(t)
        else:
            # skip the initial timestep because these variables are calibrated according to the DICE2010R excel file
            if t >= 1:
                self.total_factor_productivity = calculate_total_factor_productivity(t)
                self.labour_force = calculate_labour_force(t)
                self.capital_stock = calculate_capital_stock(t)
                self.gross_output = gross_production_function(t)

        # Levers ----------------------------------------
        self.S = calculate_S(t, sr)
        self.mu = calculate_mu(t, mu_target)
        # Variables for emissions ---------------------
        self.paybacktime = calculate_paybacktime(t, year)
        self.sigma_gr, self.sigma_ratio = calculate_sigma_ratio(t)
        self.Eind, self.Etree, self.E = calculate_yearly_emissions(t)
        self.Eind_cum, self.Etree_cum, self.E_cum = calculate_cumulative_emissions(t)
        return

    def run_net(self, t, year, temp_atm, SLRDAMAGES):
        # Damage function ---------------------------------------------------

        def damage_function(t, temp_atm):
            self.damage_fraction[:, t] = (self.lambda1 * temp_atm[t] + self.lambda2 * (temp_atm[t] ** self.lambda3)) * 0.01
            return self.damage_fraction

        def calculate_damages(t, SLRDAMAGES):
            if t >= 1:
                self.damages[:, t] = self.gross_output[:, t] * (self.damage_fraction[:, t] + (SLRDAMAGES[:, t]/100))
            else:
                self.damages[:, 0] = self.gross_output[:, 0] - (self.gross_output[:, 0]/(1.0 + self.damage_fraction[:, 0]))
            return self.damages

        # Abatement function ------------------------------------------------

        def calculate_beta1(t):
            self.beta1[:, t] = (self.paybacktime[:, t] * self.sigma_ratio[:, t]) / self.beta2
            return self.beta1

        def abatement_function(t):
            self.abatement_fraction[:, t] = self.beta1[:, t] * self.mu[:, t]**self.beta2
            return self.abatement_fraction

        def calculate_abatement_cost(t):
            self.abatement_cost[:, t] = self.gross_output[:, t] * self.abatement_fraction[:, t]
            return self.abatement_cost

        # net output functions -------------------------------------------------

        def calculate_net_output(t):
            self.net_output[:, t] = self.gross_output[:, t] - self.damages[:, t] - self.abatement_cost[:, t]
            # Lower limit for net output
            self.net_output[:, t] = np.where(self.net_output[:, t] > 0, self.net_output[:, t], 0)
            return self.net_output

        def calculate_carbon_price(t):
            self.marginal_abatement_cost[:, t] = self.paybacktime[:, t] * (self.mu[:, t]**(self.beta2-1))
            self.carbon_price[:, t] = 1000 * self.marginal_abatement_cost[:, t]
            return self.marginal_abatement_cost, self.carbon_price

        # consumption functions ------------------------------------------------

        def calculate_investment(t):
            self.I[:, t] = self.S[:, t] * self.net_output[:, t]
            # check lower bound investments
            self.I[:, t] = np.where(self.I[:, t] > 0, self.I[:, t], 0)
            return self.I

        def calculate_consumption(t):
            self.consumption[:, t] = self.net_output[:, t] - self.I[:, t]
            # check for lower bound on C
            c_lo = 2
            self.consumption[:, t] = np.where(self.consumption[:, t] > c_lo, self.consumption[:, t], c_lo)
            return self.consumption

        def calculate_average_consumption_per_capita(t):
            # average consumption per capita per region
            self.CPC[:, t] = (1000 * self.consumption[:, t]) / self.labour_force[:, t]
            # check for lower bound on consumption per capita
            CPC_lo = 0.01
            self.CPC[:, t] = np.where(self.CPC[:, t] > CPC_lo, self.CPC[:, t], CPC_lo)
            return self.CPC

        # Miscellaneous --------------------------------------------------------

        def calculate_climate_impact_relative_to_capita(t, year):
            # This function is basically copied from Ivar
            CPC_lo = 0.01

            # calculate pre damage consumption aggregated per region
            self.pre_damage_total__region_consumption[:, t] = self.consumption[:, t] + self.damages[:, t]

            for i in range(0, self.n_regions):
                self.damage_share[i, :] = self.damage_share[i, :] / self.sum_damage[i]

            # calculate disaggregated per capita consumption based on income shares BEFORE damages
            self.CPC_pre_damage[year] = ((self.pre_damage_total__region_consumption[:,
                                               t] * self.income_shares.transpose()) / (
                                                          self.labour_force[:, t] * (1 / 5))) * 1000

            # calculate disaggregated per capita consumption based on income shares AFTER damages
            self.CPC_post_damage[year] = self.CPC_pre_damage[year] - (((self.damages[:,
                                                                                  t] * self.damage_share.transpose()) / (
                                                                                             self.labour_force[:, t] * (
                                                                                                 1 / 5))) * 1000)

            # check for lower bound on C
            self.CPC_pre_damage[year] = np.where(self.CPC_pre_damage[year] > CPC_lo,
                                                      self.CPC_pre_damage[year], CPC_lo)
            self.CPC_post_damage[year] = np.where(self.CPC_post_damage[year] > CPC_lo,
                                                       self.CPC_post_damage[year], CPC_lo)

            # calculate damage per quintile equiv
            self.climate_impact_relative_to_capita[year] = ((self.damages[:,
                                                                  t] * self.damage_share.transpose() * self.delta_t ** 12) / (
                                                                             0.2 * self.labour_force[:, t] * self.delta_t ** 6)) / (
                                                                            self.CPC_pre_damage[year] * 1000)

            self.climate_impact_relative_to_capita[year] = np.where(
                self.climate_impact_relative_to_capita[year] > 1, 1,
                self.climate_impact_relative_to_capita[year])
            return self.climate_impact_relative_to_capita

        # Variables for damage_fraction -----------------
        self.damage_fraction = damage_function(t, temp_atm)
        self.damages = calculate_damages(t, SLRDAMAGES)
        # Variables for abatement fraction --------------
        self.beta1 = calculate_beta1(t)
        self.abatement_fraction = abatement_function(t)
        self.abatement_cost = calculate_abatement_cost(t)
        # Variables for net_output ---------------------
        self.net_output = calculate_net_output(t)
        self.marginal_abatement_cost, self.carbon_price = calculate_carbon_price(t)
        # Variables for consumption --------------------
        self.I = calculate_investment(t)
        self.consumption = calculate_consumption(t)
        self.CPC = calculate_average_consumption_per_capita(t)
        # Miscellaneous
        self.climate_impact_relative_to_capita = calculate_climate_impact_relative_to_capita(t, year)
        return

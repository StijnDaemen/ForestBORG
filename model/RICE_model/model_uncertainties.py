'''
From thesis Max:

levers:
saving rate -> range: [0.1, 0.5]
emission control rate -> range: [2065, 2305]
initial rate of social time preference consumption -> range: [0.001, 0.015]

uncertainties:
population growth rate -> range: {short-term, long-term}
total factor productivity growth rate -> range: {short-term, long-term}
availability of fossil fuel -> range: [4000, 13649]
emission to output growth rate -> range: {short-term, long-term}
elasticity of disutility of damage -> range: [0.001, 0.6]
climate sensitivity parameter -> range: {normal, lognormal, Cauchy}
elasticity of climate impact -> range: {-1, 0, 1}
price of backstop technology -> range: [1260, 1890]
availability of negative emissions technology -> range: {yes(20%), no}

'''
import pandas as pd
import os


package_directory = os.path.dirname(os.path.abspath(__file__))


class Uncertainties:
    def __init__(self, years, regions):
        self.years = years
        self.simulation_horizon = len(self.years)
        self.n_regions = len(regions)

        input_file = os.path.join(package_directory, 'input_data', 'SSP_projections.xlsx')
        SSP_pop_projections = pd.read_excel(input_file, sheet_name='pop_SSPs')
        SSP_GDP_projections = pd.read_excel(input_file, sheet_name='GDP_SSPs')
        SSP_tfp_projections = pd.read_excel(input_file, sheet_name='tfp_SSPs')
        # SSP_pop_projections = pd.read_excel("RICE_model/input_data/SSP_projections.xlsx", sheet_name='pop_SSPs')
        # SSP_GDP_projections = pd.read_excel("RICE_model/input_data/SSP_projections.xlsx", sheet_name='GDP_SSPs')
        # SSP_tfp_projections = pd.read_excel("RICE_model/input_data/SSP_projections.xlsx", sheet_name='tfp_SSPs')

        self.SSP_dict = {}
        for i in range(1, 6):
            self.SSP_dict[f'SSP_{i}_pop'] = SSP_pop_projections.filter(regex=f'{i}').T.to_numpy()
            self.SSP_dict[f'SSP_{i}_GDP'] = SSP_GDP_projections.filter(regex=f'{i}').T.to_numpy()
            self.SSP_dict[f'SSP_{i}_tfp'] = SSP_tfp_projections.filter(regex=f'{i}').T.to_numpy()
        # In model Max there were only 3 sigma_gr options (0.5, 1, 1.5) but I extended it to a range of 5 -> NEED TO CHECK IN LITERATURE
        self.SSP_dict['SSP_1_sigma_gr_mapping'] = 0.5
        self.SSP_dict['SSP_2_sigma_gr_mapping'] = 0.75
        self.SSP_dict['SSP_3_sigma_gr_mapping'] = 1
        self.SSP_dict['SSP_4_sigma_gr_mapping'] = 1.25
        self.SSP_dict['SSP_5_sigma_gr_mapping'] = 1.5
        self.SSP_dict['SSP_1_fosslim_limits'] = [11720, 13650] # In the subsequent code the last number is not read so mathematically it is: [4000, 5930)
        self.SSP_dict['SSP_2_fosslim_limits'] = [9790, 11720]
        self.SSP_dict['SSP_3_fosslim_limits'] = [7860, 9790]
        self.SSP_dict['SSP_4_fosslim_limits'] = [5930, 7860]
        self.SSP_dict['SSP_5_fosslim_limits'] = [4000, 5930]

        self.climate_dict = {'elasticity_climate_impact': [-1, 0, 1],
                             't2xco2_index': [0, 999]}
        self.climate_sensitivity_distribution_map = {'log': 0,
                                                    'lognormal': 1,
                                                    'Cauchy': 2}

        self.backstop_dict = {'price_backstop_tech_limits': [1.260, 1.470, 1.680, 1.890]}  # [1260, 1890]
        self.backstop_tech_possible_map = {'yes': 1.2,
                                           'no': 1}

    def create_uncertainty_dict(self, scenario):
        # # The scenario input must always be structured as follows: a list at zero index the SSP scenario, at first index the availability of fossil fuel. E.g. [2, 10000]
        # # uncertainty_dict = {key: value for key, value in self.SSP_dict.items() if scenario[0] in key.lower()}
        # uncertainty_dict = {'pop': self.SSP_dict[f'SSP_{scenario[0]}_pop'],
        #                     'GDP': self.SSP_dict[f'SSP_{scenario[0]}_GDP'],
        #                     'sigma_gr_mapping': self.SSP_dict[f'SSP_{scenario[0]}_sigma_gr_mapping']}
        # if (scenario[1] < self.SSP_dict[f'SSP_{scenario[0]}_fosslim_limits'][0]) | (scenario[1] >= self.SSP_dict[f'SSP_{scenario[0]}_fosslim_limits'][1]):
        #     # print(scenario[1])
        #     # print(self.SSP_dict[f'SSP_{scenario[0]}_fosslim_limits'][0])
        #     # print(self.SSP_dict[f'SSP_{scenario[0]}_fosslim_limits'][1])
        #     print('Availability of fossil fuel does not match the SSP scenario. Please change it to a value wihtin the limits.')
        # else:
        #     uncertainty_dict['fosslim'] = scenario[1]

        ## UPDATE WITH SCENARIO IN DICT FORM INSTEAD OF IN LIST FORM ##
        # example scenario:
        # scenario = {'SSP_scenario': 5,                                        # 1, 2, 3, 4, 5
        #             'fosslim': 4200,                                          # range(4000, 13650), depending on SSP scenario
        #             'climate_sensitivity_distribution': 'lognormal',          # 'log', 'lognormal', 'Cauchy'
        #             'elasticity_climate_impact': -1,                          # -1, 0, 1
        #             'price_backstop_tech': 1260,                              # range(1.260, 1.890) [1.260, 1.470, 1.680, 1.890]
        #             'negative_emissions_possible': 'yes',                     # 'yes' or 'no'
        #             't2xco2_index': 500}                                      # [0, 999]

        # uncertainty_dict = {key: value for key, value in self.SSP_dict.items() if scenario[0] in key.lower()}
        uncertainty_dict = {}
        if ('SSP_scenario' in list(scenario.keys())) & ('fosslim' in list(scenario.keys())):
            uncertainty_dict = {'pop': self.SSP_dict[f'SSP_{scenario["SSP_scenario"]}_pop'],
                                # 'GDP': self.SSP_dict[f'SSP_{scenario["SSP_scenario"]}_GDP'],
                                'tfp': self.SSP_dict[f'SSP_{scenario["SSP_scenario"]}_tfp'],
                                'sigma_gr_mapping': self.SSP_dict[f'SSP_{scenario["SSP_scenario"]}_sigma_gr_mapping']}
            # if (scenario["fosslim"] < self.SSP_dict[f'SSP_{scenario["SSP_scenario"]}_fosslim_limits'][0]) | (
            #         scenario["fosslim"] >= self.SSP_dict[f'SSP_{scenario["SSP_scenario"]}_fosslim_limits'][1]):
            #     print(
            #         'Availability of fossil fuel does not match the SSP scenario. Please change it to a value wihtin the limits.')
            # else:
            #     uncertainty_dict['fosslim'] = scenario["fosslim"]
            uncertainty_dict['fosslim'] = scenario["fosslim"]
        else:
            print('No SSP scenario and/or availability of fossil fuels was given in the scenario.')

        if ('climate_sensitivity_distribution' in list(scenario.keys())) & ('elasticity_climate_impact' in list(scenario.keys())):

            uncertainty_dict['climate_sensitivity_distribution'] = self.climate_sensitivity_distribution_map[scenario['climate_sensitivity_distribution']]
            uncertainty_dict['t2xco2_index'] = scenario['t2xco2_index']
            uncertainty_dict['elasticity_climate_impact'] = scenario['elasticity_climate_impact']
        else:
            print('No climate sensitivity distribution and/or elasticity of climate impact was given in the scenario.')

        if ('price_backstop_tech' in list(scenario.keys())) & ('negative_emissions_possible' in list(scenario.keys())):
            # if (scenario['price_backstop_tech'] < self.backstop_dict['price_backstop_tech_limits'][0]) | (scenario['price_backstop_tech'] > self.backstop_dict['price_backstop_tech_limits'][1]):
            # if scenario['price_backstop_tech'] not in self.backstop_dict['price_backstop_tech_limits']:
            #     print('Price of backstop technology given in the scenario does not fall within the limits.')
            # else:
            #     uncertainty_dict['price_backstop_tech'] = scenario['price_backstop_tech']
            uncertainty_dict['price_backstop_tech'] = scenario['price_backstop_tech']
            uncertainty_dict['negative_emissions_possible'] = self.backstop_tech_possible_map[scenario['negative_emissions_possible']]
        else:
            print('No price_backstop_tech and/or negative_emissions_possible was given in the scenario.')
        return uncertainty_dict



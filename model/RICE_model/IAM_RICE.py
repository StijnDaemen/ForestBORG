import pandas as pd
import numpy as np
import sqlite3

from model.RICE_model import *


class RICE:
    def __init__(self, years, regions, scenario=None, levers=None, save_location=None, file_name=None):
        self.years = years
        self.regions = regions
        self.rng = np.random.default_rng(seed=42)

        self.save_location = save_location
        self.file_name = file_name
        # Initialize sqlite database to save outcomes of a run - note use one universal database called 'Experiments.db'
        if save_location:
            self.database = f'{save_location}/Experiments.db'
        else:
            self.database = None

        # Initialize scenario dict, if provided
        if scenario:
            self.scenario = scenario
            self.uncertainty_dict = model_uncertainties.Uncertainties(self.years, self.regions).create_uncertainty_dict(
                scenario=scenario)
        else:
            # Basic RICE scenario in case no scenario is provided
            self.uncertainty_dict = None
            self.scenario = None

        # Initialize levers dict, if provided
        if levers:
            self.levers = levers
        else:
            # Nordhaus policy in case no policy is provided
            self.levers = {'mu_target': 2135,
                           'sr': 0.248,
                           'irstp': 0.015}

        # Initialize submodels
        self.economic_submodel = RICE_economic_submodel.EconomicSubmodel(self.years, self.regions,
                                                                         uncertainty_dict=self.uncertainty_dict)
        self.carbon_submodel = RICE_carboncycle_submodel.CarbonSubmodel(self.years)
        self.climate_submodel = RICE_climate_submodel.ClimateSubmodel(self.years, self.regions, uncertainty_dict=self.uncertainty_dict)
        self.welfare_submodel = welfare_submodel.WelfareSubmodel(self.years, self.regions)

    def run(self, write_to_excel=False, write_to_sqlite=False):
        t = 0
        for year in self.years:
            self.economic_submodel.run_gross(t, year, mu_target=self.levers['mu_target'], sr=self.levers['sr'])
            self.carbon_submodel.run(t, year, E=self.economic_submodel.E)
            self.climate_submodel.run(t, year, forc=self.carbon_submodel.forc,
                                      gross_output=self.economic_submodel.gross_output)
            self.economic_submodel.run_net(t, year, temp_atm=self.climate_submodel.temp_atm,
                                           SLRDAMAGES=self.climate_submodel.SLRDAMAGES)
            self.welfare_submodel.run_utilitarian(t, year, CPC=self.economic_submodel.CPC,
                                                  labour_force=self.economic_submodel.labour_force,
                                                  damages=self.economic_submodel.damages,
                                                  net_output=self.economic_submodel.net_output,
                                                  temp_atm=self.climate_submodel.temp_atm, irstp=self.levers['irstp'])
            t += 1
        if write_to_excel:
            if not self.file_name:
                print('You have to provide a save_location and file_name to the init of RICE() if you want to save simulation results to excel.')
                return None
            else:
                self.write_to_excel(collection='executive variables')  # f'scenario 2_10000--levers 2135_0248_0015'
            # self.write_to_excel(collection='executive variables',
            #                     file_name='record_executive_variables_for_verification_w_pyRICE2022_5')
        if write_to_sqlite:
            if not self.file_name:
                print('You have to provide a save_location and file_name to the init of RICE() if you want to save simulation results to an sqlite database.')
                return None
            else:
                self.write_to_sqlite(collection='executive variables')  # 'test3' # all_scenarios_1_policy_run_1
        # print(self.welfare_submodel.global_period_util_ww)
        # print(self.get_metrics())
        return None

    def write_to_excel(self, collection='all variables'):
        def collect_all_variables():
            # Collect all class variables from the different submodels
            model_variables = vars(self.economic_submodel)
            model_variables.update(vars(self.carbon_submodel))
            model_variables.update(vars(self.climate_submodel))
            model_variables.update(vars(self.welfare_submodel))
            model_variables_names = list(model_variables.keys())

            # Create a dictionary with all of the region variables (so the matrix variables/ ndarray) expanded into unique keys holding an array
            model_variables_region = []
            for name in model_variables_names:
                if type(model_variables[name]) == np.ndarray:
                    if model_variables[name].shape == (len(self.regions), len(self.years)):
                        model_variables_region.append(name)

            sub_dict = {}
            for name in model_variables_region:
                for index, key in enumerate(self.regions):
                    sub_dict[f'{name}_{key}'] = model_variables[name][index]

            model_variables_general = []
            for name in model_variables_names:
                if type(model_variables[name]) == np.ndarray:
                    if model_variables[name].shape == (len(self.years),):
                        model_variables_general.append(name)

            model_variables_dynamic = {}
            for key in self.regions:
                model_variables_dynamic[key] = {}
                for name in model_variables_region:
                    model_variables_dynamic[key][name] = sub_dict[f"{name}_{key}"]
                for name in model_variables_general:
                    model_variables_dynamic[key][name] = model_variables[name]
            return model_variables_dynamic

        def collect_executive_variables():
            # executive_variables_dict = {'mu': self.economic_submodel.mu,
            #                             'S': self.economic_submodel.S,
            #                             'E': self.economic_submodel.E,
            #                             'damages': self.economic_submodel.damages,
            #                             'abatement_cost': self.economic_submodel.abatement_cost,
            #                             'abatement_fraction': self.economic_submodel.abatement_fraction,
            #                             'SLRDAMAGES': self.climate_submodel.SLRDAMAGES,
            #                             'gross_output': self.economic_submodel.gross_output,
            #                             'net_output': self.economic_submodel.net_output,
            #                             'I': self.economic_submodel.I,
            #                             'CPC': self.economic_submodel.CPC,
            #                             'forc': self.carbon_submodel.forc,
            #                             'temp_atm': self.climate_submodel.temp_atm,
            #                             'temp_ocean': self.climate_submodel.temp_ocean,
            #                             'global_damage': self.welfare_submodel.global_damage,
            #                             'global_output': self.welfare_submodel.global_output,
            #                             'global_per_util_ww': self.welfare_submodel.global_per_util_ww,
            #                             'TOTAL_SLR': self.climate_submodel.TOTALSLR,
            #                             'mat': self.carbon_submodel.mat,
            #                             'mup': self.carbon_submodel.mup,
            #                             'ml': self.carbon_submodel.ml,
            #                             'forcoth': self.carbon_submodel.forcoth,
            #                             'E_worldwide_per_year': self.carbon_submodel.E_worldwide_per_year,
            #                             'labour_force': self.economic_submodel.labour_force,
            #                             'total_factor_productivity': self.economic_submodel.total_factor_productivity,
            #                             'capital_stock': self.economic_submodel.capital_stock,
            #                             'sigma_ratio': self.economic_submodel.sigma_ratio,
            #                             'Eind': self.economic_submodel.Eind,
            #                             'sigma_gr': self.economic_submodel.sigma_gr,
            #                             'damage_frac': self.economic_submodel.damage_fraction,
            #                             'SLRTHERM': self.climate_submodel.SLRTHERM,
            #                             'GSICCUM': self.climate_submodel.GSICCUM,
            #                             'GISCUM': self.climate_submodel.GISCUM,
            #                             'AISCUM': self.climate_submodel.AISCUM,
            #                             }
            executive_variables_dict = {'mu': self.economic_submodel.mu,
                                        'S': self.economic_submodel.S,
                                        # 'irstp': self.levers['irstp'],
                                        'net_output': self.economic_submodel.net_output,
                                        'mat': self.carbon_submodel.mat,
                                        'global_utility': self.welfare_submodel.global_per_util_ww,
                                        'global_damage': self.welfare_submodel.global_damage,
                                        'temp_overshoots': self.welfare_submodel.temp_overshoots,
                                        'temp_atm': self.climate_submodel.temp_atm,
                                        }

            exec_var_dict = {}
            for idx, region in enumerate(self.regions):
                exec_var_dict[region] = {}
                for key, item in executive_variables_dict.items():
                    if item.shape == (len(self.regions), len(self.years)):
                        exec_var_dict[region][key] = item[idx]
                    else:
                        exec_var_dict[region][key] = item[:]
            return exec_var_dict

        def collect_metrics_variables():
            metrics_dict = {}
            region = 'global'
            metrics_dict[region] = {}
            metrics = self.get_metrics()
            for idx, metric in enumerate(metrics):
                metrics_dict[region][f'metric_{idx}'] = metric

            # region = 'global'
            # metrics_dict = {region: {'global_period_util_ww': self.welfare_submodel.global_period_util_ww,
            #                          'global_output': self.welfare_submodel.global_output,
            #                          'temp_overshoots': self.welfare_submodel.temp_overshoots}}
            return metrics_dict

        model_variables_to_excel = {}
        if collection == 'executive variables':
            model_variables_to_excel = collect_executive_variables()
        elif collection == 'all variables':
            model_variables_to_excel = collect_all_variables()
        elif collection == 'metrics':
            model_variables_to_excel = collect_metrics_variables()

        # Write dictionaries to an excel file
        writer = pd.ExcelWriter(f'{self.save_location}/{self.file_name}.xlsx')
        for region_key in model_variables_to_excel:
            df = pd.DataFrame.from_dict(model_variables_to_excel[region_key])
            df.index = self.years
            df.to_excel(writer, sheet_name=region_key)
        input_df = pd.DataFrame.from_dict([self.levers])
        if self.scenario:
            input_df['SSP_scenario'] = self.scenario['SSP_scenario']
            input_df['Availability of fossil fuels'] = self.scenario['fosslim']
            input_df['climate_sensitivity_distribution'] = self.scenario['climate_sensitivity_distribution']
            input_df['elasticity_climate_impact'] = self.scenario['elasticity_climate_impact']
            input_df['t2xco2_index'] = self.scenario['t2xco2_index']
            input_df['price_backstop_tech'] = self.scenario['price_backstop_tech']
            input_df['negative_emissions_possible'] = self.scenario['negative_emissions_possible']
        input_df.to_excel(writer, sheet_name='Input')
        # if self.uncertainty_dict:
        #     uncertainty_df = pd.DataFrame.from_dict(self.uncertainty_dict)
        #     uncertainty_df.to_excel(writer, sheet_name='Input Uncertainty')
        writer.close()
        return

    def write_to_sqlite(self, collection='executive variables'):
        def collect_executive_variables():
            executive_variables_dict = {'mu': self.economic_submodel.mu,
                                        'S': self.economic_submodel.S,
                                        'E': self.economic_submodel.E,
                                        'damages': self.economic_submodel.damages,
                                        'abatement_cost': self.economic_submodel.abatement_cost,
                                        'abatement_fraction': self.economic_submodel.abatement_fraction,
                                        'SLRDAMAGES': self.climate_submodel.SLRDAMAGES,
                                        'gross_output': self.economic_submodel.gross_output,
                                        'net_output': self.economic_submodel.net_output,
                                        'I': self.economic_submodel.I,
                                        'CPC': self.economic_submodel.CPC,
                                        'forc': self.carbon_submodel.forc,
                                        'temp_atm': self.climate_submodel.temp_atm,
                                        'global_damages': self.welfare_submodel.global_damages,
                                        'global_output': self.welfare_submodel.global_output,
                                        'global_period_util_ww': self.welfare_submodel.global_period_util_ww,
                                        # 'region_utility': self.welfare_submodel.region_util,
                                        'TOTAL_SLR': self.climate_submodel.TOTALSLR,
                                        'mat': self.carbon_submodel.mat,
                                        'forcoth': self.carbon_submodel.forcoth,
                                        'E_worldwide_per_year': self.carbon_submodel.E_worldwide_per_year,
                                        'labour_force': self.economic_submodel.labour_force,
                                        'total_factor_productivity': self.economic_submodel.total_factor_productivity,
                                        'capital_stock': self.economic_submodel.capital_stock,
                                        'sigma_ratio': self.economic_submodel.sigma_ratio,
                                        'Eind': self.economic_submodel.Eind,
                                        'sigma_gr': self.economic_submodel.sigma_gr,
                                        'damage_frac': self.economic_submodel.damage_fraction,
                                        # 'SLRTHERM': self.climate_submodel.SLRTHERM,
                                        # 'GSICCUM': self.climate_submodel.GSICCUM,
                                        # 'GISCUM': self.climate_submodel.GISCUM,
                                        # 'AISCUM': self.climate_submodel.AISCUM,
                                        'Eind_cum': self.economic_submodel.Eind_cum,
                                        'E_cum': self.economic_submodel.E_cum
                                        }

            exec_var_dict = {}
            for idx, region in enumerate(self.regions):
                exec_var_dict[region] = {}
                for key, item in executive_variables_dict.items():
                    if item.shape == (len(self.regions), len(self.years)):
                        exec_var_dict[region][key] = item[idx]
                    else:
                        exec_var_dict[region][key] = item[:]
            return exec_var_dict

        def collect_metrics_variables():
            metrics_dict = {}
            region = 'global'
            metrics_dict[region] = {}
            metrics = self.get_metrics()
            for idx, metric in enumerate(metrics):
                metrics_dict[region][f'metric_{idx}'] = metric
            # region = 'global'
            # metrics_dict = {region: {'global_period_util_ww': self.welfare_submodel.global_period_util_ww,
            #                          'global_output': self.welfare_submodel.global_output,
            #                          'temp_overshoots': self.welfare_submodel.temp_overshoots,
            #                          'global_emissions': self.carbon_submodel.E_worldwide_per_year}}
            return metrics_dict

        model_variables_to_sqlite = {}
        if collection == 'metrics':
            model_variables_to_sqlite = collect_metrics_variables()
        elif collection == 'executive variables':
            model_variables_to_sqlite = collect_executive_variables()

        df = pd.DataFrame()
        for region_key in model_variables_to_sqlite:
            df = pd.DataFrame.from_dict(model_variables_to_sqlite[region_key])
            df.index = self.years

        # Unpack dictionary with quintiles per region
        climate_impact_relative_to_capita_dict = {}
        for year in list(self.economic_submodel.climate_impact_relative_to_capita.keys()):
            climate_impact_relative_to_capita_dict[year] = {}
            for i in range(5):
                for idx, region in enumerate(self.regions):
                    climate_impact_relative_to_capita_dict[year][f'{region}_quintile_{i}'] = \
                        self.economic_submodel.climate_impact_relative_to_capita[year][i][idx]
        df_ = pd.DataFrame.from_dict(climate_impact_relative_to_capita_dict, orient='index')

        df = pd.merge(df, df_, left_index=True, right_index=True)

        df['mu_target'] = self.levers['mu_target']
        df['sr'] = self.levers['sr']
        df['irstp'] = self.levers['irstp']
        if self.scenario:
            df['SSP_scenario'] = self.scenario['SSP_scenario']
            df['Availability of fossil fuels'] = self.scenario['fosslim']
            df['climate_sensitivity_distribution'] = self.scenario['climate_sensitivity_distribution']
            df['elasticity_climate_impact'] = self.scenario['elasticity_climate_impact']
            df['t2xco2_index'] = self.scenario['t2xco2_index']
            df['price_backstop_tech'] = self.scenario['price_backstop_tech']
            df['negative_emissions_possible'] = self.scenario['negative_emissions_possible']

        conn = sqlite3.connect(self.database)

        df.to_sql(name=self.file_name, con=conn, if_exists='append')
        conn.commit()
        conn.close()

    def get_metrics(self):
        # Define metrics of the model
        # objective_function_value assumes minimization
        utilitarian_objective_function_value1 = -self.welfare_submodel.global_per_util_ww.sum()/10000

        # dict_obj = self.economic_submodel.climate_impact_relative_to_capita
        # sum_years = []
        # for year in dict_obj.keys():
        #     list1 = []
        #     # Take the first = worst (?) quintile and sum for all regions -> sum worst quintile globally for every year then sum again
        #     for quint in range(5):
        #         year_quint = dict_obj[year][quint]
        #         list1.append(year_quint)
        #     sum_years.append(sum(list1))
        # print(sum_years)
        # utilitarian_objective_function_value2 = sum(sum_years)

        utilitarian_objective_function_value2 = self.welfare_submodel.global_damage.sum()/1000

        # utilitarian_objective_function_value2 = -self.welfare_submodel.utility/1000000000
        utilitarian_objective_function_value3 = self.welfare_submodel.temp_overshoots.sum()
        return utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3

    # def get_metrics(self):
    #     # Define metrics of the model
    #     # objective_function_value assumes minimization
    #     utilitarian_objective_function_value1 = -self.welfare_submodel.global_period_util_ww.sum()/10000
    #     utilitarian_objective_function_value3 = self.welfare_submodel.temp_overshoots.sum()
    #     return utilitarian_objective_function_value1, utilitarian_objective_function_value3

    def POT_control_continuous(self, P):
        # Note that currently the indicator variables are hardcoded in, so check if they match the feature_names! (
        # also in the same order)
        t = 0
        for year in self.years:
            # Determine policy based on indicator variables
            policy, rules = P.evaluate(
                [self.carbon_submodel.mat[t], self.economic_submodel.net_output[:, t].sum(axis=0), year])

            policies = policy.split('|')
            for policy_ in policies[:-1]:
                if policy:
                    policy_unpacked = policy_.split('_')
                    policy_name = policy_unpacked[0]
                    policy_value = float(policy_unpacked[1])

                    # print(policy_name)

                    if policy_name == 'miu':
                        mu_target = policy_value
                    elif policy_name == 'sr':
                        sr = policy_value
                    elif policy_name == 'irstp':
                        irstp = policy_value

            # Run one timestep of RICE
            self.economic_submodel.run_gross(t, year, mu_target=mu_target, sr=sr)
            self.carbon_submodel.run(t, year, E=self.economic_submodel.E)
            self.climate_submodel.run(t, year, forc=self.carbon_submodel.forc,
                                      gross_output=self.economic_submodel.gross_output)
            self.economic_submodel.run_net(t, year, temp_atm=self.climate_submodel.temp_atm,
                                           SLRDAMAGES=self.climate_submodel.SLRDAMAGES)
            self.welfare_submodel.run_utilitarian(t, year, CPC=self.economic_submodel.CPC,
                                                  labour_force=self.economic_submodel.labour_force,
                                                  damages=self.economic_submodel.damages,
                                                  net_output=self.economic_submodel.net_output,
                                                  temp_atm=self.climate_submodel.temp_atm,
                                                  irstp=irstp)
            t += 1

        utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3 = self.get_metrics()
        # utilitarian_objective_function_value1, utilitarian_objective_function_value2 = self.get_metrics()

        # Save policy P with the objective function values
        if self.database:
            policy_dict = {'policy': str(P),
                           'utilitarian_ofv1': [utilitarian_objective_function_value1],
                           'utilitarian_ofv2': [utilitarian_objective_function_value2],
                           'utilitarian_ofv3': [utilitarian_objective_function_value3], }
            df = pd.DataFrame(data=policy_dict)

            conn = sqlite3.connect(self.database)
            df.to_sql(name=self.file_name, con=conn, if_exists='append')
            conn.commit()
            conn.close()

        # return utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3  # objective_function_value
        return utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3

    def POT_control_discrete(self, P):
        # Note that currently the indicator variables are hardcoded in, so check if they match the feature_names! (
        # also in the same order)
        t = 0
        for year in self.years:
            # Determine policy based on indicator variables
            policy, rules = P.evaluate(
                [self.carbon_submodel.mat[t], self.economic_submodel.net_output[:, t].sum(axis=0), year])

            # # mu_target = 2300
            # # sr = 0.1
            # # irstp = 0.015
            #
            # policy_unpacked = policy.split('_')
            # policy_name = policy_unpacked[0]
            # policy_value = float(policy_unpacked[1])
            #
            # # Initialize levers
            # # mu_target = 2135
            # # sr = 0.248
            # # irstp = 0.015
            mu_target = self.rng.integers(2100, 2250)
            sr = self.rng.uniform(0.2, 0.5)
            irstp = self.rng.uniform(0.01, 0.1)
            #
            #
            # if policy_name == 'miu':
            #     mu_target = policy_value
            # elif policy_name == 'sr':
            #     sr = policy_value
            # elif policy_name == 'irstp':
            #     irstp = policy_value

            if policy == 'miu_2065':
                mu_target = 2065
            elif policy == 'miu_2100':
                mu_target = 2100
            elif policy == 'miu_2180':
                mu_target = 2180
            elif policy == 'miu_2250':
                mu_target = 2250
            elif policy == 'miu_2305':
                mu_target = 2305
            elif policy == 'sr_01':
                sr = 0.1
            elif policy == 'sr_02':
                sr = 0.2
            elif policy == 'sr_03':
                sr = 0.3
            elif policy == 'sr_04':
                sr = 0.4
            elif policy == 'sr_05':
                sr = 0.5
            elif policy == 'irstp_0005':
                irstp = 0.005
            elif policy == 'irstp_0015':
                irstp = 0.015
            elif policy == 'irstp_0025':
                irstp = 0.025

            # Run one timestep of RICE
            self.economic_submodel.run_gross(t, year, mu_target=mu_target, sr=sr)
            self.carbon_submodel.run(t, year, E=self.economic_submodel.E)
            self.climate_submodel.run(t, year, forc=self.carbon_submodel.forc,
                                      gross_output=self.economic_submodel.gross_output)
            self.economic_submodel.run_net(t, year, temp_atm=self.climate_submodel.temp_atm,
                                           SLRDAMAGES=self.climate_submodel.SLRDAMAGES)
            self.welfare_submodel.run_utilitarian(t, year, CPC=self.economic_submodel.CPC,
                                                  labour_force=self.economic_submodel.labour_force,
                                                  damages=self.economic_submodel.damages,
                                                  net_output=self.economic_submodel.net_output,
                                                  temp_atm=self.climate_submodel.temp_atm,
                                                  irstp=irstp)
            t += 1

        utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3 = self.get_metrics()
        # utilitarian_objective_function_value1, utilitarian_objective_function_value2 = self.get_metrics()

        # Save policy P with the objective function values
        if self.database:
            policy_dict = {'policy': str(P),
                           'utilitarian_ofv1': [utilitarian_objective_function_value1],
                           'utilitarian_ofv2': [utilitarian_objective_function_value2],
                           'utilitarian_ofv3': [utilitarian_objective_function_value3], }
            df = pd.DataFrame(data=policy_dict)

            conn = sqlite3.connect(self.database)
            df.to_sql(name=self.file_name, con=conn, if_exists='append')
            conn.commit()
            conn.close()

        return utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3  # objective_function_value
        # return utilitarian_objective_function_value1, utilitarian_objective_function_value2

    def ema_workbench_control(self):
        t = 0
        for year in self.years:
            self.economic_submodel.run_gross(t, year, mu_target=self.levers['mu_target'], sr=self.levers['sr'])
            self.carbon_submodel.run(t, year, E=self.economic_submodel.E)
            self.climate_submodel.run(t, year, forc=self.carbon_submodel.forc,
                                      gross_output=self.economic_submodel.gross_output)
            self.economic_submodel.run_net(t, year, temp_atm=self.climate_submodel.temp_atm,
                                           SLRDAMAGES=self.climate_submodel.SLRDAMAGES)
            self.welfare_submodel.run_utilitarian(t, year, CPC=self.economic_submodel.CPC,
                                                  labour_force=self.economic_submodel.labour_force,
                                                  damages=self.economic_submodel.damages,
                                                  net_output=self.economic_submodel.net_output,
                                                  temp_atm=self.climate_submodel.temp_atm, irstp=self.levers['irstp'])
            t += 1
        utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3 = self.get_metrics()
        return utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3  # objective_function_value


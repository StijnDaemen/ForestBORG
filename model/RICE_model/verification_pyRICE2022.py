import pandas as pd
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(package_directory)

# df_pyRICE2022 = pd.read_excel(input_path+"/input_data/pyRICE2022_df_main.xlsx", index_col=0)
# df_IAM_RICE = pd.read_excel(input_path+"/input_data/test_RICE_verification.xlsx", index_col=0)
#
# df_pyRICE2022 = df_pyRICE2022[['Damages', 'Atmospheric temperature', 'Industrial emission', 'Total output']]
# df_IAM_RICE = df_IAM_RICE[['damages', 'temp_atm', 'Eind', 'net_output']]
#
# print(df_pyRICE2022.head())
# print(df_IAM_RICE.head())

df_old = pd.read_excel(input_path+"/input_data/pyRICE_2022 important variables for validation_11.xlsx", index_col=0)
df_new = pd.read_excel(input_path+"/input_data/test_RICE_verification.xlsx", index_col=0)

df_new = df_new[['mu',
                 'S',
                 'E',
                 'damages',
                 'abatement_cost',
                 'abatement_fraction',
                 'SLRDAMAGES',
                 'gross_output',
                 'net_output',
                 'I',
                 'CPC',
                 'forc',
                 'forcoth',
                 'temp_atm',
                 # *'temp_ocean',
                 # *'global_damages',
                 # *'global_output',
                 # *'global_period_util_ww',
                 'TOTAL_SLR',
                 'mat',
                 # *'mup',
                 # *'ml',
                 'E_worldwide_per_year',
                 'labour_force',
                 'total_factor_productivity',
                 'capital_stock',
                 'sigma_ratio',
                 'Eind',
                 'sigma_gr',
                 'damage_frac',
                 'SLRTHERM',
                 'GSICCUM',
                 'GISCUM',
                 'AISCUM']]

df_old = df_old[['miu',
                 'S',
                 'E',
                 'damages',
                 'Abatement_cost',
                 'Abatement_cost_RATIO',
                 'SLRDAMAGES',
                 'Y_gross',
                 'Y',
                 'I',
                 'CPC',
                 'forc',
                 'forcoth',
                 'temp_atm',
                 'TOTAL_SLR',
                 'mat',
                 'E_ww_per_year',
                 'labour_force',
                 'tfp',
                 'k',
                 'sigma_region',
                 'Eind',
                 'sigma_gr',
                 'dam_frac',
                 'SLRTHERM',
                 'GSICCUM',
                 'GISCUM',
                 'AISCUM']]

df_old.rename(columns={'miu': 'mu',
                      'Y_gross': 'gross_output',
                      'E_ww_per_year': 'E_worldwide_per_year',
                      'Abatement_cost': 'abatement_cost',
                      'Abatement_cost_RATIO': 'abatement_fraction',
                      'Y_gross': 'net_output',
                    'Y': 'net_output',
                    'tfp': 'total_factor_productivity',
                    'k': 'capital_stock',
                    'sigma_region': 'sigma_ratio',
                    'dam_frac': 'damage_frac',
                      }, inplace=True)

print(df_old.head())
print(df_new.head())

df_diff = df_new-df_old
df_perc = (df_new-df_old)/df_old
print(df_diff)
print(df_perc)

writer = pd.ExcelWriter('verification.xlsx')
df_diff.to_excel(writer, sheet_name=r"difference new_old")
df_perc.to_excel(writer, sheet_name=r"perc new_old_old")
writer.close()

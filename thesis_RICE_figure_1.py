from model.RICE_model.IAM_RICE import RICE
from optimizer.forest_borg import ForestBorg
from visualizations import Visualization
import pickle
import numpy as np
import time
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
path_to_dir = os.path.join(package_directory)


def optimization_RICE_ForestBorg_continuous(save_location, model_name, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval, version,
                               scenario=None, scenario_name=None):
    title_of_run = f'{model_name}_ForestBORG_continuous_seed{seed}_nfe{max_nfe}_depth{depth}_epsilons{epsilons}_gamma{gamma}_tau{tau}_restart{restart_interval}_v{version}'
    start = time.time()

    # Set up the model
    years_10 = []
    for i in range(2005, 2315, 10):
        years_10.append(i)

    regions = [
        "US",
        "OECD-Europe",
        "Japan",
        "Russia",
        "Non-Russia Eurasia",
        "China",
        "India",
        "Middle East",
        "Africa",
        "Latin America",
        "OHI",
        "Other non-OECD Asia",
    ]
    model = RICE(years_10, regions)
    # model = RICE(years_10, regions, scenario=scenario)

    # Set up the optimizer - ForestBORG
    master_rng = np.random.default_rng(seed)  # Master RNG
    snapshots = ForestBorg(pop_size=100, master_rng=master_rng,
                           model=model.POT_control_continuous,
                           metrics=['period_utility', 'damages', 'temp_overshoots'],
                           # Tree variables
                           action_names=['miu', 'sr', 'irstp'],
                           action_bounds=[[2065, 2305], [0.1, 0.5], [0.01, 0.1]],
                           feature_names=['mat', 'net_output', 'year'],
                           feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
                           max_depth=depth,
                           discrete_actions=False,
                           discrete_features=False,
                           # Optimization variables
                           mutation_prob=0.5,
                           max_nfe=max_nfe,  # 20000,
                           epsilons=epsilons,  # np.array([0.05, 0.05, 0.05]),
                           gamma=gamma,  # 4,
                           tau=tau,  # 0.02,
                           # restart_interval=restart_interval,
                           save_location=save_location,
                           title_of_run=title_of_run,
                           ).run()
    pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
    end = time.time()
    return print(f'Total elapsed time: {(end - start) / 60} minutes.')


if __name__ == '__main__':
    # Define input parameters for optimization run
    save_location = path_to_dir + '/output_data'
    model_name = "RICE"
    version = 1

    # Algorithm input variables
    max_nfe = 30000  # Determines the total number of offspring that are created
    depth = 3  # Determines the maximum tree depth of the tree structured decision variable
    epsilons = [0.05, 0.05, 0.05]  # Sets the margin of insignificance (when 2 solutions are perceived as equally optimal) per objective, so for a 3 objective problem the array should contain 3 epsilon values
    gamma = 4  # Sets the population-to-archive ratio
    tau = 0.02  #
    restart_interval = 5000
    seeds = [17, 42, 104, 303, 902]
    for seed in seeds:
        optimization_RICE_ForestBorg_continuous(save_location, model_name, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval, version,
                                   scenario=None, scenario_name=None)

    # Wait for optimization to complete...

    # Produce figure
    save = True
    visualizations = Visualization(save, model_name, seeds, max_nfe, depth, epsilons, gamma, tau, restart_interval, version)
    visualizations.runtime_dynamics_thesis_figure_1()

    # You can run figures 2, 4 and 5 based on the same data, so no need to rerun the optimization. If you still want to do so see separate scripts in this directory
    visualizations.search_operator_dynamics_thesis_figure_2()
    visualizations.parallel_axis_plot_thesis_figure_4()
    visualizations.individual_trees_thesis_figure_5()


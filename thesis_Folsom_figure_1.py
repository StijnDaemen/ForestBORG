from model.Folsom_model.folsom import Folsom
from optimizer.forest_borg import ForestBorg
from visualizations import VisualizationDiscrete
from model.Folsom_model.ptreeopt.opt import PTreeOpt
import logging
import pickle
import numpy as np
import time
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
path_to_dir = os.path.join(package_directory)


def Folsom_Herman_discrete(save_location, seed, max_nfe, depth, epsilons):
    title_of_run = f'Folsom_Herman_seed{seed}_nfe{max_nfe}_depth{depth}_epsilons{epsilons}_v1'
    np.random.seed(seed)

    start = time.time()

    model = Folsom('model/Folsom_model/folsom/data/folsom-daily-w2016.csv',
                   sd='1995-10-01', ed='2016-09-30', use_tocs=False, multiobj=True)

    algorithm = PTreeOpt(model.f,
                         feature_bounds=[[0, 1000], [1, 365], [0, 300]],
                         feature_names=['Storage', 'Day', 'Inflow'],
                         discrete_actions=True,
                         action_names=['Release_Demand', 'Hedge_90', 'Hedge_80',
                                       'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                         mu=20,  # number of parents per generation
                         cx_prob=0.70,  # crossover probability
                         population_size=100,
                         max_depth=depth,
                         multiobj=True,
                         epsilons=epsilons.tolist()  # [0.01, 1000, 0.01, 10]
                         )

    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    # With only 1000 function evaluations this will not be very good
    best_solution, best_score, snapshots = algorithm.run(max_nfe=max_nfe,  # 20000,
                                                         log_frequency=100,
                                                         snapshot_frequency=100)

    pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
    end = time.time()
    return print(f'Total elapsed time: {(end - start) / 60} minutes.')


def optimization_Folsom_ForestBorg_discrete(save_location, model_name, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval, version):
    title_of_run = f'{model_name}_ForestBORG_discrete_seed{seed}_nfe{max_nfe}_depth{depth}_epsilons{epsilons}_gamma{gamma}_tau{tau}_restart{restart_interval}_v{version}'
    start = time.time()

    # Set up the model
    model = Folsom('model/Folsom_model/folsom/data/folsom-daily-w2016.csv',
                   sd='1995-10-01', ed='2016-09-30', use_tocs=False, multiobj=True)
    master_rng = np.random.default_rng(seed)  # Master RNG

    # Set up the optimizer - ForestBORG
    snapshots = ForestBorg(pop_size=100,
                           model=model.f,
                           master_rng=master_rng,
                           # metrics=['period_utility', 'damages', 'temp_overshoots'],
                           # Tree variables
                           action_names=['Release_Demand', 'Hedge_90', 'Hedge_80',
                                         'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                           action_bounds=None,
                           feature_bounds=[[0, 1000], [1, 365], [0, 300]],
                           feature_names=['Storage', 'Day', 'Inflow'],
                           max_depth=depth,
                           discrete_actions=True,
                           discrete_features=None,
                           # Optimization variables
                           mutation_prob=0.5,
                           max_nfe=max_nfe,
                           epsilons=epsilons,  # [0.01, 1000, 0.01, 10],
                           gamma=gamma,  # 4,
                           tau=tau,  # 0.02,
                           restart_interval=restart_interval,
                           save_location=save_location,
                           title_of_run=title_of_run,
                           ).run()
    pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
    end = time.time()
    return print(f'Total elapsed time: {(end - start) / 60} minutes.')


if __name__ == '__main__':
    # Define input parameters for optimization run
    save_location = path_to_dir + '/output_data'
    model_name = "folsom"
    version = 1

    # Algorithm input variables
    max_nfe = 30000
    depth = 3
    epsilons = np.array([0.01, 1000, 0.01, 10])
    gamma = 4
    tau = 0.02
    restart_interval = 5000
    seed = 42
    # -- I & II -- Run time and operator dynamics - 5 seeds, other controls constant
    # Run the optimization of the Folsom lake model with the original POT algorithm/ Herman's algorithm
    Folsom_Herman_discrete(save_location, seed, max_nfe, depth, epsilons)
    # Run the optimization of the Folsom lake model with ForestBORG
    seeds = [17, 42, 104, 303, 902]
    for seed in seeds:
        optimization_Folsom_ForestBorg_discrete(save_location, model_name, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval, version)

    # Wait for optimization to complete...

    # Produce figure
    save = True
    visualizations = VisualizationDiscrete(save, model_name, seeds, max_nfe, depth, epsilons, gamma, tau, restart_interval, version)
    visualizations.runtime_dynamics_thesis_figure_1()

    # You can run figures 2, 4 and 5 based on the same data, so no need to rerun the optimization. If you still want to do so see separate scripts in this directory
    visualizations.search_operator_dynamics_thesis_figure_2()
    visualizations.parallel_axis_plot_thesis_figure_4()
    visualizations.individual_trees_thesis_figure_5()

# ForestBorg Package

ForestBorg is a Python package designed for optimizing decision trees using evolutionary algorithms. It leverages genetic operators to evolve trees based on specified metrics, action names and bounds, feature names and bounds, and other parameters to find optimal solutions for complex decision-making processes.

## Features

- Capable of handling multi-objective optimization problems through reinforcement learning.
- Utilizes a variety of genetic operators for tree mutation and crossover.
- Supports both discrete and continuous action and feature spaces.
- Implements restart mechanisms based on epsilon progress and population-to-archive ratio.
- Provides visualization tools for analyzing the evolutionary process and results.

## Installation

To install ForestBorg, you will need Python 3.6 or later. It is recommended to install ForestBorg in a virtual environment to avoid conflicts with other packages.

```bash
# Clone the repository
git clone https://github.com/StijnDaemen/ForestBORG.git
cd your-repository-directory

# It's recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package
pip install .
```

## Usage
To use the framework, you need to define your optimization problem, including the model for evaluating solutions, the objectives, and any constraints. Below is a high-level example of how to set up and run an optimization task:

```python
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
                           max_nfe=max_nfe, 
                           epsilons=epsilons,
                           gamma=gamma,
                           tau=tau,
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
```
The code snippit above divides the setup of the optimization in two: one part is set within a function handler and the other part falls directly under main.
The part in the function handler usually only needs to be set up once for every model that is coupled to ForestBORG. The inputs under main are more prone to change during testing or when running different experiments.

### Connecting a model
Connecting a model to ForestBORG is relatively straight forward. Before setting up the connection there are a few things to consider:
1) Which model variables do I want to use as actions? Also, are these actions discrete or continuous? If they are continuous what range of values can they be set to?
2) Which model variables do I want to use as features? These model variables should be most informative about system performance and preferably show trend behaviour and as little noise as possible. 
An extra note about action variables: If your model allows for one variable to be changed by the modeller, you have one (overarching) action. This terminology is slightly confusing as when you define different values for this (overarching) action, you will be inclined to call these specific actions, actions too. Especially in the case of discrete actions. E.g. in the Folsom model, the overarching action is (unexplicitly) 'water release'. However, the discrete actions are given names too, e.g. 'release_demand' or 'hedge_80'. These specified actions are called actions too.
   This distinction is important because currently ForestBORG supports only one overarching action if discrete actions are used (as in the Folsom lake model). If continuous actions are used, ForestBORG supports multiple overarching actions (e.g. mu, sr and irstp in the RICE model, which can all be set to different values on a continuous range).
   
When you know what variables in your model serve as indicators (=features) and which as actions and what range of values they can attain, you can connect the model to ForestBORG.
First, you have to create a function inside your model. For the RICE model this function is called 'POT_control_continuous' and looks like this:
```python
def POT_control_continuous(self, P):
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
    return utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3
```
This function takes as input a policy tree P and returns the objective function values. In general, the policy tree is evaluated and based on system condition an set/ set of actions are chosen according to the policy tree. One timestep of the model is run with this action/ set of actions which produces a reward for each metric. The rewards for each timestep are collected and serve as the output of this function; relates a reward to a policy tree. 
For every timestep in the model, the policy tree is unpacked (important: code the feature variables in the same order as the input for ForestBORG) and the system is evaluated through the feature variables. The policy tree points to a certain action based on the system evaluation. For a system with 3 overarching actions (mu, sr and irstp), these actions are set at every timestep. The model is then run for one timestep, which produces a reward based on the specified metrics. This is repeated for every timestep and at the end of the simulation all rewards are collected and outputted as objective function values for each metric. Note that over the simulation horizon the policy tree P is static but the actions are dynamic due to changing system conditions.
Once you have added this function within your model class, you can set the function parameters of ForestBORG.

### What is the output?
The output of the optimization run is a pickle file that contains snapshots of the optimization run.
The pickle file holds a dictionary, setup in forest_borg.py, that holds the following information:
```python
self.snapshot_dict = {'nfe': [],
                      'time': [],
                      'Archive_solutions': [],
                      'Archive_trees': [],
                      'epsilon_progress': [],
                      'mutation_operators': {},
                      'meta_info': {'metrics': self.metrics,
                                    'features': self.feature_names,
                                    'feature_bounds': self.feature_bounds,
                                    'actions': self.action_names,
                                    'action_bounds': self.action_bounds,
                                    'max_nfe': self.max_nfe,
                                    'max_tree_depth': self.max_depth,
                                    'epsilons': self.epsilons,
                                    'gamma': self.gamma,
                                    'tau': self.tau,
                                    'discrete_features': self.discrete_features,
                                    'discrete_actions': self.discrete_actions}}
```
Visualizations are used to turn such a dictionary of snapshots into comprehensible results. Note that currently the visualizations are specified to the RICE and Folsom models and need tweaking if other models are used. You can also create your own visualizations with the snapshots dictionary. The optimization run and the visualizations are not neccesarily connected.

## Visualization
The package contains a visualizations.py file which offers methods to plot the objective space, decision trees, and the distribution of genetic operators over time. The currently implemented visualizations are used in the thesis work 'Policy_Optimization_Trees_master_thesis_final.pdf'. The range of currently possible visualizations can be quickly understood from the Results section of the thesis.
Unlike the ForestBORG framework, the Visualization and VisualizationDiscrete classes are specifically tailored to the RICE and the Folsom models and aim to exactly reproduce the figures as used in the thesis. A generalized visualization class is not yet available.

[comment]: <> (## Contributing)

[comment]: <> (Contributions to the ForestBorg Optimization Framework are welcome! Please read CONTRIBUTING.md for guidelines on how to contribute to this project.)

## License
None
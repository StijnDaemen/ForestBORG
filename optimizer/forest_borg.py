import numpy as np
from optimizer.tree import PTree
import copy
import math
import time
import re

from collections import Counter
import itertools


class ForestBorg:
    def __init__(self, pop_size, master_rng,
                 model,
                 metrics,
                 action_names,
                 action_bounds,
                 feature_names,
                 feature_bounds,
                 max_depth,
                 discrete_actions,
                 discrete_features,
                 mutation_prob,
                 max_nfe,
                 epsilons,
                 gamma=4,
                 tau=0.02,
                 restart_interval=5000,
                 save_location=None,
                 title_of_run=None,):

        self.pop_size = pop_size

        self.rng_init = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_populate = np.random.default_rng(master_rng.integers(0, 1e9))
        # self.rng_natural_selection = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_tree = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_crossover = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_mutate = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_gauss = np.random.default_rng(master_rng.integers(0, 1e9))

        self.rng_iterate = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_tournament = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_population = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_revive = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_crossover_subtree = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_mutation_point = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_mutation_subtree = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_choose_operator = np.random.default_rng(master_rng.integers(0, 1e9))

        self.model = model

        self.metrics = metrics
        self.action_names = action_names
        self.action_bounds = action_bounds
        self.feature_names = feature_names
        self.feature_bounds = feature_bounds
        self.max_depth = max_depth
        self.discrete_actions = discrete_actions
        self.discrete_features = discrete_features
        self.mutation_prob = mutation_prob
        self.max_nfe = max_nfe
        self.epsilons = epsilons
        self.gamma = gamma
        self.tau = tau
        self.tournament_size = 2
        self.restart_interval = restart_interval

        self.epsilon_progress = 0
        self.epsilon_progress_counter = 0
        self.epsilon_progress_tracker = np.array([])
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

        self.number_of_restarts = 0

        if self.discrete_actions:
            self.GAOperators = {'mutation_random': [0],
                                'mutation_point_1': [0],
                                'mutation_subtree_1': [0],}
        else:
            self.GAOperators = {'mutation_random': [0],}
            n_actions = len(self.action_names)
            for i in range(1, n_actions + 1):
                self.GAOperators[f'mutation_point_{i}'] = [0]
                self.GAOperators[f'mutation_subtree_{i}'] = [0]

        self.nfe = 0

        self.start_time = time.time()

        self.save_location = save_location
        self.file_name = title_of_run

    def run(self, snapshot_frequency=100):
        self.population = np.array([self.spawn() for _ in range(self.pop_size)])
        print(f'size pop: {np.size(self.population)}')

        # Add the epsilon non-dominated solutions from the population to the Archive (initialize the Archive with the initial population, running add_to_Archive() function ensures no duplicates will be present.
        self.Archive = [self.population[0]]
        for sol in self.population:
            self.add_to_Archive(sol)
        print(f'size Archive: {np.size(self.Archive)}')

        # -- main loop -----------------------------------------

        last_snapshot = 0
        main_loop_counter = 1
        while self.nfe < self.max_nfe:
            self.iterate(main_loop_counter)
            main_loop_counter += 1

            if self.nfe >= last_snapshot + snapshot_frequency:
                last_snapshot = self.nfe
                # Record snapshot
                self.record_snapshot()

                intermediate_time = time.time()
                print(
                    f'\rnfe: {self.nfe}/{self.max_nfe} -- epsilon convergence: {self.epsilon_progress} -- elapsed time: {(intermediate_time - self.start_time) / 60} min -- number of restarts: {self.number_of_restarts}',
                    end='', flush=True)

        # -- Create visualizations of the run -------------------
        self.snapshot_dict['mutation_operators'] = self.GAOperators
        self.end_time = time.time()

        print(
            f'Total elapsed time: {(self.end_time - self.start_time) / 60} min -- {len(self.Archive)} non-dominated solutions were found.')
        return self.snapshot_dict

    def iterate(self, i):
        if i%self.restart_interval == 0:
            # Check gamma (the population to Archive ratio)
            gamma = len(self.population) / len(self.Archive)

            # Trigger restart if the latest epsilon tracker value is not different from the previous 10 -> 11th iteration without progress.
            # Officially in the borg paper I believe it is triggered if the latest epsilon tracker value is the same as the one of that before

            if self.check_unchanged(self.epsilon_progress_tracker):
                print('restart because of epsilon')
                self.restart(self.Archive, self.gamma, self.tau)
            # Check if gamma value warrants a restart (see Figure 2 in paper borg)
            elif (gamma > 1.25 * self.gamma) or (gamma < 0.75 * self.gamma):
                print('restart because of gamma')
                self.restart(self.Archive, self.gamma, self.tau)

        # Selection of recombination operator
        parents_required = 2
        parents = []
        # One parent is uniformly randomly selected from the archive
        parents.append(self.rng_iterate.choice(self.Archive))
        # The other parent(s) are selected from the population using tournament selection
        for parent in range(parents_required-1):
            parents.append(self.tournament(self.tournament_size))

        # Create the offspring
        offspring = Organism()
        offspring.dna = GAOperators.crossover_subtree(self, parents[0].dna, parents[1].dna)[0]
        # bloat control
        while offspring.dna.get_depth() > self.max_depth:
            offspring.dna = GAOperators.crossover_subtree(self, parents[0].dna, parents[1].dna)[0]
        # Let it mutate (in-built chance of mutation)
        offspring = self.mutate_with_feedbackloop(offspring)
        offspring.fitness = self.policy_tree_model_fitness(offspring.dna)
        self.nfe += 1

        # Add to population
        self.add_to_population(offspring)

        # Add to Archive if eligible
        self.add_to_Archive(offspring)

        # Update the epsilon progress tracker
        self.epsilon_progress_tracker = np.append(self.epsilon_progress_tracker, self.epsilon_progress) # self.epsilon_progress_tracker = np.append(self.epsilon_progress_tracker, self.epsilon_progress_counter)

        # Record GA operator distribution
        self.record_GAOperator_distribution()
        return

    def record_GAOperator_distribution(self):
        # Count the occurrences of each attribute value
        distribution = Counter(member.operator for member in self.Archive)
        for key in self.GAOperators.keys():
            if key in distribution:
                self.GAOperators[key].append(distribution[key])
            else:
                self.GAOperators[key].append(0)
        return

    def check_unchanged(self, arr):
        if len(arr) < 10:
            return False
        return len(set(arr[-10:])) == 1

    def mutate_with_feedbackloop(self, offspring):
        # Mutation based on performance feedback loop
        operator = GAOperators.choose_mutation_operator(self)

        if self.discrete_actions:
            # Mapping operators to their respective functions
            mutation_operations = {'mutation_random': lambda dna: GAOperators.mutation_random(self, dna),
                                   'mutation_point_1': lambda dna: GAOperators.mutation_point(self, dna, 1),
                                   'mutation_subtree_1': lambda dna: GAOperators.mutation_subtree(self, dna, 1),}
        else:
            # Mapping operators to their respective functions
            mutation_operations = {'mutation_random': lambda dna: GAOperators.mutation_random(self, dna),}
            n_actions = len(self.action_names)
            for i in range(1, n_actions + 1):
                mutation_operations[f'mutation_point_{i}'] = lambda dna, i=i: GAOperators.mutation_point(self, dna, i)
                mutation_operations[f'mutation_subtree_{i}'] = lambda dna, i=i: GAOperators.mutation_subtree(self, dna, i)

        # Execute the mutation operation based on the selected operator
        mutation_function = mutation_operations.get(operator)
        if mutation_function:
            offspring.dna = mutation_function(offspring.dna)
            offspring.operator = operator
        else:
            raise ValueError(f"Unknown mutation operator: {operator}")

        return offspring

    def restart(self, Archive, gamma, tau):
        print('triggered restart')
        current_Archive = copy.deepcopy(Archive)
        self.population = np.array([])
        self.population = np.array(current_Archive)
        new_size = gamma * len(current_Archive)
        # Inject mutated Archive members into the new population
        restart_counter = 0
        while len(self.population) < new_size:
            # Select a random solution from the Archive
            volunteer = self.rng_revive.choice(current_Archive)
            volunteer = self.mutate_with_feedbackloop(volunteer)
            volunteer.fitness = self.policy_tree_model_fitness(volunteer.dna)
            self.nfe += 1

            self.population = np.append(self.population, volunteer)

            # Update Archive with new solution
            self.add_to_Archive(volunteer)

            if restart_counter % 100 == 0:
                self.record_snapshot()

            if self.nfe > self.max_nfe:
                return

            restart_counter += 1
        # Adjust tournament size to account for the new population size
        self.tournament_size = max(2, math.floor(tau * new_size))
        self.number_of_restarts += 1

        # Record snapshot
        self.record_snapshot()
        return

    def tournament(self, k):
        # Choose k random members in the population
        members = self.rng_tournament.choice(self.population, k)
        # Winner is defined by pareto dominance.
        # If there are no winners, take a random member, if there are, take a random winner.
        winners = []
        for idx in range(len(members)-1):
            if self.dominates(members[idx].fitness, members[idx+1].fitness):
                winners.append(members[idx])
            elif self.dominates(members[idx+1].fitness, members[idx].fitness):
                winners.append(members[idx+1])

        if not winners:
            return self.rng_tournament.choice(members, 1)[0]
        else:
            return self.rng_tournament.choice(winners, 1)[0]

    def spawn(self):
        organism = Organism()
        organism.dna = self.random_tree()
        organism.fitness = self.policy_tree_model_fitness(organism.dna)
        return organism

    def random_tree(self, terminal_ratio=0.5):
        num_features = len(self.feature_names)

        depth = self.rng_tree.integers(1, self.max_depth + 1)
        L = []
        S = [0]

        while S:
            current_depth = S.pop()

            # action node
            if current_depth == depth or (current_depth > 0 and
                                          self.rng_tree.random() < terminal_ratio):
                if self.discrete_actions:
                    L.append([str(self.rng_tree.choice(self.action_names))])
                else:
                    action_input = ''
                    for idx, action in enumerate(self.action_names):
                        action_value = round(self.rng_tree.uniform(*self.action_bounds[idx]), 3)
                        action_input = action_input + f'{action}_{action_value}|'
                    L.append([action_input])
            else:
                x = self.rng_tree.choice(num_features)
                v = self.rng_tree.uniform(*self.feature_bounds[x])
                L.append([x, v])
                S += [current_depth + 1] * 2

        T = PTree(L, self.feature_names, self.discrete_features)
        T.prune()
        return T

    def policy_tree_model_fitness(self, T):
        metrics = np.array(self.model(T))
        return metrics

    def bounded_gaussian(self, x, bounds):
        # do mutation in normalized [0,1] to avoid sigma scaling issues
        lb, ub = bounds
        xnorm = (x - lb) / (ub - lb)
        x_trial = np.clip(xnorm + self.rng_gauss.normal(0, scale=0.1), 0, 1)

        return lb + x_trial * (ub - lb)

    def dominates(self, a, b):
        # assumes minimization
        # a dominates b if it is <= in all objectives and < in at least one
        # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
        # So to circumvent this but still allow multiobjective optimisation in different directions under the
        # constraint that every number is positive, just add a large number to every index.

        large_number = 1000000000
        a = a + large_number
        b = b + large_number

        return np.all(a <= b) and np.any(a < b)

    def compare(self, solution1, solution2):
        #      Returns -1 if the first solution dominates the second, 1 if the
        #         second solution dominates the first, or 0 if the two solutions are
        #         mutually non-dominated.

        # then use epsilon dominance on the objectives
        dominate1 = False
        dominate2 = False

        for i in range(len(solution1)):
            o1 = solution1[i]
            o2 = solution2[i]

            epsilon = float(self.epsilons[i % len(self.epsilons)])
            i1 = math.floor(o1 / epsilon)
            i2 = math.floor(o2 / epsilon)

            if i1 < i2:
                dominate1 = True

                if dominate2:
                    return 0
            elif i1 > i2:
                dominate2 = True

                if dominate1:
                    return 0

        if not dominate1 and not dominate2:
            dist1 = 0.0
            dist2 = 0.0

            for i in range(len(solution1)):
                o1 = solution1[i]
                o2 = solution2[i]

                epsilon = float(self.epsilons[i % len(self.epsilons)])
                i1 = math.floor(o1 / epsilon)
                i2 = math.floor(o2 / epsilon)

                dist1 += math.pow(o1 - i1 * epsilon, 2.0)
                dist2 += math.pow(o2 - i2 * epsilon, 2.0)

            if dist1 < dist2:
                return -1
            else:
                return 1
        elif dominate1:
            return -1
        else:
            return 1

    def same_box_platypus(self, solution1, solution2):

        # then use epsilon dominance on the objectives
        dominate1 = False
        dominate2 = False

        for i in range(len(solution1)):
            o1 = solution1[i]
            o2 = solution2[i]

            epsilon = float(self.epsilons[i % len(self.epsilons)])
            i1 = math.floor(o1 / epsilon)
            i2 = math.floor(o2 / epsilon)

            if i1 < i2:
                dominate1 = True

                if dominate2:
                    return False
            elif i1 > i2:
                dominate2 = True

                if dominate1:
                    return False

        if not dominate1 and not dominate2:
            return True
        else:
            return False

    def add_to_Archive(self, solution):
        flags = [self.compare(solution.fitness, s.fitness) for s in self.Archive]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]
        dominated = [x < 0 for x in flags]
        not_same_box = [not self.same_box_platypus(solution.fitness, s.fitness) for s in self.Archive]

        if any(dominates):
            return False
        else:
            self.Archive = list(itertools.compress(self.Archive, nondominated)) + [solution]

            if dominated and not_same_box:
                self.epsilon_progress += 1

    def add_to_population(self, offspring):
        # If the offspring dominates one or more population members, the offspring replaces
        # one of these dominated members randomly.

        #  If the offspring is dominated by at least one population member, the offspring
        #  is not added to the population.

        # Otherwise, the offspring is nondominated and replaces a randomly selected member
        # of the population.

        flags = [self.compare(offspring.fitness, s.fitness) for s in self.Archive]
        dominates = [x > 0 for x in flags]
        dominated = [x < 0 for x in flags]

        # If dominated, find a random solution in the population that can be replaced with the offspring
        # Find indices of all -1 values
        indices = [i for i, x in enumerate(flags) if x == -1]

        if indices:  # Check if there are any -1 values at all
            # Randomly choose one index to keep as -1
            keep_index = self.rng_population.choice(indices)

            # Set all other -1 values to 2
            for i in indices:
                if i != keep_index:
                    flags[i] = 2

        keep = [x > -1 for x in flags]

        # Apply the logic for inclusion of the offspring into the population
        if any(dominates):
            return False
        elif any(dominated):
            self.population = list(itertools.compress(self.population, keep)) + [offspring]
        else:
            # Randomly select an index from population
            idx = self.rng_population.integers(0, len(self.population) - 1)
            self.population[idx] = offspring
        return

    def record_snapshot(self):
        self.snapshot_dict['nfe'].append(self.nfe)
        self.snapshot_dict['time'].append((time.time() - self.start_time) / 60)
        self.snapshot_dict['Archive_solutions'].append([item.fitness for item in self.Archive])
        self.snapshot_dict['Archive_trees'].append([item.dna for item in self.Archive])
        self.snapshot_dict['epsilon_progress'].append(self.epsilon_progress)
        return


class GAOperators(ForestBorg):
    def choose_mutation_operator(self, zeta=1):
        if self.discrete_actions:
            operators = ['mutation_random', 'mutation_point_1', 'mutation_subtree_1']
        else:
            operators = ['mutation_random']
            n_actions = len(self.action_names)
            for i in range(1, n_actions+1):
                operators.append(f'mutation_point_{i}')
                operators.append(f'mutation_subtree_{i}')

        # Initially give every operator an equal chance, then feedback loop based on occurance in self.Archive
        operator_dict = {}
        for operator in operators:
            num_solutions_operator = 0
            for member in self.Archive:
                if member.operator == operator:
                    num_solutions_operator += 1
            operator_dict[operator] = num_solutions_operator+zeta

        probability_dict = {}
        for operator in operator_dict.keys():
            resultset = np.array([value for key, value in operator_dict.items()]).sum()
            probability = operator_dict[operator] / (resultset)
            probability_dict[operator] = probability

        return self.rng_choose_operator.choice(list(probability_dict.keys()), p=list(probability_dict.values()))

    def crossover_subtree(self, P1, P2):
        P1, P2 = [copy.deepcopy(P) for P in (P1, P2)]
        # should use indices of ONLY feature nodes
        feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
        feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
        index1 = self.rng_crossover_subtree.choice(feature_ix1)
        index2 = self.rng_crossover_subtree.choice(feature_ix2)
        slice1 = P1.get_subtree(index1)
        slice2 = P2.get_subtree(index2)
        P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
        P1.build()
        P2.build()
        return (P1, P2)

    def mutation_subtree(self, T, nr_actions):
        T = copy.deepcopy(T)
        # should use indices of ONLY feature nodes
        feature_ix = [i for i in range(T.N) if T.L[i].is_feature]
        index = self.rng_mutation_subtree.choice(feature_ix)
        slice = T.get_subtree(index)
        for node in T.L[slice]:
            if node.is_feature:  # if isinstance(node, Feature):
                low, high = self.feature_bounds[node.index]
                if node.is_discrete:
                    node.threshold = self.rng_mutation_subtree.integers(low, high + 1)
                else:
                    node.threshold = self.bounded_gaussian(
                        node.threshold, [low, high])
            else:
                if self.discrete_actions:
                    node.value = str(self.rng_mutation_subtree.choice(self.action_names))
                else:
                    action_input = node.value
                    for _ in range(nr_actions):
                        # randomly pick an action and a new value from its action bounds
                        num_actions = len(self.action_names)
                        x = self.rng_mutation_subtree.choice(num_actions)
                        action_name = self.action_names[x]
                        action_value_new = round(self.rng_mutation_subtree.uniform(*self.action_bounds[x]), 3)
                        # wrap it in a string, consistent with later processing
                        action_substring = f'{action_name}_{action_value_new}|'
                        # Replace the substring in the large string
                        pattern = re.escape(action_name) + r".*?\|"
                        action_input = re.sub(pattern, action_substring, action_input)
                    node.value = action_input
        return T

    def mutation_point(self, T, nr_actions):
        # Point mutation at either feature or action node
        T = copy.deepcopy(T)
        item = self.rng_mutation_point.choice(T.L)
        if item.is_feature:
            low, high = self.feature_bounds[item.index]
            if item.is_discrete:
                item.threshold = self.rng_mutation_point.integers(low, high + 1)
            else:
                item.threshold = self.bounded_gaussian(
                    item.threshold, [low, high])
        else:
            if self.discrete_actions:
                item.value = str(self.rng_mutation_point.choice(self.action_names))
            else:
                action_input = item.value
                for _ in range(nr_actions):
                    # randomly pick an action and a new value from its action bounds
                    num_actions = len(self.action_names)
                    x = self.rng_mutation_point.choice(num_actions)
                    action_name = self.action_names[x]
                    action_value_new = round(self.rng_mutation_point.uniform(*self.action_bounds[x]), 3)
                    # wrap it in a string, consistent with later processing
                    action_substring = f'{action_name}_{action_value_new}|'
                    # Replace the substring in the large string
                    pattern = re.escape(action_name) + r".*?\|"
                    action_input = re.sub(pattern, action_substring, action_input)
                item.value = action_input
        return T

    def mutation_random(self, P, mutate_actions=True):
        P = copy.deepcopy(P)

        for item in P.L:
            if self.rng_mutate.random() < self.mutation_prob:
                if item.is_feature:
                    low, high = self.feature_bounds[item.index]
                    if item.is_discrete:
                        item.threshold = self.rng_mutate.integers(low, high + 1)
                    else:
                        item.threshold = self.bounded_gaussian(
                            item.threshold, [low, high])
                elif mutate_actions:
                    if self.discrete_actions:
                        item.value = str(self.rng_mutate.choice(self.action_names))
                    else:
                        action_input = ''
                        for idx, action in enumerate(self.action_names):
                            action_value = round(self.rng_tree.uniform(*self.action_bounds[idx]), 3)
                            action_input = action_input + f'{action}_{action_value}|'
                        item.value = action_input

        return P


class Organism:
    def __init__(self):
        self.dna = None
        self.fitness = None
        self.operator = None

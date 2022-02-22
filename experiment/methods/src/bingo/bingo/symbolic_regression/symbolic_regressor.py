import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from bingo.evolutionary_optimizers.island import Island
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.stats.hall_of_fame import HallOfFame
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression import AGraphGenerator, ExplicitRegression, \
    ExplicitTrainingData


class SymbolicRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, population_size=500, stack_size=32,
                 operators=None, use_simplification=False,
                 crossover_prob=0.4, mutation_prob=0.4,
                 metric="mse", parallel=False, clo_alg="lm",
                 generations=int(1e30), fitness_threshold=1.0e-16,
                 max_time=1800, evolutionary_algorithm="age fitness",
                 island="normal"):
        self.population_size = population_size
        self.stack_size = stack_size

        if operators is None:
            operators = {"+", "-", "*", "/"}
        self.operators = operators

        self.use_simplification = use_simplification

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.metric = metric

        self.parallel = parallel

        self.clo_alg = clo_alg

        self.generations = generations
        self.fitness_threshold = fitness_threshold
        self.max_time = max_time

        if evolutionary_algorithm == "deterministic crowding":
            self.evolutionary_algorithm = DeterministicCrowdingEA
        else:
            self.evolutionary_algorithm = AgeFitnessEA

        if island == "fitness predictor":
            self.island = FitnessPredictorIsland
        else:
            self.island = Island

        self.best_ind = None

    def set_params(self, **params):
        # TODO not clean
        new_params = self.get_params()
        new_params.update(params)
        super().set_params(**new_params)
        self.__init__(**new_params)
        return self

    def _get_archipelago(self, X, y):
        self.component_generator = ComponentGenerator(X.shape[1])
        for operator in self.operators:
            self.component_generator.add_operator(operator)

        self.crossover = AGraphCrossover()
        self.mutation = AGraphMutation(self.component_generator)

        self.generator = AGraphGenerator(self.stack_size, self.component_generator,
                                         use_simplification=self.use_simplification)

        training_data = ExplicitTrainingData(X, y)
        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
        local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm=self.clo_alg)
        evaluator = Evaluation(local_opt_fitness)

        if self.evolutionary_algorithm == AgeFitnessEA:
            ea = self.evolutionary_algorithm(evaluator, self.generator, self.crossover,
                                             self.mutation, self.crossover_prob, self.mutation_prob,
                                             self.population_size)

        else:  # DeterministicCrowdingEA
            ea = self.evolutionary_algorithm(evaluator, self.crossover, self.mutation,
                                             self.crossover_prob, self.mutation_prob)

        island = self.island(ea, self.generator, self.population_size)

        # TODO pareto front based on complexity?
        hof = HallOfFame(5)

        if self.parallel:
            return ParallelArchipelago(island, hall_of_fame=hof)
        else:
            return SerialArchipelago(island, hall_of_fame=hof)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            print("sample weight not None, TODO")
            raise NotImplementedError

        self.archipelago = self._get_archipelago(X, y)
        opt_result = self.archipelago.evolve_until_convergence(
            max_generations=self.generations,
            fitness_threshold=self.fitness_threshold,
            max_time=self.max_time)
        # print(opt_result.ea_diagnostics)
        self.best_ind = self.archipelago.hall_of_fame[0]
        # print("------------------hall of fame------------------", self.archipelago.hall_of_fame, sep="\n")
        # print("\nbest individual:", self.best_ind)

    def predict(self, X):
        return self.best_ind.evaluate_equation_at(X)


if __name__ == '__main__':
    # SRSerialArchipelagoExample with SymbolicRegressor
    import random
    random.seed(7)
    np.random.seed(7)
    x = np.linspace(-10, 10, 100).reshape([-1, 1])
    y = x**2 + 3.5*x**3

    regr = SymbolicRegressor(population_size=100, stack_size=10,
                             operators=["+", "-", "*"],
                             use_simplification=True,
                             crossover_prob=0.4, mutation_prob=0.4, metric="mae",
                             parallel=False, clo_alg="lm", generations=500, fitness_threshold=1.0e-4,
                             evolutionary_algorithm="age fitness", island="normal")
    print(regr.get_params())

    regr.fit(x, y)
    print(regr.best_ind)

    # TODO MPI.COMM_WORLD.bcast in parallel?
    # TODO rank in MPI

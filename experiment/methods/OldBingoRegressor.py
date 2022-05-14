from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from bingo.evolutionary_optimizers.island import Island

hyper_params = [
    # (100, 24), (100, 64), (500, 24), (500, 48), (2500, 16), (2500, 32)
    {"population_size": [100], "stack_size": [24]},
    {"population_size": [100], "stack_size": [64]},
    {"population_size": [500], "stack_size": [24]},
    {"population_size": [500], "stack_size": [48]},
    {"population_size": [2500], "stack_size": [16]},
    {"population_size": [2500], "stack_size": [32]}
]

est = SymbolicRegressor(population_size=500, stack_size=24,
                        operators=["+", "-", "*", "/",
                                   "sin", "cos", "exp", "log"],
                        use_simplification=True,
                        crossover_prob=0.3, mutation_prob=0.45, metric="mse",
                        parallel=False, clo_alg="lm", max_time=2*60*60, max_evals=int(5e5),
                        evolutionary_algorithm=AgeFitnessEA,
                        island=FitnessPredictorIsland,
                        clo_threshold=1.0e-5)


def complexity(est):
    return est.best_ind.get_complexity()


def model(est):
    return str(est.best_ind).replace(")(", ")*(").replace("^", "**")  # sympy string


if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(7)
    np.random.seed(7)
    x = np.linspace(-10, 10, 10).reshape([-1, 1])
    y = x**2 + 3.5*x**3

    est.fit(x, y)

    print(complexity(est))
    print(model(est))

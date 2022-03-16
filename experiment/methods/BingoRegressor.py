from bingo_w_clo.symbolic_regression.symbolic_regressor import SymbolicRegressor
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from bingo.evolutionary_optimizers.island import Island

hyper_params = [{
    "population_size": (100, 500, 2500),  # 2 hr time increase
    "stack_size": (16, 24, 32, 48, 64),
    # age fitness, simplification True,
    # mutation and crossover rates

    # TODO see if CLO tolerance matters
    # optimize during final selection

    # "evolutionary_algorithm": (AgeFitnessEA, DeterministicCrowdingEA),
    # "island": (FitnessPredictorIsland, Island),
    # "use_simplification": (False, True)

    # "crossover_prob": (0.1, 0.3, 0.4, 0.5),
    # "mutation_prob": (0.1, 0.3, 0.4, 0.5)

    # "clo_threshold": (1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-14)

    # ea (agefitnessea, deterministic) & island (coev of fitness predictors, fitness predictor, ), & simplification

    # pareto front vs. Hof operations,

    # operations

    # metric?
    # semi-hyperparam study
    # crossover mutation probs
    # operations "exp", "log", "sqrt"?
    # population size and generations
    # max runtime
}]

# coev of fitness predictors
# fitness predictor island


est = SymbolicRegressor(population_size=100, stack_size=32,
                        operators=["+", "-", "*", "/",
                                   "sin", "cos", "exp", "log"],
                        use_simplification=True,
                        crossover_prob=0.4, mutation_prob=0.4, metric="mse",
                        parallel=False, clo_alg="lm", max_time=2*60*60, max_evals=int(5e5),
                        evolutionary_algorithm=AgeFitnessEA,
                        island=Island,
                        clo_threshold=1.0e-3)

def complexity(est):
    return est.best_ind.get_complexity()


def model(est):
    return str(est.best_ind)


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

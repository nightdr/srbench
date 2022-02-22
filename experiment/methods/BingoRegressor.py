from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor

hyper_params = [{
    # "population_size": (100, 500, 2500),
    # "stack_size": (16, 32, 64),
    "evolutionary_algorithm": ("age fitness", "deterministic crowding"),
    "island": ("normal", "fitness predictor"),
    "use_simplification": (False, True)

    # ea (agefitnessea, deterministic) & island (coev of fitness predictors, fitness predictor, ), & simplification

    # mutation and crossover rates

    # pareto front vs. Hof operations,

    # "use_simplification": (True, False),
    # metric?
    # semi-hyperparam study
    # crossover mutation probs
    # operations "exp", "log", "sqrt"?
    # population size and generations
    # max runtime
}]

# coev of fitness predictors
# fitness predictor island


est = SymbolicRegressor(population_size=100, stack_size=10,
                        operators=["+", "-", "*", "/",
                                   "sin", "cos", "exp", "log"],
                        use_simplification=True,
                        crossover_prob=0.4, mutation_prob=0.4, metric="mae",
                        parallel=False, clo_alg="lm", max_time=1800,
                        evolutionary_algorithm="age fitness",
                        island="normal")


def complexity(est):
    return est.best_ind.get_complexity()


def model(est):
    return str(est.best_ind)


if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(7)
    np.random.seed(7)
    x = np.linspace(-10, 10, 100).reshape([-1, 1])
    y = x**2 + 3.5*x**3

    est.fit(x, y)

    print(complexity(est))
    print(model(est))
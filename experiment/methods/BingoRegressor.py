from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor

hyper_params = [{
    "population_size": (100, 500, 2500),  # 2 hr time increase
    "stack_size": (16, 24, 32, 48, 64),
    # age fitness, simplification True,
    # mutation and crossover rates
    # see if CLO tolerance matters

    "island": ("normal", "fitness predictor"),

    # "evolutionary_algorithm": ("age fitness", "deterministic crowding"),

    # "use_simplification": (False, True)

    # ea (agefitnessea, deterministic) & island (coev of fitness predictors, fitness predictor, ), & simplification

    # pareto front vs. Hof operations,



    # operations

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


est = SymbolicRegressor(population_size=100, stack_size=32,
                        operators=["+", "-", "*", "/",
                                   "sin", "cos", "exp", "log"],
                        use_simplification=True,
                        crossover_prob=0.4, mutation_prob=0.4, metric="mse",
                        parallel=False, clo_alg="BFGS", max_time=1800, max_evals=int(5e5),
                        evolutionary_algorithm="age fitness",
                        island="fitness predictor")


def complexity(est):
    return est.best_ind.get_complexity()


def model(est):
    return str(est.best_ind)


if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(7)
    np.random.seed(7)
    x = np.linspace(-10, 10, 11).reshape([-1, 1])
    y = x**2 + 3.5*x**3

    est.fit(x, y)

    print(complexity(est))
    print(model(est))
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland

params = {
    "clo_alg": "lm",
    "clo_threshold": 1e-05,
    "crossover_prob": 0.3,
    "evolutionary_algorithm": AgeFitnessEA,
    "fitness_threshold": 1e-16,
    "generations": 1000000000000000019884624838656,
    "island": FitnessPredictorIsland,
    "max_evals": 1000000,
    "max_time": 8*60*60,
    "metric": "mse",
    "mutation_prob": 0.45,
    "operators": [
        "+",
        "-",
        "*",
        "/",
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt"  # NOTE: added sqrt
    ],
    "parallel": False,
    "population_size": 500,
    "stack_size": 24,
    "use_simplification": True
}

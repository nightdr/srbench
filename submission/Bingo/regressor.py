from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland

hyper_params = [
    # (100, 24), (100, 64), (500, 24), (500, 48), (2500, 16), (2500, 32)
    {"population_size": [100], "stack_size": [24]},
    {"population_size": [100], "stack_size": [64]},
    {"population_size": [500], "stack_size": [24]},
    {"population_size": [500], "stack_size": [48]},
    {"population_size": [2500], "stack_size": [16]},
    {"population_size": [2500], "stack_size": [32]}
]

"""
est: a sklearn-compatible regressor.
"""
est = SymbolicRegressor(population_size=500, stack_size=24,
                        operators=["+", "-", "*", "/",
                                   "sin", "cos", "exp", "log"],
                        use_simplification=True,
                        crossover_prob=0.3, mutation_prob=0.45, metric="mse",
                        parallel=False, clo_alg="lm", max_time=2*60*60, max_evals=int(5e5),
                        evolutionary_algorithm=AgeFitnessEA,
                        island=FitnessPredictorIsland,
                        clo_threshold=1.0e-5)

# TODO wrapping in CV class?
# want to tune your estimator? wrap it in a sklearn CV class.


def model(est, X=None):
    """
    Return a sympy-compatible string of the final model. 

    Parameters
    ----------
    est: sklearn regressor
        The fitted model. 
    X: pd.DataFrame, default=None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model. 

    Notes
    -----

    Ensure that the variable names appearing in the model are identical to 
    those in the training data, `X`, which is a `pd.Dataframe`. 
    If your method names variables some other way, e.g. `[x_0 ... x_m]`, 
    you can specify a mapping in the `model` function such as:

        ```
        def model(est, X):
            mapping = {'x_'+str(i):k for i,k in enumerate(X.columns)}
            new_model = est.model_
            for k,v in mapping.items():
                new_model = new_model.replace(k,v)
        ```
    """
    model_str = str(est.best_ind)

    # replace X_# with data variables names
    mapping = {'X_' + str(i): k for i, k in enumerate(X.columns)}
    for k,v in mapping.items():
        model_str = model_str.replace(k,v)

    model_str = model_str.replace(")(", ")*(").replace("^", "**")  # replace operators for sympy
    return model_str

################################################################################
# Optional Settings
################################################################################


"""
eval_kwargs: a dictionary of variables passed to the evaluate_model()
    function. 
    Allows one to configure aspects of the training process.

Options 
-------
    test_params: dict, default = None
        Used primarily to shorten run-times during testing. 
        for running the tests. called as 
            est = est.set_params(**test_params)
    max_train_samples:int, default = 0
        if training size is larger than this, sample it. 
        if 0, use all training samples for fit. 
    scale_x: bool, default = True 
        Normalize the input data prior to fit. 
    scale_y: bool, default = True 
        Normalize the input label prior to fit. 
    pre_train: function, default = None
        Adjust settings based on training data. Called prior to est.fit. 
        The function signature should be (est, X, y). 
            est: sklearn regressor; the fitted model. 
            X: pd.DataFrame; the training data. 
            y: training labels.
"""

def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X 
       versus relative to FEAT's batch size setting. 
    """
    if est.batch_size < len(X):
        est.gens = int(est.gens*len(X)/est.batch_size)
    print('FEAT gens adjusted to',est.gens)
    # adjust max dim
    est.max_dim=min(max(est.max_dim, X.shape[1]), 20)
    print('FEAT max_dim set to',est.max_dim)

# define eval_kwargs.
eval_kwargs = dict(
                   pre_train=my_pre_train_fn,
                   test_params = {'gens': 5,
                                  'pop_size': 10
                                 }
                  )

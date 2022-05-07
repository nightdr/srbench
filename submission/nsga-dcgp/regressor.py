import numpy as np
import pandas as pd

from nsga import NSGA
from dcgp import Parameter, DifferentialCGP
"""
est: a sklearn-compatible regressor. 
    if you don't have one they are fairly easy to create. 
    see https://scikit-learn.org/stable/developers/develop.html
"""
dcgp_params = Parameter(
    n_output=1,
    n_row=10, n_col=10, n_constant=1,
    primitive_set=None,
    levels_back=None
)
est = NSGA(
    DifferentialCGP, dcgp_params,
    pop_size=1000, n_gen=1000, n_parent=200, prob=0.4, nsga=True,
    newton_step=10, stop=1e-6, verbose=10
)


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

    If you have special operators such as protected division or protected log,
    you will need to handle these to assure they conform to sympy format. 
    One option is to replace them with the unprotected versions. Post an issue
    if you have further questions: 
    https://github.com/cavalab/srbench/issues/new/choose
    """
    mappings = {'x'+str(i): k for i, k in enumerate(X.columns)}
    model_ = est.expr()
    for k, v in reversed(mappings.items()):
        model_ = model_.replace(k, v)
    return model_

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
    if len(X) <= 1000:
        max_time = 3600 - 10
    else:
        max_time = 36000 - 10
    est.max_time = max_time


# define eval_kwargs.
eval_kwargs = dict(
    pre_train=my_pre_train_fn
)


"""Test myself"""
dataset = np.loadtxt('/home/luoyuanzhen/STORAGE/dataset/sr_benchmark/Keijzer-9_train.txt')
X, y = pd.DataFrame(dataset[:, :-1], columns=['x0']), dataset[:, -1]
n_variable = X.shape[1]
my_pre_train_fn(est, X, y)
est.fit(X, y)
print(model(est, X))

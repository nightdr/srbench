from experiment.methods.BingoRegressor import complexity, model, est
# from ..src.bingo.bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
from experiment.methods.tuned.params._bingoregressor import params

est.set_params(**params)
est.max_time = 8*60*60  # 8 hours
est.max_evals = int(1e6)  # 1,000,000 evals

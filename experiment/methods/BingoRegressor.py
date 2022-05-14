from submission.Bingo.regressor import est as BingoEst

hyper_params = []

est = BingoEst


def complexity(est):
    return est.get_best_individual().get_complexity()


def model(est):
    return str(est.get_best_individual()).replace(")(", ")*(").replace("^", "**")


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

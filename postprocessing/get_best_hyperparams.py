import pandas as pd
import matplotlib.pyplot as plt


def visualize_params(feather_path, hyperparams):
    df = pd.read_feather(feather_path)

    df["params"] = df["params"].replace(
        {"<class 'bingo.evolutionary_optimizers.fitness_predictor_island.FitnessPredictorIsland'>": "Fitness Predictor",
         "<class 'bingo.evolutionary_optimizers.island.Island'>": "Island",
         "<class 'bingo.evolutionary_algorithms.age_fitness.AgeFitnessEA'>": "Age Fitness",
         "<class 'bingo.evolutionary_algorithms.deterministic_crowding.DeterministicCrowdingEA'>": "Det Crowding"}, regex=True)
    df["hyperparam combinations"] = df["params"].apply(lambda x: x[1:-1])

    print(df["params"].iloc[50])
    df[hyperparams] = df["hyperparam combinations"].str.split(", ", n=1, expand=True)

    plot_labels = hyperparams + ["params"]
    x_labels = hyperparams + [
        f"hyperparam combinations ({', '.join(hyperparams)})"]
    for i, (plot_label, x_label) in enumerate(zip(plot_labels, x_labels)):
        value_counts = df[plot_label].value_counts()
        plt.bar(list(value_counts.index), list(value_counts))
        # plt.title(f"{x_label.title()} Occurrences")
        plt.xlabel(x_label.title())
        if i == len(plot_labels) - 1:
            plt.xticks(rotation=-45)
        plt.xticks(rotation=-45)
        plt.ylabel("Occurrences")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    visualize_params(r"../pop_and_stack_long/black-box_results.feather", ["population size", "stack size"])
    visualize_params(r"../mutation_and_crossover/black-box_results.feather", ["crossover_rate", "mutation_rate"])
    visualize_params(r"../ea_island_simplification_runs/black-box_results.feather", ["island", "evolutionary_algorithm"])

    # TODO look at experiments with low r^2 values

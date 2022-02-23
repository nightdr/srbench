import pandas as pd
import matplotlib.pyplot as plt


def visualize_params(feather_path, hyperparams):
    df = pd.read_feather(feather_path)
    df["hyperparam combinations"] = df["params"].apply(lambda x: x[1:-1])
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
        plt.ylabel("Occurrences")
        plt.show()


if __name__ == '__main__':
    visualize_params(r"../pop_and_stack_runs/black-box_results.feather", ["population size", "stack size"])

    # TODO look at experiments with low r^2 values

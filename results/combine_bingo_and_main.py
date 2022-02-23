import pandas as pd

if __name__ == '__main__':
    main_feather = pd.read_feather("black-box_results.feather")
    my_feather = pd.read_feather("my_black-box_results.feather")
    my_feather["symbolic_alg"] = True

    combined_feather = pd.concat((main_feather, my_feather), axis=0)

    # fix indexing so Bingo runs have correct indices
    combined_feather = combined_feather.reset_index(drop=True)
    print(combined_feather)

    combined_feather.to_feather("combined_black-box_results.feather")

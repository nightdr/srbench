python analyze.py ../../../pmlb/datasets/1089_USCrime/1089_USCrime.tsv.gz -results ../testing_47 -ml BingoRegressor --slurm -A hochhalter-np -q hochhalter-shared-np -n_jobs 1 --noskips -skip_tuning
# python analyze.py ../../../pmlb/datasets/1089_USCrime/1089_USCrime.tsv.gz -results ../testing_47 -ml BingoRegressor --local -n_jobs 1 --noskips -skip_tuning

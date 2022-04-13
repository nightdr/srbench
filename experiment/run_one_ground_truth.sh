#!/bin/bash
python analyze.py \
"../../../pmlb/datasets/strogatz_vdp2/strogatz_vdp2.tsv.gz" \
-results ../ground_truth_runs_hochhalter \
-target_noise 0 \
-sym_data \
-n_trials 1 \
-m 4096 \
-time_limit 9:00 \
-tuned \
-ml tuned.BingoRegressor \
--slurm \
-A hochhalter-np \
-q hochhalter-shared-np

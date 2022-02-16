#!/usr/bin/bash
#SBATCH -o ../test_runs/1027_ESL/1027_ESL_BingoRegressor_23654_evaluate_model.%J.out
#SBATCH --error=../test_runs/1027_ESL/1027_ESL_BingoRegressor_23654_evaluate_model.%J.err
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -J 1027_ESL_BingoRegressor_23654_evaluate_model
#SBATCH --account=hochhalter-np # account - abbreviated by -A 
#SBATCH --partition=hochhalter-shared-np # partition, abbreviated by -p # # set data and working directories
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=16384

conda info
source plg_modules.sh

python evaluate_model.py ../../../pmlb/test_datasets/1027_ESL/1027_ESL.tsv.gz -ml BingoRegressor -results_path ../test_runs/1027_ESL/ -seed 23654  -target_noise 0.0  -feature_noise 0.0

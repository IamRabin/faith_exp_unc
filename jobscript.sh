#!/bin/bash
#SBATCH --job-name="expunc_wofaith"
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH -w g001
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 5-12:00 # time (D-HH:MM)
#SBATCH -o outputs/slurm.%N.%j.out
#SBATCH -e outputs/slurm.%N.%j.err
##SBATCH --exclusive

set -euo pipefail

RUNPATH="/home/rabindra/D1/projects/explain-uncertainity"
mkdir -p /home/rabindra/D1/projects/explain-uncertainity/outputs/
cd $RUNPATH


pwd

#module purge
#module load slurm/21.08.8
module load cuda12.2/toolkit


# for SEED in 0 1 2
# do
#   echo "===== RUNNING SEED $SEED ====="

# srun python scripts/train_lightning.py \
#      --dataset cifar10 \
#      --seed 2 \
#      --split_seed 0 \
#      --method ours \
#      --model vit \
#      --epochs 170 \
#      --batch_size 32 \
#      --num_workers 4 \
#      --lr 0.034322 \
#      --weight_decay 0.0004928 \
#      --K 4 \
#      --eps_raw 0.0078431373 \
#      --beta 10 \
#      --tau 0 \
#      --kappa 2 \
#      --lambda_sal 0.045687 \
#      --lambda_unc 0.08927 \
#      --lambda_faith 0.30832 \
#      --T 2.0 \
#     --devices 4


# done




#-----------------------------
#Paths (EDIT THESE)
#-----------------------------
RUN_CE="/home/rabindra/D1/projects/explain-uncertainity/runs/cifar10/standard_ce/seed2_20260124_185020"
RUN_OURS="/home/rabindra/D1/projects/explain-uncertainity/runs/cifar100/ours/seed2_20260126_102010"

FIG_DIR="/home/rabindra/D1/projects/explain-uncertainity/figures"

# -----------------------------
# Shared eval params
# -----------------------------
SPLIT="test"
DELINS_STEPS=20
DELINS_BATCHES=10

# Captum explainers params
IG_STEPS=32
SHAP_BASELINES=8
ATTR_BASELINE_IG="zero"
ATTR_BASELINE_SHAP="mean"

# -----------------------------
# Helper functions
# -----------------------------
run_eval () {
 local RUN_DIR="$1"
 local EXPL="$2"
 local EXTRA_ARGS="$3"

 echo "=== EVAL: run_dir=${RUN_DIR} explainer=${EXPL} ==="
 srun python scripts/eval.py \
   --run_dir "${RUN_DIR}" \
   --split "${SPLIT}" \
   --dataset cifar100 \
   --compute_delins \
   --delins_steps "${DELINS_STEPS}" \
   --delins_batches "${DELINS_BATCHES}" \
   --explainer "${EXPL}" \
   ${EXTRA_ARGS}
}

run_plot () {
 local EXPL="$1"
 local TITLE="$2"

 echo "=== PLOT: explainer=${EXPL} ==="
 srun python scripts/plot_delins.py \
   --inputs \
     "${RUN_CE}/delins_curves_${SPLIT}_${EXPL}.pt" \
     "${RUN_OURS}/delins_curves_${SPLIT}_${EXPL}.pt" \
   --labels "CE" "Ours" \
   --output_dir "${FIG_DIR}/${EXPL}" \
   --title "${TITLE}" \
   --ylim 0,1
}

# -----------------------------
# Run all explainers
# -----------------------------

run_eval "${RUN_CE}"   "vanilla" ""
run_eval "${RUN_OURS}" "vanilla" ""
run_plot "vanilla" "CIFAR-100 (Vanilla Gradients)"

run_eval "${RUN_CE}"   "mean" ""
run_eval "${RUN_OURS}" "mean" ""
run_plot "mean" "CIFAR-100 (Noise-averaged Gradients / Mean Saliency)"

run_eval "${RUN_CE}"   "ig" "--ig_steps ${IG_STEPS} --attr_baseline ${ATTR_BASELINE_IG}"
run_eval "${RUN_OURS}" "ig" "--ig_steps ${IG_STEPS} --attr_baseline ${ATTR_BASELINE_IG}"
run_plot "ig" "CIFAR-10 (Integrated Gradients)"

run_eval "${RUN_CE}"   "deepshap" "--shap_baselines ${SHAP_BASELINES} --attr_baseline ${ATTR_BASELINE_SHAP}"
run_eval "${RUN_OURS}" "deepshap" "--shap_baselines ${SHAP_BASELINES} --attr_baseline ${ATTR_BASELINE_SHAP}"
run_plot "deepshap" "CIFAR-100 (Gradient SHAP)"

run_eval "${RUN_CE}"   "smoothgrad" "--smoothgrad_samples 16 --smoothgrad_stdev 0.1"
run_eval "${RUN_OURS}" "smoothgrad" "--smoothgrad_samples 16 --smoothgrad_stdev 0.1"
run_plot "smoothgrad" "CIFAR-100 (SmoothGrad)"

run_eval "${RUN_CE}"   "captum_grad" ""
run_eval "${RUN_OURS}" "captum_grad" ""
run_plot "captum_grad" "CIFAR-100 (Captum Gradient)"

echo "✅ All eval + plots complete."
echo "Figures saved under: ${FIG_DIR}/"




# srun python scripts/tuner.py \
#  --study_name cifar100_ours_tune \
#  --storage sqlite:///optuna.db \
#  --out_dir runs/optuna \
#  --dataset cifar100 \
#  --model resnet18 \
#  --epochs 50 \
#  --devices 1 \
#  --precision 16-mixed \
#  --n_trials 15 \
#  --monitor val/acc1 \
#  --seed 0 \
#  --split_seed 0


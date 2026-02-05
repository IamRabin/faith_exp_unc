# Faithful Explanations under Uncertainty

Training-time explanation uncertainty and faithfulness on CIFAR models.

This repo focuses on saliency-based explanation stability/uncertainty and how to train models with explanation-aware objectives. The main training path uses PyTorch Lightning; evaluation scripts compute invariance, stability, and deletion/insertion metrics.

## Project Structure

- `src/` — core library code
  - `explain/` — saliency, perturbations, attribution helpers
  - `losses/` — explanation-aware training loss
  - `metrics/` — stability/invariance/deletion-insertion metrics
  - `models/` — ResNet/ViT builders
  - `train/` — Lightning module and optimizer helpers
- `scripts/` — training, evaluation, plotting, and visualization scripts
- `configs/`, `data/`, `runs/`, `outputs/` — configs, datasets, experiment outputs

## Setup

Recommended (minimal) dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The core dependencies are:
- `torch`, `torchvision`
- `lightning`
- `captum`
- `numpy`, `pyyaml`

If you use Optuna tuning or plotting scripts, you will also need:
- `optuna`
- `matplotlib`

## Training (Lightning)

Main training entrypoint:

```bash
python scripts/train_lightning.py \
  --dataset cifar10 \
  --model resnet18 \
  --method ours \
  --epochs 170 \
  --batch_size 128
```

Notes:
- `--method standard_ce` runs standard cross-entropy.
- `--method ours` enables explanation-aware training via `ExplanationTrainingLoss`.
- For ViT, SDPA math backend forcing is handled inside `LitCifar` when needed.

## Evaluation

Evaluate invariance, stability, and optional deletion/insertion metrics from a Lightning checkpoint:

```bash
python scripts/eval.py \
  --run_dir runs/cifar10/ours/seed0_xxx \
  --split test \
  --compute_delins
```

Outputs:
- `metrics_<split>_<explainer>.json`
- `eval_metrics_<explainer>.csv`
- `delins_curves_<split>_<explainer>.pt` (if enabled)

## Plotting

Plot deletion/insertion curves:

```bash
python scripts/plot_delins.py \
  --inputs runs/.../delins_curves_test_vanilla.pt runs/.../delins_curves_test_vanilla.pt \
  --labels CE Ours \
  --output_dir figures/vanilla
```

Plot attribution distributions:

```bash
python scripts/plot_attr_distributions.py \
  --run_ce runs/cifar10/standard_ce/seed0_xxx \
  --run_ours runs/cifar10/ours/seed0_xxx \
  --explainer mean
```

Visualize attribution overlays:

```bash
python scripts/visualize_attributions.py \
  --run_ce runs/cifar10/standard_ce/seed0_xxx \
  --run_ours runs/cifar10/ours/seed0_xxx \
  --n_images 8 \
  --only_correct
```

## Perturbation Visualization

Generate CIFAR-10 perturbation strips:

```bash
python scripts/save_cifar10_perturbations.py \
  --n 20 \
  --scale 6
```

Use `--no_upsample` to disable upsampling.

## Reproducibility Notes

- Seeds are set in training/evaluation scripts.
- DDP behavior is handled by Lightning; for ViT with explanation loss, SDPA math backend is used when needed.

## License

Apache 2.0. See `LICENSE`.

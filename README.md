# Rheed-Folo

Training and evaluation of FOLO, the spot-locating CNN of the FastRHEED pipeline.

## Training on synthetic data

`train_synthetic.py` is the script version of `Generated_Flow.ipynb` (minus the
hls4ml conversion). All constants live in a YAML config:

```bash
python train_synthetic.py configs/synthetic.yaml
```

The run writes `Models/<output.model_name>/` containing the saved model, a copy
of the config, `history.json`, `test_metrics.json` and a few
image / label / prediction plots under `plots/`.

To evaluate a saved model without retraining, set `training.train: false` and
`training.load_from: Models/<name>` in the config.

## Training on real data

`train_real.py` is the script version of `Real_Flow.ipynb` (minus the hls4ml
conversion), configured by `configs/real.yaml`:

```bash
python train_real.py configs/real.yaml
```

`data.path` points at the h5 file in `FastRHEED/data/` and is relative to the
directory the script is run from. Outputs go to `Models/<output.model_name>/`
as above.

## Training on real and synthetic data

`train_hybrid.py` trains the `train_real.py` model on a mix of real and synthetic
frames, configured by `configs/hybrid.yaml`:

```bash
python train_hybrid.py configs/hybrid.yaml
```

`hybrid.synthetic_fraction` (default 0.5) is the synthetic share of the training
and validation sets. Real frames are split and labelled as in `train_real.py`,
synthetic ones are generated as in `train_synthetic.py` (`gaussians` section).
The model is tested on `dataset.test_size` real and `dataset.test_size`
synthetic frames separately; `test_metrics.json` holds both losses and
`plots/real/` and `plots/synthetic/` the example plots.

To benchmark it over seeds: `python run_trials.py --out <dir> --train train_hybrid.py --config configs/hybrid.yaml`.

## Multi-seed benchmark

`run_trials.py` trains FOLO with N seeds (default 30) and benchmarks every model through
Rheed-Eval's `inference.py` pipeline on the synthetic frames of `synthetic_ood.h5` and on
five real frames. On an interactive Slurm allocation the seeds run in parallel (3 workers
per GPU by default):

```bash
salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node=1 --gpus-per-node=4 -A <account> --no-shell
python run_trials.py --out $SCRATCH/trials/<name> --jobid <id> [--config configs/real.yaml] [--train train_real.py]
```

`--out` gets the synthetic hit rates at 4/8/12/16/20 px (`hit_rates.md`, also printed, and
`hit_rates.csv`), the TopCrop boxes of all models overlaid on each frame
(`topcrop_overlay_synth.pdf`, `topcrop_overlay_real.pdf`) and every trained model
(`models/seed_<k>/`, with `models/r<r>px_<best|median|worst>` links to the best, median
and worst one at each radius). The run is resumable: rerun the same command after a crash.
After a change to the eval code, `--eval-only` re-evaluates all the models without
retraining. See the docstring for all options.

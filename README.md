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
(`topcrop_overlay_synth.pdf`, `topcrop_overlay_real.pdf`) and the best, median and worst
model at each radius (`models/`). The run is resumable: rerun the same command after a
crash. See the docstring for all options.

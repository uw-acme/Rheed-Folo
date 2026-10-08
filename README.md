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

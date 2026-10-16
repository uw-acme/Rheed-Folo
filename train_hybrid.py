"""Train the FOLO QKeras model on a mix of real and synthetic RHEED frames.

Combines ``train_real.py`` and ``train_synthetic.py``. All constants live in a
YAML config file:

    python train_hybrid.py configs/hybrid.yaml

Pipeline:
    1. Real frames: split the growths into train / validation / test and
       pseudo-label sampled frames, exactly as ``train_real.py``.
    2. Synthetic frames: 2D-Gaussian images with soft labels, exactly as
       ``train_synthetic.py``.
    3. Mix them: ``hybrid.synthetic_fraction`` of the training and validation
       sets is synthetic, the rest real (training reshuffles the mixed set every
       epoch).
    4. Build (or load) the quantized residual ``baby_yolo`` model (shared with
       ``train_real.py`` and ``train_synthetic.py``) and train it with a weighted BCE loss, early stopping
       and LR reduction.
    5. Test separately on real and synthetic test sets (``test_size`` frames
       each), then save the model, the config used, the training history, the
       test losses and a few image / label / prediction plots per source.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from train_real import RealGenerator, load_splits
from train_real import load_config as load_real_config
from train_synthetic import (
    SyntheticGenerator,
    build_model,
    load_model,
    make_datasets,
    make_loss_wbce,
    parse_angle,
    save_plots,
    setup_tensorflow,
    train,
)


# ============================================================================
# Config
# ============================================================================


def load_config(path: str | Path) -> dict:
    cfg = load_real_config(path)

    g = cfg["gaussians"]
    g["min_theta"] = parse_angle(g["min_theta"])
    g["max_theta"] = parse_angle(g["max_theta"])
    if g["min_num"] > g["max_num"]:
        raise ValueError("gaussians.min_num must be <= gaussians.max_num")

    if not 0.0 <= cfg["hybrid"]["synthetic_fraction"] <= 1.0:
        raise ValueError("hybrid.synthetic_fraction must be in [0, 1]")

    return cfg


# ============================================================================
# Mixing
# ============================================================================


def split_sizes(total: int, synthetic_fraction: float) -> tuple[int, int]:
    """(real, synthetic) counts that add up to ``total``."""
    n_synthetic = int(round(total * synthetic_fraction))
    return total - n_synthetic, n_synthetic


def mix(real, synthetic) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the (images, labels) of both sources. Not shuffled here: make_datasets
    shuffles the whole training set every epoch, and the order of validation doesn't matter."""
    return np.concatenate([real[0], synthetic[0]]), np.concatenate([real[1], synthetic[1]])


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"])

    # ---- Data -----------------------------------------------------------
    d = cfg["dataset"]
    frac = cfg["hybrid"]["synthetic_fraction"]
    n_train_real, n_train_syn = split_sizes(d["training_size"], frac)
    n_val_real, n_val_syn = split_sizes(d["validation_size"], frac)
    print(f"[Train Mix]: {n_train_real} real + {n_train_syn} synthetic")
    print(f"[Val Mix]: {n_val_real} real + {n_val_syn} synthetic")

    splits = load_splits(cfg, rng)
    real_gen = RealGenerator(cfg, rng)
    train_real = real_gen.gen_data(n_train_real, splits["Train"], desc="Train (real)")
    val_real = real_gen.gen_data(n_val_real, splits["Val"], desc="Val (real)")
    test = {"real": real_gen.gen_data(d["test_size"], splits["Test"], transform=False, desc="Test (real)")}
    del splits

    syn_gen = SyntheticGenerator(cfg, rng)
    train_syn = syn_gen.gen_data(n_train_syn, desc="Train (synthetic)")[:2]
    val_syn = syn_gen.gen_data(n_val_syn, desc="Val (synthetic)")[:2]
    test["synthetic"] = syn_gen.gen_data(d["test_size"], desc="Test (synthetic)")[:2]

    train_img, train_label = mix(train_real, train_syn)
    val_img, val_label = mix(val_real, val_syn)
    del train_real, train_syn, val_real, val_syn

    # ---- Model ----------------------------------------------------------
    tf = setup_tensorflow(cfg["seed"])
    loss_fn = make_loss_wbce(cfg["loss"]["pos_weight"], cfg["loss"]["neg_weight"])

    t = cfg["training"]
    if t.get("load_from"):
        model = load_model(t["load_from"], loss_fn)
    else:
        model = build_model(cfg)

    # ---- Train ----------------------------------------------------------
    history = None
    if t["train"]:
        train_dataset, val_dataset = make_datasets(
            tf, cfg, train_img, train_label, val_img, val_label
        )
        history = train(tf, cfg, model, loss_fn, train_dataset, val_dataset)
    elif not t.get("load_from"):
        print("[Warning]: training.train is false and no training.load_from given; "
              "the model is untrained.")

    model.summary()

    # ---- Test (each source separately) ----------------------------------
    predictions, test_losses = {}, {}
    for source, (img, label) in test.items():
        predictions[source] = model.predict(img, batch_size=d["batch_size"])
        test_losses[f"test_loss_wbce_{source}"] = float(loss_fn(label, predictions[source]).numpy())
        print(f"[Test Loss (wBCE), {source}]: {test_losses[f'test_loss_wbce_{source}']:.6f}")

    # ---- Save -----------------------------------------------------------
    o = cfg["output"]
    model_dir = Path(o["models_dir"]) / o["model_name"]
    if o["save_model"]:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir))
        print(f"[Saved Model]: {model_dir}")

        shutil.copy(args.config, model_dir / "config.yaml")
        if history is not None:
            with open(model_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f, indent=2)
        with open(model_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_losses, f, indent=2)

    if o["num_plot_examples"] > 0:
        plot_dir = (model_dir if o["save_model"] else Path(o["models_dir"])) / "plots"
        for source, (img, label) in test.items():
            save_plots(plot_dir / source, img, label, predictions[source], o["num_plot_examples"])


if __name__ == "__main__":
    main()

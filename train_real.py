"""Train the FOLO QKeras model on real RHEED frames.

Script version of ``Real_Flow.ipynb`` (everything upstream of the hls4ml
conversion). All constants live in a YAML config file:

    python train_real.py configs/real.yaml

Pipeline:
    1. Split the growths of the h5 file into train / validation / test and
       sample frames from each, zero-padding them up to the model input size.
    2. Pseudo-label every frame: smooth the log-scaled spot region, threshold
       it, and put a soft bump at the center of mass of each connected blob.
       Train / validation frames get a random circular shift (label shifted
       to match); test frames are left in place.
    3. Build (or load) the quantized residual ``baby_yolo`` model and train it
       with a weighted BCE loss, early stopping and LR reduction.
    4. Save the model, the config used, the training history and a few
       image / label / prediction plots.

Shared pieces (TensorFlow setup, loss, loading, training loop, plotting) are
imported from ``train_synthetic.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.ndimage import center_of_mass, gaussian_filter, label as label_blobs
from tqdm import tqdm

from train_synthetic import (
    load_model,
    make_datasets,
    make_loss_wbce,
    save_plots,
    setup_tensorflow,
    train,
)


# ============================================================================
# Config
# ============================================================================


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    img = cfg["image"]
    if img["x"] % img["scale_factor"] or img["y"] % img["scale_factor"]:
        raise ValueError("image.x and image.y must be multiples of image.scale_factor")
    img["label_x"] = img["x"] // img["scale_factor"]
    img["label_y"] = img["y"] // img["scale_factor"]

    return cfg


# ============================================================================
# Real data loading / labeling
# ============================================================================


def load_splits(cfg: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Shuffle the h5 groups, split them, and sample frames from each split."""
    d = cfg["data"]
    splits = {}

    with h5py.File(d["path"], "r") as h5:
        growths = [g for g in h5.keys() if g not in d["exclude_groups"]]
        rng.shuffle(growths)  # Shuffle Growths

        n_train, n_val, n_test = d["training_growths"], d["validation_growths"], d["test_growths"]
        if n_train + n_val + n_test > len(growths):
            raise ValueError(
                f"Requested {n_train + n_val + n_test} growths but the file has {len(growths)}"
            )
        split_growths = {
            "Train": growths[:n_train],
            "Val": growths[n_train : n_train + n_val],
            "Test": growths[n_train + n_val : n_train + n_val + n_test],
        }

        for name, names in split_growths.items():
            print(f"Raw {name} Data Set:")
            frames = []
            for growth in names:
                indices = rng.choice(h5[growth].shape[0], size=d["images_per_growth"], replace=False)
                indices.sort()
                frames.append(h5[growth][indices])
                print(f"[Growth]: {growth:<25}, [Shape]: {frames[-1].shape}")
            splits[name] = np.concatenate(frames)
            print(f"[{name} Data Set Shape]: {splits[name].shape}")

    return splits


class RealGenerator:
    """Samples padded real frames and builds their soft FOLO pseudo-labels."""

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.rng = rng
        self.image_x = cfg["image"]["x"]
        self.image_y = cfg["image"]["y"]
        self.scale_factor = cfg["image"]["scale_factor"]
        self.label_x = cfg["image"]["label_x"]
        self.label_y = cfg["image"]["label_y"]
        self.labels = cfg["labels"]
        self.augment = cfg["augment"]

        self.yy, self.xx = np.meshgrid(
            np.arange(self.label_y), np.arange(self.label_x), indexing="ij"
        )

    def pad(self, frame: np.ndarray) -> np.ndarray:
        """Zero-pad a raw (H, W) frame symmetrically to (image_y, image_x, 1)."""
        pad_h = self.image_y - frame.shape[0]
        pad_w = self.image_x - frame.shape[1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"Frame {frame.shape} is larger than image size "
                             f"({self.image_y}, {self.image_x})")
        pad_config = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        return np.pad(frame.astype(np.float32), pad_config, mode="constant")[..., None]

    def make_soft_label(self, centers) -> np.ndarray:
        """Max of Gaussian bumps at ``centers`` (label-cell coordinates)."""
        label = np.zeros((self.label_y, self.label_x), dtype=np.float32)
        sigma = self.labels["soft_label_sigma"]

        for gx, gy in centers:
            bump = np.exp(-((self.xx - gx) ** 2 + (self.yy - gy) ** 2) / (2 * sigma**2))
            label = np.maximum(label, bump)

        return label[..., None]

    def orig_label(self, img: np.ndarray) -> np.ndarray:
        clip = self.labels["clip"]
        cropped = img[clip["top"] : clip["bottom"], clip["left"] : clip["right"], 0]
        smoothed = gaussian_filter(np.log1p(cropped), sigma=self.labels["smoothing_sigma"])
        threshold = np.percentile(smoothed, self.labels["percentile"])
        blobs, num_features = label_blobs(smoothed > threshold)
        centers = center_of_mass(smoothed, blobs, range(1, num_features + 1))

        return self.make_soft_label(
            [
                ((x + clip["left"]) / self.scale_factor, (y + clip["top"]) / self.scale_factor)
                for y, x in centers
            ]
        )

    def img_shift(self, img: np.ndarray, img_label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self.augment
        x_roll = self.rng.integers(low=a["min_x_roll"], high=a["max_x_roll"])
        y_roll = self.rng.integers(low=a["min_y_roll"], high=a["max_y_roll"])

        img = np.roll(img, y_roll, axis=0)
        img = np.roll(img, x_roll, axis=1)

        dy = int(y_roll // self.scale_factor)
        dx = int(x_roll // self.scale_factor)
        label_shifted = np.roll(img_label, shift=(dy, dx), axis=(0, 1))

        return img, label_shifted

    def img_gen(self, frames: np.ndarray, transform: bool) -> tuple[np.ndarray, np.ndarray]:
        index = self.rng.integers(low=0, high=frames.shape[0])
        img = self.pad(frames[index])
        img_label = self.orig_label(img)

        if transform:
            img, img_label = self.img_shift(img, img_label)

        # Convert to 8 bit int
        img_label = (img_label * 255).astype(np.uint8)
        img = img.astype(np.uint8)

        return img, img_label

    def gen_data(
        self, num_images: int, frames: np.ndarray, transform: bool = True, desc: str = ""
    ) -> tuple[np.ndarray, np.ndarray]:
        img_arr = np.empty((num_images, self.image_y, self.image_x, 1), dtype=np.float32)
        label_arr = np.empty((num_images, self.label_y, self.label_x, 1), dtype=np.float32)

        for i in tqdm(range(num_images), desc=desc):
            img, label = self.img_gen(frames, transform)
            img_arr[i] = img.astype(np.float32) / 256.0  # Normalize values to [0, 1)
            label_arr[i] = label.astype(np.float32) / 256.0

        print(f"[{desc} Images Shape]: {img_arr.shape}")
        print(f"[{desc} Labels Shape]: {label_arr.shape}")

        return img_arr, label_arr


# ============================================================================
# Model
# ============================================================================


def build_model(cfg: dict):
    from tensorflow.keras.activations import relu, sigmoid
    from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Input
    from tensorflow.keras.models import Model
    from qkeras import QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu

    m = cfg["model"]
    total_bits, integer_bits = m["total_bits"], m["integer_bits"]

    w_quant = quantized_bits(
        total_bits, integer_bits, symmetric=False, keep_negative=True, alpha=1
    )
    relu_quant = quantized_relu(total_bits, integer_bits)

    input_layer = Input(shape=(cfg["image"]["y"], cfg["image"]["x"], 1))

    x = input_layer
    for i, filters in enumerate(m["block_filters"], start=1):
        x = QConv2D(
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            name=f"qconv2d_{i}_1",
            kernel_quantizer=w_quant,
            kernel_initializer="lecun_uniform",
        )(x)
        x = QActivation(relu_quant, name=f"qact_{i}_1")(x)
        y = QConv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"qconv2d_{i}_2",
            kernel_quantizer=w_quant,
            kernel_initializer="lecun_uniform",
        )(x)
        y = BatchNormalization(name=f"b_{i}_2")(y)
        x = Add(name=f"add_{i}")([x, y])

    n = len(m["block_filters"])
    x = Conv2D(m["head_filters"], 1, padding="same", use_bias=True, name=f"qconv2d_{n + 1}")(x)
    x = Activation(relu, name=f"qact_{n + 1}")(x)

    x = Conv2D(1, 1, padding="same", use_bias=True, name=f"qconv2d_{n + 2}")(x)
    x_prob = Activation(sigmoid, name="sigmoid_out")(x)

    model = Model(inputs=input_layer, outputs=x_prob, name=m["name"])

    expected = (cfg["image"]["label_y"], cfg["image"]["label_x"], 1)
    if tuple(model.output_shape[1:]) != expected:
        raise ValueError(
            f"Model output shape {model.output_shape[1:]} does not match label shape "
            f"{expected}. len(model.block_filters) must equal log2(image.scale_factor)."
        )
    return model


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
    splits = load_splits(cfg, rng)
    gen = RealGenerator(cfg, rng)
    d = cfg["dataset"]
    train_img, train_label = gen.gen_data(d["training_size"], splits["Train"], desc="Train")
    val_img, val_label = gen.gen_data(d["validation_size"], splits["Val"], desc="Val")
    test_img, test_label = gen.gen_data(d["test_size"], splits["Test"], transform=False, desc="Test")
    del splits

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

    # ---- Test -----------------------------------------------------------
    predictions = model.predict(test_img, batch_size=d["batch_size"])
    print(f"[Test Shape]: {test_label.shape}")
    print(f"[Prediction Shape]: {predictions.shape}")
    test_loss = float(loss_fn(test_label, predictions).numpy())
    print(f"[Test Loss (wBCE)]: {test_loss:.6f}")

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
            json.dump({"test_loss_wbce": test_loss}, f, indent=2)

    if o["num_plot_examples"] > 0:
        plot_dir = (model_dir if o["save_model"] else Path(o["models_dir"])) / "plots"
        save_plots(plot_dir, test_img, test_label, predictions, o["num_plot_examples"])


if __name__ == "__main__":
    main()

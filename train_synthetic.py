"""Train the FOLO QKeras model on synthetic RHEED-like frames.

Script version of ``Generated_Flow.ipynb`` (everything upstream of the hls4ml
conversion). All constants live in a YAML config file:

    python train_synthetic.py configs/synthetic.yaml

Pipeline:
    1. Generate train / validation / test sets of 2D-Gaussian images and their
       soft 1/scale_factor-resolution labels.
    2. Build (or load) the quantized ``baby_yolo`` model.
    3. Train with a weighted BCE loss, early stopping and LR reduction.
    4. Save the model, the config used, the training history and a few
       image / label / prediction plots.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================================
# Config
# ============================================================================


def parse_angle(value) -> float:
    """Accept a number or a string like ``pi``, ``pi/2`` or ``2*pi``."""
    if isinstance(value, str):
        return float(eval(value, {"__builtins__": {}}, {"pi": np.pi}))
    return float(value)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    img = cfg["image"]
    if img["x"] % img["scale_factor"] or img["y"] % img["scale_factor"]:
        raise ValueError("image.x and image.y must be multiples of image.scale_factor")
    img["label_x"] = img["x"] // img["scale_factor"]
    img["label_y"] = img["y"] // img["scale_factor"]

    g = cfg["gaussians"]
    g["min_theta"] = parse_angle(g["min_theta"])
    g["max_theta"] = parse_angle(g["max_theta"])
    if g["min_num"] > g["max_num"]:
        raise ValueError("gaussians.min_num must be <= gaussians.max_num")

    return cfg


# ============================================================================
# Synthetic data generation
# ============================================================================


class SyntheticGenerator:
    """Generates images made of rotated 2D Gaussians plus soft FOLO labels."""

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.rng = rng
        self.image_x = cfg["image"]["x"]
        self.image_y = cfg["image"]["y"]
        self.scale_factor = cfg["image"]["scale_factor"]
        self.label_x = cfg["image"]["label_x"]
        self.label_y = cfg["image"]["label_y"]
        self.g = cfg["gaussians"]
        self.soft_label_sigma = cfg["labels"]["soft_label_sigma"]

        # Pixel grids reused by every gaussian_gen call
        self.X, self.Y = np.meshgrid(np.arange(self.image_x), np.arange(self.image_y))
        self.yy, self.xx = np.meshgrid(
            np.arange(self.label_y), np.arange(self.label_x), indexing="ij"
        )

    def gaussian_gen(
        self, center_x: float, center_y: float, std_x: float, std_y: float, theta: float
    ) -> np.ndarray:
        cos_theta_sqrd = np.cos(theta) ** 2
        sin_theta_sqrd = np.sin(theta) ** 2
        sin_cos_theta = np.sin(theta) * np.cos(theta)

        std_x_sqrd = std_x**2
        std_y_sqrd = std_y**2

        a = cos_theta_sqrd / (2 * std_x_sqrd) + sin_theta_sqrd / (2 * std_y_sqrd)
        b = -sin_cos_theta / (2 * std_x_sqrd) + sin_cos_theta / (2 * std_y_sqrd)
        c = sin_theta_sqrd / (2 * std_x_sqrd) + cos_theta_sqrd / (2 * std_y_sqrd)

        X, Y = self.X, self.Y
        gaussian = np.exp(
            -(
                a * (X - center_x) ** 2
                + 2 * b * (X - center_x) * (Y - center_y)
                + c * (Y - center_y) ** 2
            )
        )
        return np.expand_dims(gaussian, -1)

    def make_soft_label(self, params) -> np.ndarray:
        label = np.zeros((self.label_y, self.label_x), dtype=np.float32)
        sigma = self.soft_label_sigma

        for center_x, center_y, _std_x, _std_y, _theta, _intensity in params:
            gx = center_x / self.scale_factor
            gy = center_y / self.scale_factor
            bump = np.exp(-((self.xx - gx) ** 2 + (self.yy - gy) ** 2) / (2 * sigma**2))
            label = np.maximum(label, bump)

        return label[..., None]

    def img_gen(self) -> tuple[np.ndarray, np.ndarray, list]:
        g, rng = self.g, self.rng
        img = np.zeros(shape=(self.image_y, self.image_x, 1))
        params = []

        num_gaussians = rng.integers(low=g["min_num"], high=g["max_num"] + 1)
        for _ in range(num_gaussians):
            center_x = rng.integers(low=g["crop_x"] // 2, high=self.image_x - g["crop_x"] // 2)
            center_y = rng.integers(low=g["crop_y"] // 2, high=self.image_y - g["crop_y"] // 2)

            std_x = rng.uniform(g["min_std_x"], g["max_std_x"])
            std_y = rng.uniform(g["min_std_y"], g["max_std_y"])
            theta = rng.uniform(g["min_theta"], g["max_theta"])
            intensity = g["min_intensity"] + rng.random() * (
                g["max_intensity"] - g["min_intensity"]
            )

            params.append((center_x, center_y, std_x, std_y, theta, intensity))
            img += self.gaussian_gen(center_x, center_y, std_x, std_y, theta) * intensity

        label = self.make_soft_label(params)

        # Convert to 8 bit int
        label = (label * 255).astype(np.uint8)
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

        return img, label, params

    def gen_data(self, num_images: int, desc: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_num = self.g["max_num"]
        img_arr = np.empty((num_images, self.image_y, self.image_x, 1), dtype=np.float32)
        label_arr = np.empty((num_images, self.label_y, self.label_x, 1), dtype=np.float32)
        # Unused slots (when fewer than max_num Gaussians are drawn) are NaN.
        param_arr = np.full((num_images, max_num, 6), np.nan, dtype=np.float32)

        for i in tqdm(range(num_images), desc=desc):
            img, label, params = self.img_gen()
            img_arr[i] = img.astype(np.float32) / 256.0  # Normalize values to [0, 1)
            label_arr[i] = label.astype(np.float32) / 256.0
            param_arr[i, : len(params)] = params

        print(f"[{desc} Images Shape]: {img_arr.shape}")
        print(f"[{desc} Labels Shape]: {label_arr.shape}")
        print(f"[{desc} Params Shape]: {param_arr.shape}")

        return img_arr, label_arr, param_arr


# ============================================================================
# Model / loss
# ============================================================================


def setup_tensorflow(seed: int):
    import tensorflow as tf

    tf.random.set_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[Device]: GPU ({len(gpus)} found, CUDA build: {tf.test.is_built_with_cuda()})")
    else:
        print("[Device]: CPU (no CUDA GPU visible to TensorFlow)")

    return tf


def build_model(cfg: dict):
    from tensorflow.keras.activations import relu, sigmoid
    from tensorflow.keras.layers import Activation, Conv2D, Input
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
    for i, filters in enumerate(m["stem_filters"], start=1):
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

    n = len(m["stem_filters"])
    x = Conv2D(m["head_filters"], 1, padding="same", use_bias=True, name=f"qconv2d_{n + 1}")(x)
    x = Activation(relu, name=f"qact_{n + 1}")(x)

    x = Conv2D(1, 1, padding="same", use_bias=True, name=f"qconv2d_{n + 2}")(x)
    x_prob = Activation(sigmoid, name="sigmoid_out")(x)

    model = Model(inputs=input_layer, outputs=x_prob, name=m["name"])

    expected = (cfg["image"]["label_y"], cfg["image"]["label_x"], 1)
    if tuple(model.output_shape[1:]) != expected:
        raise ValueError(
            f"Model output shape {model.output_shape[1:]} does not match label shape "
            f"{expected}. len(model.stem_filters) must equal log2(image.scale_factor)."
        )
    return model


def make_loss_wbce(pos_weight: float, neg_weight: float):
    import tensorflow as tf

    def loss_wbce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        weights = y_true * pos_weight + (1.0 - y_true) * neg_weight
        return tf.reduce_mean(weights * bce)

    return loss_wbce


def load_model(path: str, loss_fn):
    import tensorflow as tf
    from qkeras.utils import _add_supported_quantized_objects

    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    custom_objects["loss_wbce"] = loss_fn
    print(f"[Load]: {path}")
    return tf.keras.models.load_model(path, custom_objects=custom_objects)


# ============================================================================
# Training / evaluation
# ============================================================================


def make_datasets(tf, cfg, train_img, train_label, val_img, val_label):
    batch_size = cfg["dataset"]["batch_size"]

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_img, train_label))
        .shuffle(len(train_img), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_img, val_label))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_dataset, val_dataset


def train(tf, cfg, model, loss_fn, train_dataset, val_dataset):
    t = cfg["training"]

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=t["learning_rate"],
        global_clipnorm=t["global_clipnorm"],
    )
    model.compile(optimizer=optimizer, loss=loss_fn)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=t["early_stopping"]["patience"],
        min_delta=t["early_stopping"]["min_delta"],
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=t["reduce_lr"]["factor"],
        patience=t["reduce_lr"]["patience"],
        min_lr=t["reduce_lr"]["min_lr"],
        verbose=1,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=t["epochs"],
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return history


def save_plots(out_dir: Path, test_img, test_label, predictions, num_examples: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(num_examples, len(test_img))):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(test_img[i].squeeze(), cmap="gray")
        axes[0].set_title("Test Image")
        axes[1].imshow(test_label[i].squeeze(), cmap="viridis", interpolation="none")
        axes[1].set_title("Test Label")
        axes[2].imshow(predictions[i].squeeze(), cmap="viridis", interpolation="none")
        axes[2].set_title("QK Prediction")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"example_{i}.png", dpi=120)
        plt.close(fig)
    print(f"[Plots]: {out_dir}")


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
    gen = SyntheticGenerator(cfg, rng)
    d = cfg["dataset"]
    train_img, train_label, _ = gen.gen_data(d["training_size"], desc="Train")
    val_img, val_label, _ = gen.gen_data(d["validation_size"], desc="Val")
    test_img, test_label, test_params = gen.gen_data(d["test_size"], desc="Test")

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

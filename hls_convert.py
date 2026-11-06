"""Convert a trained FOLO model to an hls4ml / Vitis HLS project.

Script version of the "HLS" cells of ``Generated_Flow.ipynb`` / ``Real_Flow.ipynb``:
the model is loaded from a directory written by one of the train_*.py scripts
instead of being defined in the notebook.

    python hls_convert.py Models/hybrid_flow_model
    python hls_convert.py Models/hybrid_flow_model --out folo_hls4ml --build

Steps:
    1. Load the QKeras model (and its config.yaml, for the test-vector generator).
    2. Save a few test frames and the QKeras predictions as <out>/tb_data/*.npy and hand
       them to hls4ml, which writes the ``tb_input_features.dat`` / ``tb_output_predictions.dat``
       the C++ testbench reads (the notebooks wrote these under Utils/ by hand).
    3. ``config_from_keras_model`` (granularity "name", backend Vitis), strategy Latency.
    4. ``convert_from_keras_model`` into <out> (io_stream, 2 ns clock, part xcku035).
    5. Optionally ``compile()`` the C++ model and compare its predictions with QKeras.
    6. Optionally ``build()`` with Vitis HLS (needs the Xilinx tools; see --xilinx).

Only steps 1-5 run without Vitis; step 6 is what the colleague with Vivado runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import yaml

from train_synthetic import SyntheticGenerator, parse_angle

# hls4ml settings used by the notebooks
BACKEND = "Vitis"
PART = "xcku035-fbva676-2-e"
CLOCK_PERIOD = 2.0
CLOCK_UNCERTAINTY = "12.5%"
IO_TYPE = "io_stream"
STRATEGY = "Latency"
PROJECT_NAME = "folo"

# Xilinx install used by the notebooks; override with --xilinx or the XILINX_* env vars
XILINX_DEFAULT = "/home/tools/Xilinx/2025/2025.2"


# ============================================================================
# Model
# ============================================================================


def load_qkeras_model(model_dir: Path):
    """Load the saved QKeras model without its training loss (not needed for conversion)."""
    import tensorflow as tf
    from qkeras.utils import _add_supported_quantized_objects

    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    print(f"[Load]: {model_dir}")
    return tf.keras.models.load_model(str(model_dir), custom_objects=custom_objects, compile=False)


def load_model_config(model_dir: Path) -> dict | None:
    """config.yaml saved next to the model by train_*.py (None if missing)."""
    path = model_dir / "config.yaml"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    img = cfg["image"]
    img["label_x"] = img["x"] // img["scale_factor"]
    img["label_y"] = img["y"] // img["scale_factor"]
    if "gaussians" in cfg:
        cfg["gaussians"]["min_theta"] = parse_angle(cfg["gaussians"]["min_theta"])
        cfg["gaussians"]["max_theta"] = parse_angle(cfg["gaussians"]["max_theta"])
    return cfg


def make_test_inputs(model, cfg: dict | None, num: int, seed: int, inputs: Path | None) -> np.ndarray:
    """Frames to compare QKeras and HLS on: a .npy file, else synthetic frames from the
    model's config, else uniform noise (same shape as the model input)."""
    if inputs is not None:
        x = np.load(inputs).astype(np.float32)
        if x.ndim == 3:
            x = x[..., None]
        print(f"[Test Inputs]: {inputs} {x.shape}")
        return x[:num]

    rng = np.random.default_rng(seed)
    if cfg is not None and "gaussians" in cfg and "labels" in cfg:
        x, _, _ = SyntheticGenerator(cfg, rng).gen_data(num, desc="Test")
        return x

    shape = (num, *model.input_shape[1:])
    print(f"[Test Inputs]: uniform noise {shape} (no synthetic section in config.yaml)")
    return rng.random(shape, dtype=np.float32)


# ============================================================================
# hls4ml
# ============================================================================


def setup_xilinx(root: str | None):
    """Export the XILINX_* variables the Vitis backend looks for (as the notebooks did)."""
    root = root or os.environ.get("XILINX_ROOT", XILINX_DEFAULT)
    os.environ.setdefault("XILINX_HLS", f"{root}/Vitis")
    os.environ.setdefault("XILINX_VITIS", f"{root}/Vitis")
    os.environ.setdefault("XILINX_VIVADO", f"{root}/Vivado")


def make_hls_config(model, reuse_factor: int | None, fifo_opt: bool) -> dict:
    import hls4ml
    from hls4ml.utils import config_from_keras_model

    config = config_from_keras_model(model, granularity="name", backend=BACKEND)
    config["Model"]["Strategy"] = STRATEGY

    if fifo_opt:
        # Fifo Depth Optimization (greatly reduces BRAM). Applied when the model graph is
        # built: hls4ml synthesizes and cosimulates the project with the tb data to size
        # the FIFOs, so this needs Vitis HLS.
        config["Flows"] = ["vitis:fifo_depth_optimization"]
        hls4ml.model.optimizer.get_optimizer("vitis:fifo_depth_optimization").configure(
            profiling_fifo_depth=1000
        )

    if reuse_factor is not None:
        config["Model"]["ReuseFactor"] = reuse_factor
        for layer_cfg in config["LayerName"].values():
            if "ReuseFactor" in layer_cfg:
                layer_cfg["ReuseFactor"] = reuse_factor

    return config


def convert(model, config: dict, out_dir: Path, tb_input: Path, tb_output: Path):
    import hls4ml

    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(out_dir),
        input_data_tb=str(tb_input),
        output_data_tb=str(tb_output),
        io_type=IO_TYPE,
        clock_period=CLOCK_PERIOD,
        clock_uncertainty=CLOCK_UNCERTAINTY,
        backend=BACKEND,
        part=PART,
        project_name=PROJECT_NAME,
    )


def write_tb_data(out_dir: Path, x: np.ndarray, y: np.ndarray) -> tuple[Path, Path]:
    """Test vectors for the C++ testbench; hls4ml flattens them into tb_data/*.dat on write."""
    tb_dir = out_dir / "tb_data"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_input, tb_output = tb_dir / "tb_input_features.npy", tb_dir / "tb_output_predictions.npy"
    np.save(tb_input, x)
    np.save(tb_output, y)
    print(f"[TB Data]: {tb_dir} ({len(x)} samples)")
    return tb_input, tb_output


def compare(y_qk: np.ndarray, y_hls: np.ndarray) -> dict:
    diff = np.abs(y_qk - y_hls)
    stats = {
        "num_samples": int(len(y_qk)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "qk_max": float(y_qk.max()),
        "hls_max": float(y_hls.max()),
    }
    # Does the HLS model put its peak in the same cell?
    qk_arg = y_qk.reshape(len(y_qk), -1).argmax(axis=1)
    hls_arg = y_hls.reshape(len(y_hls), -1).argmax(axis=1)
    stats["same_argmax_fraction"] = float((qk_arg == hls_arg).mean())
    return stats


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="Saved model directory, e.g. Models/hybrid_flow_model")
    parser.add_argument("--out", type=Path, default=None,
                        help="hls4ml project directory (default: <PROJECT_NAME>_hls4ml_<model name>)")
    parser.add_argument("--inputs", type=Path, default=None,
                        help=".npy of test frames (N, y, x[, 1]) in [0, 1); default: synthetic frames")
    parser.add_argument("--num-test", type=int, default=8, help="number of test frames (default 8)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse-factor", type=int, default=None, help="set ReuseFactor on every layer")
    parser.add_argument("--fifo-opt", action="store_true",
                        help="enable vitis:fifo_depth_optimization (runs a Vitis HLS cosim during "
                             "conversion to profile the FIFOs; needs the Xilinx tools)")
    parser.add_argument("--no-compile", action="store_true", help="only write the project; skip C++ compile/predict")
    parser.add_argument("--build", action="store_true", help="run Vitis HLS synthesis (hls_model.build)")
    parser.add_argument("--xilinx", default=None, help=f"Xilinx install root (default {XILINX_DEFAULT})")
    args = parser.parse_args()

    setup_xilinx(args.xilinx)

    model_dir = args.model_dir.resolve()
    out_dir = (args.out or Path(f"{PROJECT_NAME}_hls4ml_{model_dir.name}")).resolve()

    # ---- Model ----------------------------------------------------------
    model = load_qkeras_model(model_dir)
    model.summary()
    cfg = load_model_config(model_dir)
    if cfg is None:
        print("[Warning]: no config.yaml next to the model; test inputs will be noise")

    # ---- Test vectors ---------------------------------------------------
    x = make_test_inputs(model, cfg, args.num_test, args.seed, args.inputs)
    y_qk = model.predict(x, batch_size=len(x), verbose=0)
    tb_input, tb_output = write_tb_data(out_dir, x, y_qk)

    # ---- hls4ml config + conversion -------------------------------------
    config = make_hls_config(model, args.reuse_factor, args.fifo_opt)
    pprint(config)
    hls_model = convert(model, config, out_dir, tb_input, tb_output)
    if args.no_compile:
        hls_model.write()
    else:
        hls_model.compile()  # writes the project, then builds the C++ emulation library
    print(f"[HLS Project]: {out_dir}")

    with open(out_dir / "hls_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # ---- QKeras vs C++ emulation ----------------------------------------
    if not args.no_compile:
        y_hls = hls_model.predict(x).reshape(y_qk.shape)
        stats = compare(y_qk, y_hls)
        print("[QKeras vs HLS]:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
        with open(out_dir / "compare.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        np.savez(out_dir / "compare.npz", x=x, y_qk=y_qk, y_hls=y_hls)

    # ---- Vitis HLS build ------------------------------------------------
    if args.build:
        # Equivalent to: cd <out> && vitis_hls -f build_prj.tcl
        hls_model.build()
        try:
            import hls4ml

            hls4ml.report.read_vivado_report(str(out_dir))
        except Exception as e:  # report parsing is best-effort
            print(f"[Report]: could not read report ({e})")


if __name__ == "__main__":
    main()

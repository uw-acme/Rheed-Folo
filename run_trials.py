"""Train FOLO with N random seeds and benchmark the models on synthetic and real frames.

Each seed trains one model with a FOLO training script (default ``train_real.py``) and the
given config, then runs every model through Rheed-Eval's ``inference.py`` pipeline
(FOLO -> TopCrop -> GaussNN) on the synthetic test frames (``synthetic_ood.h5``, with
ground truth) and on a fixed set of real frames. The seeds run in parallel on an
interactive Slurm allocation (e.g. 4 nodes x 4 GPUs, several workers per GPU):

    salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node=1 --gpus-per-node=4 \\
        -A <account> --no-shell
    python run_trials.py --out <dir> --jobid <id> [--config configs/real.yaml] [--seeds 30]

Without a Slurm job (no --jobid, no $SLURM_JOB_ID) the seeds run one after another locally.
With one, --out must be on a file system the compute nodes share (e.g. $SCRATCH, not /tmp).

Writes, into --out:
  hit_rates.md                 synthetic hit rates per match radius (also printed) and the
                               best, median and worst model at each radius
  hit_rates.csv                spots found per seed, radius and synthetic frame
  topcrop_overlay_synth.pdf    TopCrop boxes of every model on each synthetic frame (+ truth)
  topcrop_overlay_real.pdf     TopCrop boxes of every model on each real frame
  models/seed_<k>/             every trained model (model, config, training log, eval
                               outputs); models/r<r>px_<best|median|worst> link to the
                               best, median and worst one at each radius
  trials.json                  arguments, seeds and the Rheed-Folo / Rheed-Eval commits

A ground-truth spot is found at radius r if a TopCrop-selected FOLO cell centre lies
within r px of it (nearest pick per spot; picks are not matched one-to-one). The hit rate
of a model is spots found / spots over all synthetic frames.

Resumable: a seed whose training and evals all finished is not rerun. Rerun the same command
after a crash or when the allocation ends. <out>/_work/ (locks, worker logs) is deleted once
the results are written.

--eval-only reruns only the evals (e.g. after a change to inference.py) on every trained
model in <out>/models/, then rewrites the hit rates, overlays and best/median/worst links.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

FOLO_DIR = Path(__file__).resolve().parent
REPO_DIR = FOLO_DIR.parent

SYNTH_FRAMES = ["sim_3", "sim_4", "sim_5", "sim_5b", "sim_7"]  # index 0 of each, n = spot count
REAL_FRAMES = ["growth_1:1000", "growth_2:1000", "growth_2:6000", "growth_6:1000", "growth_6:6000"]
REAL_N = 3  # spots TopCrop selects on the real frames
RADII = [4, 8, 12, 16, 20]
RANKS = ["best", "median", "worst"]

CELL_SIZE = 8
CROP_DIM = 48
BOX_ORIGIN_OFFS = CELL_SIZE // 2 - CROP_DIM // 2  # -20: box top-left = 8*cell + OFFS (top_crop.py)
FRAME_DIM = 300


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# One seed: train, then evaluate (runs on a worker, one GPU)
# ============================================================================


def seed_dir(args, seed: int) -> Path:
    """Run directory of a seed (trained model, config, logs, evals); kept for good."""
    return args.out / "models" / f"seed_{seed}"


def has_model(d: Path) -> bool:
    return (d / "model" / "saved_model.pb").exists()


def eval_inputs(args) -> list[tuple[str, Path, str, int, int]]:
    """(name, h5 file, dataset, index, n) for every frame a model is evaluated on."""
    synth = []
    with h5py.File(args.synth, "r") as h5:
        for name in SYNTH_FRAMES:
            synth.append((name, args.synth, name, 0, len(h5[f"{name}_params"])))
    real = []
    for spec in REAL_FRAMES:
        growth, index = spec.split(":")
        real.append((f"{growth}_{index}", args.data, growth, int(index), REAL_N))
    return synth + real


def run_seed(args, seed: int):
    # One process per seed: hold this lock until the seed is done (the training subprocess
    # inherits it). A retry launched while an earlier attempt is still alive (e.g. srun
    # reported a failure but the step kept running) waits here, then finds DONE.
    d = seed_dir(args, seed)
    lock = open(args.out / "_work" / f"seed_{seed}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if args.eval_only:
        if not has_model(d):
            raise SystemExit(f"{d}: no trained model")
    elif (d / "DONE").exists():
        return
    # SIGTERM (srun forwards it when run_trials is stopped) -> SystemExit, which makes
    # subprocess.run kill the training process instead of leaving it running
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # several workers share a GPU
    if args.gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    if "XLA_FLAGS" not in os.environ:
        # TF builds a broken libdevice path; point XLA at the pip CUDA install if there is one
        for p in sys.path:
            if (Path(p) / "nvidia/cuda_nvcc").is_dir():
                os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={Path(p) / 'nvidia/cuda_nvcc'}"
                break

    if not args.eval_only:
        if d.exists():
            shutil.rmtree(d)  # leftovers of an interrupted run
        d.mkdir(parents=True)

        # The config with this seed, absolute data path and the run directory as output
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["seed"] = seed
        if "data" in cfg:
            cfg["data"]["path"] = str(args.data)
        cfg["output"].update(save_model=True, models_dir=str(d), model_name="model")
        with open(d / "run_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        with open(d / "train.log", "w") as out:
            subprocess.run([sys.executable, str(args.train), str(d / "run_config.yaml")],
                           stdout=out, stderr=subprocess.STDOUT, cwd=FOLO_DIR, check=True,
                           pass_fds=(lock.fileno(),))

    shutil.rmtree(d / "eval", ignore_errors=True)
    evaluate(args, d / "model", d / "eval")
    if not args.eval_only:
        (d / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def evaluate(args, model_path: Path, out_dir: Path):
    """Run the Rheed-Eval inference.py pipeline on every eval frame; one npz per frame."""
    sys.path.insert(0, str(args.eval_dir))
    import inference  # Rheed-Eval/inference.py (imports TensorFlow)

    folo = inference.load_model(model_path)
    # GaussNN scaling per model dir: older inference.py versions hard-code it
    if hasattr(inference, "load_gauss"):
        gauss, gauss_scaling = inference.load_gauss(args.gauss)
        gauss_kw = {"gauss_scaling": gauss_scaling}
    else:
        gauss, gauss_kw = inference.load_model(args.gauss), {}
    # FOLO input padding: older inference.py versions always pad the frame to 320x320
    pad = inference.folo_pad(folo) if hasattr(inference, "folo_pad") else inference.PAD
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, h5_path, growth, index, n in eval_inputs(args):
        frame = inference.load_frame(h5_path, growth, index)
        r = inference.run_pipeline(frame, folo, gauss, n=n, bg_subtract=True, nms_radius=args.nms_radius,
                                   **gauss_kw)
        np.savez(out_dir / f"{name}.npz", folo_cells=r["folo_cells"], folo_scores=r["folo_scores"],
                 folo_pad=r.get("folo_pad", pad), params=r["params"])


# ============================================================================
# Scheduling (runs where the script is started, e.g. a login node)
# ============================================================================


def slurm_slots(args) -> list[tuple[str | None, int | None]]:
    """(node, gpu) per worker: every GPU of every node of the job, workers_per_gpu times."""
    if not args.jobid:
        log("No Slurm job (--jobid / $SLURM_JOB_ID): running the seeds one after another here")
        return [(None, None)]
    nodelist = subprocess.run(["squeue", "-h", "-j", args.jobid, "-o", "%N"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if not nodelist:
        raise SystemExit(f"Slurm job {args.jobid} is not running")
    nodes = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                           capture_output=True, text=True, check=True).stdout.split()
    log(f"Job {args.jobid}: {len(nodes)} node(s) x {args.gpus_per_node} GPU(s) x "
        f"{args.workers_per_gpu} worker(s)")
    return [(node, gpu) for node in nodes for gpu in range(args.gpus_per_node)
            for _ in range(args.workers_per_gpu)]


def seed_command(args, seed: int, node: str | None, gpu: int | None) -> list[str]:
    # Workers read the config copy in --out, which (unlike e.g. a login node's /tmp) they can see
    cmd = [sys.executable, str(Path(__file__).resolve()), "--out", str(args.out),
           "--config", str(args.out / "config.yaml"), "--train", str(args.train), "--data", str(args.data),
           "--synth", str(args.synth), "--gauss", str(args.gauss), "--eval-dir", str(args.eval_dir),
           "--nms-radius", str(args.nms_radius), "--only-seed", str(seed)]
    if args.eval_only:
        cmd.append("--eval-only")
    if gpu is not None:
        cmd += ["--gpu-index", str(gpu)]
    if node is None:
        return cmd
    return ["srun", f"--jobid={args.jobid}", "--overlap", "-N1", "-n1", "-w", node] + cmd


def run_all(args, seeds: list[int], todo: list[int]) -> list[int]:
    """Run the seeds in todo on the worker slots; return the seeds that failed."""
    if not todo:
        return []
    jobs = queue.Queue()
    for s in todo:
        jobs.put((s, 0))
    failed, lock, done = [], threading.Lock(), [len(seeds) - len(todo)]
    procs, stop = set(), threading.Event()

    def worker(node, gpu):
        while not stop.is_set():
            try:
                s, attempt = jobs.get_nowait()
            except queue.Empty:
                return
            with open(args.out / "_work" / f"seed_{s}.log", "a") as out:
                # Own session: a Ctrl-C reaches only this script, which then stops the workers
                p = subprocess.Popen(seed_command(args, s, node, gpu), stdout=out,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                with lock:
                    procs.add(p)
                rc = p.wait()
                with lock:
                    procs.discard(p)
            if stop.is_set():
                return
            where = f"{node or 'local'}" + (f" gpu {gpu}" if gpu is not None else "")
            with lock:
                if rc == 0 and (args.eval_only or (seed_dir(args, s) / "DONE").exists()):
                    done[0] += 1
                    log(f"seed {s} done on {where} ({done[0]}/{len(seeds)})")
                elif attempt < args.retries:
                    log(f"seed {s} failed on {where} (rc {rc}), retrying; see _work/seed_{s}.log")
                    jobs.put((s, attempt + 1))
                else:
                    log(f"seed {s} FAILED on {where} (rc {rc}); see _work/seed_{s}.log")
                    failed.append(s)

    threads = [threading.Thread(target=worker, args=slot, daemon=True) for slot in slurm_slots(args)]
    try:
        for t in threads:
            t.start()
            time.sleep(0.2)  # stagger the srun launches
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        with lock:
            running = list(procs)
        log(f"Interrupted: stopping {len(running)} running seed(s)")
        for p in running:
            p.terminate()  # srun forwards SIGTERM to its step; locally it goes to the worker
        for p in running:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        raise SystemExit("Stopped; rerun the same command to resume")
    return sorted(failed)


# ============================================================================
# Results
# ============================================================================


def load_picks(args, seeds: list[int], names: list[str]) -> dict[int, dict[str, np.ndarray]]:
    """{seed: {frame: (k, 2) selected cells (col, row)}} plus the FOLO input padding."""
    picks = {}
    for s in seeds:
        picks[s] = {}
        for name in names:
            z = np.load(seed_dir(args, s) / "eval" / f"{name}.npz")
            cells = z["folo_cells"]
            picks[s][name] = (cells[cells[:, 0] >= 0], int(z["folo_pad"]))
    return picks


def hit_counts(picks, truth, seeds, radii) -> dict[float, np.ndarray]:
    """{radius: (n_seeds, n_synth_frames) spots found}."""
    found = {r: np.zeros((len(seeds), len(SYNTH_FRAMES)), dtype=int) for r in radii}
    for i, s in enumerate(seeds):
        for j, name in enumerate(SYNTH_FRAMES):
            cells, pad = picks[s][name]
            if not len(cells):
                continue
            centres = CELL_SIZE * cells + CELL_SIZE // 2 - pad  # frame pixels (x, y)
            dist = np.linalg.norm(truth[name][:, None, :] - centres[None, :, :], axis=2).min(axis=1)
            for r in radii:
                found[r][i, j] = int((dist <= r).sum())
    return found


def select_models(found, seeds, radii) -> dict[float, dict[str, int]]:
    """Best, median and worst seed at each radius by spots found (ties: lowest seed)."""
    picked = {}
    for r in radii:
        total = dict(zip(seeds, found[r].sum(axis=1)))
        ranked = sorted(seeds, key=lambda s: (total[s], s))
        picked[r] = {
            "best": min(seeds, key=lambda s: (-total[s], s)),
            "median": ranked[(len(seeds) - 1) // 2],
            "worst": ranked[0],
        }
    return picked


def hit_rate_table(found, seeds, radii, n_spots) -> list[str]:
    total = sum(n_spots.values())
    lines = [
        f"| radius | mean ± std | median | min | max | all found | "
        + " | ".join(SYNTH_FRAMES) + " |",
        "|---|---|---|---|---|---|" + "---|" * len(SYNTH_FRAMES),
    ]
    for r in radii:
        f = found[r]
        h = f.sum(axis=1) / total
        per_frame = f.mean(axis=0) / np.array([n_spots[n] for n in SYNTH_FRAMES])
        std = 100 * h.std(ddof=1) if len(h) > 1 else 0.0
        lines.append(
            f"| {r:g} px | {100 * h.mean():.1f} ± {std:.1f}% | {100 * np.median(h):.1f}% "
            f"| {100 * h.min():.1f}% | {100 * h.max():.1f}% | {100 * (h == 1).mean():.0f}% | "
            + " | ".join(f"{100 * p:.0f}%" for p in per_frame) + " |"
        )
    return lines


def plot_overlays(path: Path, frames, picks, seeds, truth, title, alpha):
    """One page per frame: TopCrop boxes of every model, plus the true spot centres if known."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    with PdfPages(path) as pdf:
        for name, frame in frames.items():
            fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
            im = ax.imshow(frame, cmap="viridis", vmin=0, vmax=frame.max(), interpolation="none")
            for s in seeds:
                cells, pad = picks[s][name]
                for x0, y0 in CELL_SIZE * cells + BOX_ORIGIN_OFFS - pad:
                    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), CROP_DIM, CROP_DIM, fill=False,
                                           ec="yellow", lw=1, alpha=alpha))
            if name in truth:
                ax.scatter(truth[name][:, 0], truth[name][:, 1], marker="+", c="red", s=40, lw=1,
                           label="ground truth")
                ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
            ax.set_xlim(-0.5, FRAME_DIM - 0.5)
            ax.set_ylim(FRAME_DIM - 0.5, -0.5)
            ax.set_title(f"{title}\n{name}: TopCrop boxes of {len(seeds)} models", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.8, label="counts")
            pdf.savefig(fig)
            plt.close(fig)


def git_commit(path: Path) -> str:
    r = subprocess.run(["git", "-C", str(path), "describe", "--always", "--dirty", "--abbrev=12"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def write_results(args, seeds: list[int]):
    inputs = eval_inputs(args)
    real_names = [name for name, *_ in inputs if name not in SYNTH_FRAMES]
    with h5py.File(args.synth, "r") as h5:
        truth = {n: h5[f"{n}_params"][:, :2] for n in SYNTH_FRAMES}  # (mu_x, mu_y) = (col, row)
        synth_frames = {n: h5[n][0] for n in SYNTH_FRAMES}
    real_frames = {}
    with h5py.File(args.data, "r") as h5:
        for name, _, growth, index, _ in inputs:
            if name in real_names:
                real_frames[name] = h5[growth][index]
    n_spots = {n: len(truth[n]) for n in SYNTH_FRAMES}

    picks = load_picks(args, seeds, SYNTH_FRAMES + real_names)
    found = hit_counts(picks, truth, seeds, args.radii)
    picked = select_models(found, seeds, args.radii)
    title = f"{args.train.name} {args.config.name} @ {git_commit(FOLO_DIR)}"

    # Hit rates
    table = hit_rate_table(found, seeds, args.radii, n_spots)
    with open(args.out / "hit_rates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "radius_px", *SYNTH_FRAMES, "found", "spots"])
        for r in args.radii:
            for s, row in zip(seeds, found[r]):
                w.writerow([s, f"{r:g}", *row, row.sum(), sum(n_spots.values())])

    # Best / median / worst model at each radius: links into models/ (a seed may hold several)
    models = args.out / "models"
    for r in args.radii:
        for rank, s in picked[r].items():
            link = models / f"r{r:g}px_{rank}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(f"seed_{s}")

    total = sum(n_spots.values())
    rows = [f"| {r:g} px | " + " | ".join(
        f"seed {picked[r][k]} ({found[r][seeds.index(picked[r][k])].sum()}/{total})" for k in RANKS
    ) + " |" for r in args.radii]
    md = [
        "# FOLO synthetic hit rates",
        "",
        f"`{title}`; {len(seeds)} seeds ({seeds[0]}..{seeds[-1]}); Rheed-Eval {git_commit(args.eval_dir)}.",
        f"{total} ground-truth spots over "
        + ", ".join(f"{n} ({n_spots[n]})" for n in SYNTH_FRAMES)
        + ". A spot is found at radius r if a TopCrop-selected FOLO cell centre is within r px "
        "(nearest pick per spot). Hit rate = spots found / spots, per model; stats over models.",
        "",
        *table,
        "",
        "## Best, median and worst model per radius (`models/r<r>px_<rank>`)",
        "",
        "| radius | " + " | ".join(RANKS) + " |",
        "|---|---|---|---|",
        *rows,
        "",
    ]
    (args.out / "hit_rates.md").write_text("\n".join(md))

    plot_overlays(args.out / "topcrop_overlay_synth.pdf", synth_frames, picks, seeds, truth, title, args.alpha)
    plot_overlays(args.out / "topcrop_overlay_real.pdf", real_frames, picks, seeds, {}, title, args.alpha)

    summary = {
        "complete": True,
        "seeds": seeds,
        "best_median_worst": {f"{r:g}": picked[r] for r in args.radii},
        "rheed_folo": git_commit(FOLO_DIR),
        "rheed_eval": git_commit(args.eval_dir),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.out / "trials.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(table))
    print()
    print("\n".join(md[md.index("## Best, median and worst model per radius (`models/r<r>px_<rank>`)"):]))


# ============================================================================
# Main
# ============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output directory (also holds the resumable work)")
    p.add_argument("--config", type=Path, default=FOLO_DIR / "configs/real.yaml")
    p.add_argument("--train", type=Path, default=FOLO_DIR / "train_real.py", help="FOLO training script")
    p.add_argument("--seeds", type=int, default=30, help="number of seeds")
    p.add_argument("--first-seed", type=int, default=0, help="seeds are first_seed .. first_seed + seeds - 1")
    p.add_argument("--jobid", default=os.environ.get("SLURM_JOB_ID"), help="Slurm allocation to run on")
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--workers-per-gpu", type=int, default=3)
    p.add_argument("--retries", type=int, default=1, help="reruns of a failed seed before giving up")
    p.add_argument("--data", type=Path, default=REPO_DIR / "data/STO_STO_test6_06292022-standard-compressed.h5",
                   help="real RHEED h5 (training data and real eval frames)")
    p.add_argument("--synth", type=Path, default=REPO_DIR / "Rheed-Eval/synthetic_RHEED/synthetic_ood.h5")
    p.add_argument("--gauss", type=Path, default=REPO_DIR / "Rheed-Gaussian/Models/gen_flow_model")
    p.add_argument("--eval-dir", type=Path, default=REPO_DIR / "Rheed-Eval", help="directory of inference.py")
    p.add_argument("--nms-radius", type=float, default=6, help="TopCrop exclusion radius in FOLO grid cells")
    p.add_argument("--radii", type=float, nargs="+", default=RADII, help="match radii (px)")
    p.add_argument("--alpha", type=float, default=0.15, help="opacity of each box in the overlays")
    p.add_argument("--force", action="store_true", help="rerun even if --out already holds finished results")
    p.add_argument("--eval-only", action="store_true",
                   help="rerun only the evals on the trained models already in --out (see above)")
    p.add_argument("--only-seed", type=int, help=argparse.SUPPRESS)  # worker: run one seed here
    p.add_argument("--gpu-index", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    for k in ("out", "config", "train", "data", "synth", "gauss", "eval_dir"):
        setattr(args, k, getattr(args, k).resolve())

    if args.only_seed is not None:
        run_seed(args, args.only_seed)
        return

    if args.jobid and str(args.out).startswith("/tmp"):
        raise SystemExit("--out must be on a file system the compute nodes share (not /tmp)")
    work = args.out / "_work"
    if args.eval_only:
        eval_only(args)
        return
    if (args.out / "trials.json").exists() and not work.exists() and not args.force:
        print((args.out / "hit_rates.md").read_text())
        log(f"{args.out} already holds finished results; pass --force to rerun")
        return
    if args.force and (args.out / "trials.json").exists():  # start over, dropping old results
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(args.out / "models", ignore_errors=True)
        (args.out / "trials.json").unlink()

    # A resumed run must use the same training setup as the runs already in _work
    work.mkdir(parents=True, exist_ok=True)
    setup = {"config": args.config.read_text(), "train": str(args.train), "data": str(args.data),
             "synth": str(args.synth), "gauss": str(args.gauss), "eval_dir": str(args.eval_dir),
             "nms_radius": args.nms_radius}
    setup_file = work / "setup.json"
    if setup_file.exists() and json.loads(setup_file.read_text()) != setup:
        raise SystemExit(f"{work} holds runs with a different config or setup; "
                         "use another --out or delete it")
    setup_file.write_text(json.dumps(setup, indent=2))
    if args.config != args.out / "config.yaml":
        shutil.copy(args.config, args.out / "config.yaml")

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    todo = [s for s in seeds if not (seed_dir(args, s) / "DONE").exists()]
    log(f"{len(seeds) - len(todo)}/{len(seeds)} seeds already done, {len(todo)} to run")
    failed = run_all(args, seeds, todo)
    if failed:
        raise SystemExit(f"Seeds {failed} failed; logs in {work}. Rerun the same command to retry them.")

    write_results(args, seeds)
    finish(args)


def finish(args):
    """Drop the run bookkeeping: <out>/_work and the per-seed DONE markers."""
    shutil.rmtree(args.out / "_work")
    for done in (args.out / "models").glob("seed_*/DONE"):
        done.unlink()
    log(f"Results in {args.out}")


def eval_only(args):
    """Rerun the evals on every trained model in <out>/models."""
    if args.force:
        raise SystemExit("--eval-only and --force don't go together (--force drops the models)")
    work = args.out / "_work"
    seeds = sorted(int(d.name.split("_")[1]) for d in (args.out / "models").glob("seed_*")
                   if d.is_dir() and has_model(d))
    if not seeds:
        raise SystemExit(f"No trained models in {args.out / 'models'}")
    # Name the training run in the results by its own script and config, not this call's defaults
    prev = work / "setup.json" if (work / "setup.json").exists() else args.out / "trials.json"
    if prev.exists():
        p = json.loads(prev.read_text())
        args.train = Path(p.get("train") or p["args"]["train"])
        args.config = Path(p.get("args", {}).get("config", args.out / "config.yaml"))
    log(f"Re-evaluating {len(seeds)} model(s) in {args.out / 'models'}")
    work.mkdir(parents=True, exist_ok=True)
    failed = run_all(args, seeds, seeds)
    if failed:
        raise SystemExit(f"Seeds {failed} failed; logs in {work}. Rerun the same command to retry them.")

    write_results(args, seeds)
    finish(args)


if __name__ == "__main__":
    main()

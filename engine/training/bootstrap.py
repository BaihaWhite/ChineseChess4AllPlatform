#!/usr/bin/env python3
"""NNUE Self-Play Bootstrapping Orchestrator.

Each iteration:
  1. Generate self-play data using current NNUE at deep search
  2. Mix new data with original HCE data (prevents catastrophic forgetting)
  3. Precompute flat features
  4. Fine-tune model from checkpoint
  5. Run match_runner evaluation
  6. Compare with previous best, save if improved

Usage:
  # First run: start from baseline HCE-trained model
  python bootstrap.py --baseline-checkpoint nnue_trained.pth \
      --baseline-bin nnue_trained.bin --hce-data data/all_positions.txt

  # Resume from previous state
  python bootstrap.py

Config is read from bootstrap_config.json if present, overridden by CLI args.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
STATE_FILE = SCRIPT_DIR / "bootstrap_state.json"
CONFIG_FILE = SCRIPT_DIR / "bootstrap_config.json"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "bootstrap_depth": 8,
    "games_per_iter": 500,
    "finetune_epochs": 15,
    "finetune_lr": 0.0001,
    "finetune_batch_size": 4096,
    "hce_data_txt": str(DATA_DIR / "all_positions.txt"),
    "baseline_nnue_bin": str(SCRIPT_DIR / "nnue_trained.bin"),
    "baseline_checkpoint_pth": str(SCRIPT_DIR / "nnue_trained.pth"),
    "active_nnue_bin": str(SCRIPT_DIR / "nnue_active.bin"),
    "active_checkpoint_pth": str(SCRIPT_DIR / "checkpoint_active.pth"),
    "cargo_release": True,
    "data_mix_ratio": 0.7,   # fraction of new bootstrap data
    "max_iterations": 10,
    "eval_nnue_depth": 2,    # NNUE depth for evaluation (deeper shows eval quality better)
    "eval_hce_max": 4,       # max HCE depth to test against
    "eval_games": 30,        # games per depth pair
    "gen_data_threads": 4,   # CPU threads for gen_data (rayon)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cmd(cmd, desc, timeout=None, env=None):
    """Run a command, print output in real-time, return exit code."""
    print(f"\n  [{desc}]")
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ENGINE_DIR), timeout=timeout,
                            capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if result.stdout:
        # Print last few lines
        lines = result.stdout.strip().split('\n')
        for line in lines[-30:]:
            print(f"    {line}")
    if result.stderr and result.returncode != 0:
        print(f"  STDERR: {result.stderr[:2000]}")
    print(f"  [{desc}] done in {dt:.1f}s (exit={result.returncode})")
    return result.returncode == 0


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "iteration": 0,
        "best_equivalent_depth": 0,
        "best_checkpoint": None,
        "best_checkpoint_bin": None,
        "history": [],
    }


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def gen_data(nnue_bin, output_txt, num_games, depth, cargo_release, threads=None):
    """Generate self-play data with NNUE at given depth."""
    profile = "release" if cargo_release else "debug"
    exe = ENGINE_DIR / "target" / profile / "gen_data"
    if not exe.exists():
        print(f"  Building gen_data ({profile})...")
        ok = run_cmd(["cargo", "build", "--bin", "gen_data"]
                     + (["--release"] if cargo_release else []),
                     "build gen_data")
        if not ok:
            return False
    env = os.environ.copy()
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    return run_cmd([str(exe), output_txt, str(num_games), str(depth), nnue_bin],
                   f"gen_data depth={depth} games={num_games} (threads={threads or 'auto'})",
                   env=env)


def precompute(input_txt, output_pt):
    """Convert FEN+score text file to flat features .pt file."""
    script = SCRIPT_DIR / "precompute_flat.py"
    venv_python = SCRIPT_DIR / "venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    return run_cmd([python, str(script), "--data", input_txt, "--output", output_pt],
                   f"precompute {input_txt} -> {output_pt}")


def mix_data(bootstrap_txt, hce_txt, output_txt, ratio):
    """Mix bootstrap data with HCE data at given ratio.
    ratio = fraction of lines from bootstrap_txt.
    Lines are interleaved to avoid ordering bias."""
    print(f"\n  [mix_data] ratio={ratio:.0%} bootstrap + {1-ratio:.0%} HCE")

    with open(bootstrap_txt) as f:
        bootstrap_lines = [l for l in f if l.strip() and not l.startswith('#')]
    with open(hce_txt) as f:
        hce_lines = [l for l in f if l.strip() and not l.startswith('#')]

    n_bootstrap = len(bootstrap_lines)
    n_hce = len(hce_lines)
    # Take a subset of HCE lines to match the ratio
    target_hce = int(n_bootstrap * (1 - ratio) / ratio) if ratio > 0 else n_hce
    target_hce = min(target_hce, n_hce)
    import random
    random.seed(42)
    hce_sample = random.sample(hce_lines, target_hce)

    with open(output_txt, 'w') as f:
        # Interleave
        step_b = max(1, n_bootstrap // max(1, target_hce))
        step_h = max(1, target_hce // max(1, n_bootstrap))
        bi, hi = 0, 0
        while bi < n_bootstrap or hi < target_hce:
            for _ in range(step_b):
                if bi < n_bootstrap:
                    f.write(bootstrap_lines[bi])
                    bi += 1
            for _ in range(step_h):
                if hi < target_hce:
                    f.write(hce_sample[hi])
                    hi += 1

    total = n_bootstrap + target_hce
    print(f"  Mixed: {n_bootstrap} bootstrap + {target_hce} HCE = {total} total positions")
    return True


def train_model(features_pt, checkpoint_pth, output_bin, output_pth,
                epochs, lr, batch_size, freeze_l1):
    """Fine-tune model from checkpoint."""
    script = SCRIPT_DIR / "train_nnue.py"
    venv_python = SCRIPT_DIR / "venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [
        python, str(script),
        "--features", features_pt,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--output", output_bin,
        "--save-checkpoint", output_pth,
    ]
    if freeze_l1:
        cmd += ["--warmup-epochs", "0", "--freeze-l1"]
    if checkpoint_pth:
        cmd += ["--checkpoint", checkpoint_pth]

    return run_cmd(cmd, f"train (epochs={epochs}, lr={lr}, freeze_l1={freeze_l1})")


def evaluate(nnue_bin, nnue_depth, hce_max, games, cargo_release):
    """Run match_runner to find equivalent depth."""
    profile = "release" if cargo_release else "debug"
    exe = ENGINE_DIR / "target" / profile / "match_runner"
    if not exe.exists():
        print(f"  Building match_runner ({profile})...")
        ok = run_cmd(["cargo", "build", "--bin", "match_runner"]
                     + (["--release"] if cargo_release else []),
                     "build match_runner")
        if not ok:
            return None

    ok = run_cmd([str(exe), str(nnue_depth), "1", str(hce_max), str(games), nnue_bin],
                 f"evaluate nnue_depth={nnue_depth} vs HCE 1-{hce_max} ({games} games each)")
    # Parse output for equivalent depth
    if not ok:
        return None

    # The output is captured in run_cmd. We need to re-run to parse it.
    # Instead, run again with output parsing.
    result = subprocess.run(
        [str(exe), str(nnue_depth), "1", str(hce_max), str(games), nnue_bin],
        cwd=str(ENGINE_DIR), capture_output=True, text=True, timeout=600)
    output = result.stdout

    # Parse win rates per depth
    import re
    depths = {}
    for line in output.split('\n'):
        m = re.match(r'HCE d=(\d+):\s+(\d+)%\s+\[Elo\s+([+-]?\d+)\]', line)
        if m:
            d = int(m.group(1))
            wr = int(m.group(2)) / 100.0
            elo = int(m.group(3))
            depths[d] = {"win_rate": wr, "elo": elo}

    # Find equivalent depth
    eq_depth = None
    for d in sorted(depths.keys()):
        wr = depths[d]["win_rate"]
        if abs(wr - 0.5) < 0.08:
            eq_depth = d
            break

    # Also check "Equivalent depth" line
    for line in output.split('\n'):
        if 'Equivalent depth:' in line:
            print(f"    {line.strip()}")

    if eq_depth is None and depths:
        # Find depth with wr closest to 0.5
        best_d = min(depths.keys(), key=lambda d: abs(depths[d]["win_rate"] - 0.5))
        eq_depth = best_d

    return {
        "eq_depth": eq_depth,
        "depths": depths,
    }


def run_iteration(config, state):
    """Run one bootstrapping iteration."""
    it = state["iteration"] + 1
    print(f"\n{'='*60}")
    print(f"  Bootstrapping Iteration {it}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    # Determine which checkpoint to use
    base_checkpoint = state.get("best_checkpoint") or config.get("baseline_checkpoint_pth")
    base_bin = state.get("best_checkpoint_bin") or config.get("baseline_nnue_bin")

    # Resolve relative paths relative to SCRIPT_DIR (training/)
    base_checkpoint = str(Path(base_checkpoint).resolve() if Path(base_checkpoint).is_absolute() else (SCRIPT_DIR / base_checkpoint).resolve())
    base_bin = str(Path(base_bin).resolve() if Path(base_bin).is_absolute() else (SCRIPT_DIR / base_bin).resolve())

    if not base_bin or not Path(base_bin).exists():
        print(f"  ERROR: NNUE binary not found: {base_bin}")
        print(f"  Train a baseline model first or provide --baseline-bin")
        return state

    print(f"  Using checkpoint: {base_checkpoint}")
    print(f"  Using NNUE bin:   {base_bin}")

    # Step 1: Generate bootstrapping data
    bootstrap_txt = str(DATA_DIR / f"bootstrap_iter{it}.txt")
    ok = gen_data(
        base_bin, bootstrap_txt,
        config["games_per_iter"], config["bootstrap_depth"],
        config.get("cargo_release", True),
        config.get("gen_data_threads", None))
    if not ok:
        print("  Failed to generate data, skipping iteration")
        return state

    # Step 2: Mix with HCE data
    mixed_txt = str(DATA_DIR / f"mixed_iter{it}.txt")
    hce_data = config.get("hce_data_txt", DEFAULTS["hce_data_txt"])
    ratio = config.get("data_mix_ratio", 0.7)
    if Path(hce_data).exists() and ratio > 0:
        mix_data(bootstrap_txt, hce_data, mixed_txt, ratio)
        data_txt = mixed_txt
    else:
        data_txt = bootstrap_txt
        print(f"  No HCE data to mix, using bootstrap data only")

    # Step 3: Precompute features
    features_pt = str(DATA_DIR / f"features_iter{it}.pt")
    ok = precompute(data_txt, features_pt)
    if not ok:
        print("  Failed to precompute features, skipping iteration")
        return state

    # Step 4: Fine-tune
    output_bin = str(SCRIPT_DIR / f"nnue_iter{it}.bin")
    output_pth = str(SCRIPT_DIR / f"checkpoint_iter{it}.pth")
    ok = train_model(
        features_pt, base_checkpoint, output_bin, output_pth,
        config["finetune_epochs"], config["finetune_lr"],
        config.get("finetune_batch_size", 4096),
        freeze_l1=config.get("freeze_l1", False))
    if not ok:
        print("  Training failed, skipping iteration")
        return state

    # Step 5: Evaluate
    print(f"\n  Evaluating model...")
    result = evaluate(
        output_bin, config.get("eval_nnue_depth", 2),
        config.get("eval_hce_max", 4),
        config.get("eval_games", 30),
        config.get("cargo_release", True))

    if result is None:
        print("  Evaluation failed")
        return state

    eq_depth = result.get("eq_depth")
    print(f"\n  Iteration {it} equivalent depth: {eq_depth}")

    # Step 6: Compare and maybe save
    entry = {
        "iteration": it,
        "eq_depth": eq_depth,
        "depths": result.get("depths", {}),
        "checkpoint_bin": output_bin,
        "checkpoint_pth": output_pth,
        "timestamp": datetime.now().isoformat(),
    }
    state["history"].append(entry)

    prev_best = state.get("best_equivalent_depth", 0)
    if eq_depth and eq_depth > prev_best:
        print(f"  IMPROVED! eq_depth: {prev_best} -> {eq_depth}")
        state["best_equivalent_depth"] = eq_depth
        state["best_checkpoint"] = output_pth
        state["best_checkpoint_bin"] = output_bin
        # Copy to active
        active_bin = config.get("active_nnue_bin", DEFAULTS["active_nnue_bin"])
        active_pth = config.get("active_checkpoint_pth", DEFAULTS["active_checkpoint_pth"])
        shutil.copy(output_bin, active_bin)
        shutil.copy(output_pth, active_pth)
        print(f"  Copied to {active_bin} and {active_pth}")
    elif eq_depth and eq_depth == prev_best:
        print(f"  Same as previous best ({prev_best}) — keeping previous checkpoint")
    else:
        print(f"  No improvement ({eq_depth} <= {prev_best}) — keeping previous checkpoint")

    state["iteration"] = it
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NNUE Self-Play Bootstrapping Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
First run example (start from baseline HCE-trained model):
  python bootstrap.py --baseline-checkpoint nnue_trained.pth \\
      --baseline-bin nnue_trained.bin --hce-data data/all_positions.txt \\
      --iterations 5

Resume from previous state:
  python bootstrap.py
""")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of bootstrapping iterations to run")
    parser.add_argument("--bootstrap-depth", type=int, default=8,
                        help="Search depth for self-play data generation")
    parser.add_argument("--games-per-iter", type=int, default=500,
                        help="Number of self-play games per iteration")
    parser.add_argument("--finetune-epochs", type=int, default=15,
                        help="Fine-tuning epochs per iteration")
    parser.add_argument("--finetune-lr", type=float, default=0.0001,
                        help="Fine-tuning learning rate")
    parser.add_argument("--data-mix-ratio", type=float, default=0.7,
                        help="Fraction of bootstrap data in mix (rest is HCE)")
    parser.add_argument("--hce-data", default=None,
                        help="Path to HCE FEN+score text file for data mixing")
    parser.add_argument("--baseline-checkpoint", default=None,
                        help="Initial .pth checkpoint (first iteration)")
    parser.add_argument("--baseline-bin", default=None,
                        help="Initial .bin NNUE weights (first iteration)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without running")
    args = parser.parse_args()

    # Build config from defaults, file, and CLI
    config = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config.update(json.load(f))
    for key, val in vars(args).items():
        if val is not None and key in config:
            config[key] = val

    # Special: --hce-data and --baseline-* override from CLI
    if args.hce_data:
        config["hce_data_txt"] = args.hce_data
    if args.baseline_checkpoint:
        config["baseline_checkpoint_pth"] = args.baseline_checkpoint
    if args.baseline_bin:
        config["baseline_nnue_bin"] = args.baseline_bin

    print("Bootstrapping Config:")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")

    if args.dry_run:
        return

    # Load or init state
    state = load_state()
    print(f"\nCurrent state: iteration={state['iteration']}, "
          f"best_eq_depth={state.get('best_equivalent_depth')}")

    # Run iterations
    for _ in range(args.iterations):
        state = run_iteration(config, state)
        save_state(state)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Bootstrapping Complete")
    print(f"{'='*60}")
    print(f"  Iterations: {state['iteration']}")
    print(f"  Best equivalent depth: {state.get('best_equivalent_depth')}")
    if state.get("best_checkpoint_bin"):
        print(f"  Best checkpoint bin: {state['best_checkpoint_bin']}")
    print(f"\n  History:")
    for entry in state.get("history", []):
        print(f"    Iter {entry['iteration']}: eq_depth={entry.get('eq_depth')}")
    print()


if __name__ == "__main__":
    main()

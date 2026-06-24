#!/usr/bin/env python3
"""AlphaZero-style NNUE Reinforcement Learning Orchestrator.

Each iteration:
  1. Generate self-play data with game outcomes (3-column: FEN score result)
  2. Precompute flat features (includes game_results)
  3. Train with dual loss: MSE(score) + lambda * MSE(value)
  4. Evaluate with match_runner
  5. Save if improved

Pure self-play RL — no HCE data mixing.

Usage:
  python rl_train.py --baseline-checkpoint nnue_trained.pth \
      --baseline-bin nnue_trained.bin
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
STATE_FILE = SCRIPT_DIR / "rl_state.json"
CONFIG_FILE = SCRIPT_DIR / "rl_config.json"

DEFAULTS = {
    "bootstrap_depth": 8,
    "games_per_iter": 2000,
    "finetune_epochs": 50,
    "finetune_lr": 0.0002,
    "finetune_batch_size": 4096,
    "lambda_value": 1.0,
    "baseline_nnue_bin": str(SCRIPT_DIR / "nnue_trained.bin"),
    "baseline_checkpoint_pth": str(SCRIPT_DIR / "nnue_trained.pth"),
    "active_nnue_bin": str(SCRIPT_DIR / "nnue_active.bin"),
    "active_checkpoint_pth": str(SCRIPT_DIR / "checkpoint_active.pth"),
    "cargo_release": True,
    "max_iterations": 10,
    "eval_nnue_depth": 2,
    "eval_hce_max": 8,
    "eval_games": 30,
    "gen_data_threads": 4,
    "rl_weak_depth": 1,  # weak side search depth (depth 1 = random-like, ensures decisive games)
    "freeze_l1": False,
    "warmup_epochs": 0,
}


def run_cmd(cmd, desc, timeout=None, env=None):
    print(f"\n  [{desc}]")
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ENGINE_DIR), timeout=timeout,
                            capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if result.stdout:
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


def gen_data(nnue_bin, output_txt, num_games, depth, cargo_release, threads=None, weak_depth=0):
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
    cmd = [str(exe), output_txt, str(num_games), str(depth), nnue_bin]
    if weak_depth > 0 and weak_depth < depth:
        cmd.append(str(weak_depth))
    return run_cmd(cmd,
                   f"gen_data depth={depth} weak={weak_depth} games={num_games} (threads={threads or 'auto'})",
                   env=env)


def precompute(input_txt, output_pt):
    script = SCRIPT_DIR / "precompute_flat.py"
    venv_python = SCRIPT_DIR / "venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    return run_cmd([python, str(script), "--data", input_txt, "--output", output_pt],
                   f"precompute {input_txt} -> {output_pt}")


def train_model(features_pt, checkpoint_pth, output_bin, output_pth,
                epochs, lr, batch_size, lambda_value, freeze_l1, warmup_epochs):
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
        "--lambda-value", str(lambda_value),
        "--warmup-epochs", str(warmup_epochs),
    ]
    if freeze_l1:
        cmd += ["--freeze-l1"]
    if checkpoint_pth:
        cmd += ["--checkpoint", checkpoint_pth]

    return run_cmd(cmd, f"train (epochs={epochs}, lr={lr}, lambda_value={lambda_value})")


def evaluate(nnue_bin, nnue_depth, hce_max, games, cargo_release):
    profile = "release" if cargo_release else "debug"
    exe = ENGINE_DIR / "target" / profile / "match_runner"
    if not exe.exists():
        print(f"  Building match_runner ({profile})...")
        ok = run_cmd(["cargo", "build", "--bin", "match_runner"]
                     + (["--release"] if cargo_release else []),
                     "build match_runner")
        if not ok:
            return None

    result = subprocess.run(
        [str(exe), str(nnue_depth), "1", str(hce_max), str(games), nnue_bin],
        cwd=str(ENGINE_DIR), capture_output=True, text=True, timeout=600)
    output = result.stdout

    import re
    depths = {}
    for line in output.split('\n'):
        m = re.match(r'HCE d=(\d+):\s+(\d+)%\s+\[Elo\s+([+-]?\d+)\]', line)
        if m:
            d = int(m.group(1))
            wr = int(m.group(2)) / 100.0
            elo = int(m.group(3))
            depths[d] = {"win_rate": wr, "elo": elo}

    eq_depth = None
    for d in sorted(depths.keys()):
        wr = depths[d]["win_rate"]
        if abs(wr - 0.5) < 0.08:
            eq_depth = d
            break

    for line in output.split('\n'):
        if 'Equivalent depth:' in line:
            print(f"    {line.strip()}")

    if eq_depth is None and depths:
        best_d = min(depths.keys(), key=lambda d: abs(depths[d]["win_rate"] - 0.5))
        eq_depth = best_d

    return {"eq_depth": eq_depth, "depths": depths}


def run_iteration(config, state):
    it = state["iteration"] + 1
    print(f"\n{'='*60}")
    print(f"  RL Iteration {it}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    base_checkpoint = state.get("best_checkpoint") or config.get("baseline_checkpoint_pth")
    base_bin = state.get("best_checkpoint_bin") or config.get("baseline_nnue_bin")

    base_checkpoint = str(Path(base_checkpoint).resolve() if Path(base_checkpoint).is_absolute()
                          else (SCRIPT_DIR / base_checkpoint).resolve())
    base_bin = str(Path(base_bin).resolve() if Path(base_bin).is_absolute()
                   else (SCRIPT_DIR / base_bin).resolve())

    if not base_bin or not Path(base_bin).exists():
        print(f"  ERROR: NNUE binary not found: {base_bin}")
        return state

    print(f"  Using checkpoint: {base_checkpoint}")
    print(f"  Using NNUE bin:   {base_bin}")

    # Step 1: Generate self-play data with game outcomes
    data_txt = str(DATA_DIR / f"rl_iter{it}.txt")
    ok = gen_data(
        base_bin, data_txt,
        config["games_per_iter"], config["bootstrap_depth"],
        config.get("cargo_release", True),
        config.get("gen_data_threads", None),
        config.get("rl_weak_depth", 1))
    if not ok:
        print("  Failed to generate data, skipping iteration")
        return state

    # Step 2: Precompute features (includes game_results)
    features_pt = str(DATA_DIR / f"rl_features_iter{it}.pt")
    ok = precompute(data_txt, features_pt)
    if not ok:
        print("  Failed to precompute features, skipping iteration")
        return state

    # Step 3: Train with dual loss
    output_bin = str(SCRIPT_DIR / f"nnue_rl_iter{it}.bin")
    output_pth = str(SCRIPT_DIR / f"checkpoint_rl_iter{it}.pth")
    ok = train_model(
        features_pt, base_checkpoint, output_bin, output_pth,
        config["finetune_epochs"], config["finetune_lr"],
        config.get("finetune_batch_size", 4096),
        config.get("lambda_value", 1.0),
        config.get("freeze_l1", False),
        config.get("warmup_epochs", 0))
    if not ok:
        print("  Training failed, skipping iteration")
        return state

    # Step 4: Evaluate
    print(f"\n  Evaluating model...")
    result = evaluate(
        output_bin, config.get("eval_nnue_depth", 2),
        config.get("eval_hce_max", 8),
        config.get("eval_games", 30),
        config.get("cargo_release", True))

    if result is None:
        print("  Evaluation failed")
        return state

    eq_depth = result.get("eq_depth")
    print(f"\n  Iteration {it} equivalent depth: {eq_depth}")

    # Step 5: Compare and save
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


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero-Style NNUE RL Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python rl_train.py --baseline-checkpoint nnue_trained.pth \\
      --baseline-bin nnue_trained.bin --iterations 5
""")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of RL iterations")
    parser.add_argument("--bootstrap-depth", type=int, default=8,
                        help="Search depth for self-play")
    parser.add_argument("--games-per-iter", type=int, default=2000,
                        help="Self-play games per iteration")
    parser.add_argument("--finetune-epochs", type=int, default=50,
                        help="Training epochs per iteration")
    parser.add_argument("--finetune-lr", type=float, default=0.0002)
    parser.add_argument("--finetune-batch-size", type=int, default=8192,
                        help="Training batch size")
    parser.add_argument("--lambda-value", type=float, default=1.0,
                        help="Value head loss weight")
    parser.add_argument("--gen-data-threads", type=int, default=4,
                        help="CPU threads for gen_data")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--baseline-bin", default=None)
    parser.add_argument("--freeze-l1", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config.update(json.load(f))
    for key, val in vars(args).items():
        if val is not None and key in config:
            config[key] = val

    if args.baseline_checkpoint:
        config["baseline_checkpoint_pth"] = args.baseline_checkpoint
    if args.baseline_bin:
        config["baseline_nnue_bin"] = args.baseline_bin
    if args.gen_data_threads:
        config["gen_data_threads"] = args.gen_data_threads

    print("RL Training Config:")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")

    if args.dry_run:
        return

    state = load_state()
    print(f"\nCurrent state: iteration={state['iteration']}, "
          f"best_eq_depth={state.get('best_equivalent_depth')}")

    for _ in range(args.iterations):
        state = run_iteration(config, state)
        save_state(state)

    print(f"\n{'='*60}")
    print(f"  RL Training Complete")
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

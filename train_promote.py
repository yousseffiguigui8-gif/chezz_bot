import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import evaluate_models


def parse_args():
    parser = argparse.ArgumentParser(description="Train a challenger model and promote if it beats baseline.")
    parser.add_argument("--games", type=int, default=20, help="Evaluation games")
    parser.add_argument("--depth", type=int, default=4, help="Evaluation search depth")
    parser.add_argument("--time-limit", type=float, default=1.0, help="Evaluation move time")
    parser.add_argument("--promotion-threshold", type=float, default=0.55, help="Minimum score ratio to promote challenger")
    parser.add_argument("--disable-book", action="store_true", help="Disable opening book in evaluation")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run training")
    return parser.parse_args()


def run_training(candidate_path, python_executable):
    env = os.environ.copy()
    env["MODEL_OUTPUT_PATH"] = str(candidate_path)

    cmd = [python_executable, "train.py"]
    print(f"Running training command: {' '.join(cmd)}")
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise RuntimeError("Training failed; challenger was not produced.")


def main():
    args = parse_args()

    best_model = Path("best_chess_model.keras")
    baseline_model = Path("baseline_model.keras")
    challenger_model = Path("candidate_model.keras")

    if not best_model.exists():
        raise FileNotFoundError("best_chess_model.keras not found. Train baseline model first.")

    print("Preparing baseline/challenger files...")
    shutil.copy2(best_model, baseline_model)
    if challenger_model.exists():
        challenger_model.unlink()

    run_training(challenger_model, args.python)

    if not challenger_model.exists():
        raise FileNotFoundError("Challenger model was not generated.")

    print("Evaluating challenger vs baseline...")
    challenger = evaluate_models.load_model(str(challenger_model))
    baseline = evaluate_models.load_model(str(baseline_model))

    score_for_challenger, results, games = evaluate_models.evaluate_match(
        challenger,
        baseline,
        args.games,
        args.depth,
        args.time_limit,
        disable_book=args.disable_book,
        swap_colors=True,
    )

    total_games = max(1, args.games)
    ratio = score_for_challenger / total_games

    with open("promotion_match.pgn", "w", encoding="utf-8") as pgn_file:
        for game in games:
            print(game, file=pgn_file, end="\n\n")

    print("\n=== Promotion Evaluation ===")
    print(f"Raw results (white perspective): {results}")
    print(f"Challenger score: {score_for_challenger:.1f}/{total_games} ({ratio * 100:.1f}%)")
    print(f"Threshold: {args.promotion_threshold * 100:.1f}%")

    if ratio >= args.promotion_threshold:
        shutil.copy2(challenger_model, best_model)
        print("Promotion succeeded: challenger is now best_chess_model.keras")
    else:
        print("Promotion rejected: baseline kept.")


if __name__ == "__main__":
    main()

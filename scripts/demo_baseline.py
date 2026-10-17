"""Demonstrate the shared workflow on synthetic data; not benchmark results."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import FunctionTransformer

from src.benchmark import load_results, run_comparison, save_results
from src.data import RANDOM_STATE, TASKS
from src.evaluation import comparison_table
from src.plotting import plot_comparison
from src.models.checkpoints import Unavailable


def demo_fixture(task: str, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:
    if task == "regression":
        values, target = make_regression(n_samples=150, n_features=4, noise=3, random_state=seed)
    else:
        values, target = make_classification(
            n_samples=150, n_features=4, n_informative=3, n_redundant=0,
            n_classes=2 if task == "binary" else 3, n_clusters_per_class=1,
            weights=[0.85, 0.15] if task == "binary" else None, random_state=seed,
        )
        if task == "multiclass":
            target = np.array(["class_a", "class_b", "class_c"])[target]
    X = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(4)])
    X["category"] = np.where(X["feature_0"] > 0, "high", "low")
    X.loc[::11, "feature_0"] = np.nan
    X.loc[::13, "category"] = None
    return X, pd.Series(target, name="target")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--render-only", action="store_true", help="Recreate tables and plots from saved JSON")
    parser.add_argument("--classical", action="store_true", help="Include validation-tuned XGBoost and LightGBM")
    parser.add_argument("--tabpfn", action="store_true", help="Include frozen TabPFN-3.5-Fast with native preprocessing")
    parser.add_argument("--mitra", action="store_true", help="Include frozen Mitra v2 classification and distributional regression")
    parser.add_argument("--mitra-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action="store_true", help="Allow missing pinned checkpoint downloads")
    parser.add_argument("--tabpfn-estimators", default="auto", type=lambda value: "auto" if value == "auto" else int(value))
    args = parser.parse_args()
    name = ("combined-demo" if sum((args.classical, args.tabpfn, args.mitra)) > 1 else "mitra-demo" if args.mitra else "tabpfn-demo" if args.tabpfn
            else "classical-demo" if args.classical else "baseline-demo")
    args.output_dir = args.output_dir or ROOT / "results" / name
    if (args.tabpfn or args.mitra) and not args.render_only:
        import torch

        torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        path = args.output_dir / f"{task}.json"
        if args.render_only:
            results = load_results(path)
        else:
            X, y = demo_fixture(task, args.seed)
            models = ({"Dummy mean": DummyRegressor(strategy="mean"),
                       "Dummy median": DummyRegressor(strategy="median")} if task == "regression" else
                      {"Dummy prior": DummyClassifier(strategy="prior", random_state=args.seed),
                       "Dummy stratified": DummyClassifier(strategy="stratified", random_state=args.seed)})
            search_spaces = None
            preprocessors = {}
            if args.classical:
                from src.models.classical import classical_search_spaces, make_classical_models

                models.update(make_classical_models(task, args.seed))
                search_spaces = classical_search_spaces()
            if args.tabpfn:
                from src.models.tabpfn import make_tabpfn_models

                models.update(make_tabpfn_models(task, args.seed, args.device, args.tabpfn_estimators, args.download))
                preprocessors["TabPFN - Raw"] = FunctionTransformer()
            if args.mitra:
                from src.models.mitra import make_mitra_models

                models.update(make_mitra_models(task, args.seed, args.device, args.download, args.mitra_batch_size))
            try:
                results = run_comparison(X, y, models, task=task, dataset_name=task,
                                         seed=args.seed, dataset_kind="synthetic", search_spaces=search_spaces,
                                         preprocessors=preprocessors)
            except Unavailable as exc:
                parser.exit(2, f"Not evaluated: {exc}\n")
            save_results(results, path)
        table = comparison_table(results)
        table.to_csv(args.output_dir / f"{task}.csv")
        figure = plot_comparison(results)
        figure.savefig(args.output_dir / f"{task}.png", dpi=150)
        plt.close(figure)
        print(f"Synthetic {task} demo (test metrics):\n{table.to_string()}\n")


if __name__ == "__main__":
    main()

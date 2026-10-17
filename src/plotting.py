"""Comparison plots derived from saved metrics."""

from .evaluation import comparison_table

METRIC_LABELS = {"roc_auc": "ROC-AUC (higher is better)",
                 "macro_f1": "Macro-F1 (higher is better)",
                 "rmse": "RMSE (lower is better)"}


def plot_comparison(results: dict, partition: str = "test"):
    """Return a Matplotlib figure; callers choose display, export, and close."""
    import matplotlib.pyplot as plt

    metric = results["primary_metric"]
    table = comparison_table(results, partition)
    table = table.dropna(subset=[metric])
    figure, axis = plt.subplots(figsize=(max(5, len(table) * 1.5), 4), layout="constrained")
    axis.bar(table.index, table[metric])
    axis.set_ylabel(METRIC_LABELS[metric])
    axis.set_xlabel("Model")
    axis.tick_params(axis="x", labelrotation=20)
    prefix = "Synthetic demo: " if results["dataset"]["kind"] == "synthetic" else ""
    axis.set_title(f"{prefix}{results['dataset']['name']} ({partition})")
    if table.empty:
        axis.text(0.5, 0.5, "No evaluated models", ha="center", va="center", transform=axis.transAxes)
    return figure

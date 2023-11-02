from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from multitoken_estimator.training.BenchmarkIterationsResult import AggregateMetric
from multitoken_estimator.training.sweep_lre_params import SweepResult

Metric = Literal[
    "accuracy",
    "causality",
    "accuracy_avg_over_relations",
    "causality_avg_over_relations",
    "multitoken_accuracy",
    "multitoken_causality",
    "single_token_accuracy",
    "single_token_causality",
]


def plot_sweep_results(
    sweep_result: SweepResult[Any],
    metric: Metric,
    min_total_per_relation: int = 5,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 5),
    show_uncertainty: bool = False,
) -> plt.Figure:
    """
    Plot the results of a sweep
    """

    x = sweep_result.param_values
    y_metrics: list[AggregateMetric] = []
    for res in sweep_result.results:
        if metric == "accuracy":
            y_metrics.append(res.accuracy)
        elif metric == "causality":
            y_metrics.append(res.causality)
        elif metric == "multitoken_accuracy":
            y_metrics.append(res.multitoken_accuracy)
        elif metric == "multitoken_causality":
            y_metrics.append(res.multitoken_causality())
        elif metric == "single_token_accuracy":
            y_metrics.append(res.single_token_accuracy)
        elif metric == "single_token_causality":
            y_metrics.append(res.single_token_causality())
        elif metric == "accuracy_avg_over_relations":
            y_metrics.append(res.accuracy_avg_over_relations(min_total_per_relation))
        elif metric == "causality_avg_over_relations":
            y_metrics.append(res.causality_avg_over_relations(min_total_per_relation))
        else:
            raise ValueError(f"Invalid metric: {metric}")

    y = [metric.mean for metric in y_metrics]
    y_lower = [metric.mean - metric.stdev for metric in y_metrics]
    y_upper = [metric.mean + metric.stdev for metric in y_metrics]

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme()
    sns.lineplot(x=x, y=y, ax=ax)

    if ylabel is None:
        ylabel = "Accuracy" if "accuracy" in metric else "Causality"
    if title is None:
        title = sweep_result.name.replace("_", " ")
    if xlabel is None:
        xlabel = sweep_result.param_name.replace("_", " ")

    if show_uncertainty:
        ax.fill_between(x, y_lower, y_upper, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        fig.show()
    return fig

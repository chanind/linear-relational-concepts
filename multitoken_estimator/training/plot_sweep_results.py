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

MetricType = Literal["avg", "avg_over_relations", "multitoken_avg", "single_token_avg"]


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

    sns.set_theme()
    fig, ax = plt.subplots(figsize=figsize)
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


def plot_sweep_accuracy_and_causality(
    sweep_result: SweepResult[Any],
    metric_type: MetricType = "avg",
    min_total_per_relation: int = 5,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 5),
    show_uncertainty: bool = False,
) -> plt.Figure:
    """
    Plot the results of a sweep, with accuracy and causality on the same plot
    """

    x = sweep_result.param_values
    accuracy_metrics: list[AggregateMetric] = []
    causality_metrics: list[AggregateMetric] = []
    for res in sweep_result.results:
        if metric_type == "avg":
            accuracy_metrics.append(res.accuracy)
            causality_metrics.append(res.causality)
        elif metric_type == "avg_over_relations":
            accuracy_metrics.append(
                res.accuracy_avg_over_relations(min_total_per_relation)
            )
            causality_metrics.append(
                res.causality_avg_over_relations(min_total_per_relation)
            )
        elif metric_type == "multitoken_avg":
            accuracy_metrics.append(res.multitoken_accuracy)
            causality_metrics.append(res.multitoken_causality())
        elif metric_type == "single_token_avg":
            accuracy_metrics.append(res.single_token_accuracy)
            causality_metrics.append(res.single_token_causality())
        else:
            raise ValueError(f"Invalid metric_type: {metric_type}")

    accuracy = [metric.mean for metric in accuracy_metrics]
    accuracy_lower = [metric.mean - metric.stdev for metric in accuracy_metrics]
    accuracy_upper = [metric.mean + metric.stdev for metric in accuracy_metrics]

    causality = [metric.mean for metric in causality_metrics]
    causality_lower = [metric.mean - metric.stdev for metric in causality_metrics]
    causality_upper = [metric.mean + metric.stdev for metric in causality_metrics]

    sns.set_theme()
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=x, y=accuracy, ax=ax, label="Accuracy")
    sns.lineplot(x=x, y=causality, ax=ax, label="Causality")

    if ylabel is None:
        ylabel = "Score"
    if title is None:
        title = sweep_result.name.replace("_", " ")
    if xlabel is None:
        xlabel = sweep_result.param_name.replace("_", " ")

    if show_uncertainty:
        ax.fill_between(x, accuracy_lower, accuracy_upper, alpha=0.2)
        ax.fill_between(x, causality_lower, causality_upper, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        fig.show()
    return fig

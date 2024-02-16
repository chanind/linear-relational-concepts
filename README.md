# Linear Relational Concepts

This is the code for experiments that accompany the paper: "Identifying Linear Relational Concepts in Large Language Models".

If you're interested in using Linear Relational Concepts (LRCs) or Linear Relational Embeddings (LREs) in your own work,
check out the [linear-relational library](https://github.com/chanind/linear-relational). Linear-relational contains the
core reusable ideas from this paper, and is packaged in a Python library that can be installed with pip.

## Setup

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. Make sure you have Poetry installed, and run:

```
poetry install
```

## Running experiments

There are a number of experiments from the paper in the "linear_relational_concepts/experiments" dir. These are meant to be imported and run in a Jupyter notebook or an interactive Python shell. If you want to run these as bash scripts, you'll need to build a wrapper yourself to do this.

For example, to run the `benchmark_llama2` experiment, run the following:

```python
from linear_relational_concepts.experiments.benchmark_llama2 import benchmark_llama2

# this will run the main benchmark with default values, using llama2 from huggingface
benchmark_results = benchmark_llama2()

# results are a dict, with each method as a key and IterationResults as values
for method_name, iteration_results in benchmark_results.items():
    print(f"{method_name} - classification accuracy: {iteration_results.accuracy}, causality: {iteration_results.causality}")
```

## Development

This project uses [Black](https://github.com/psf/black) for code formatting, [Flake8](https://flake8.pycqa.org/en/latest/) for linting, and [Pytest](https://docs.pytest.org/) for tests, and [Mypy](https://www.mypy-lang.org/) for type checking.

Run tests with: `poetry run pytest`

## Citation

If you used this code in your research, please cite the following paper:

```
@article{chanin2023identifying,
  title={Identifying Linear Relational Concepts in Large Language Models},
  author={David Chanin and Anthony Hunter and Oana-Maria Camburu},
  journal={arXiv preprint arXiv:2311.08968},
  year={2023}
}
```

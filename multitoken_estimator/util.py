from typing import Generator, Iterable, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")


# based on https://stackoverflow.com/a/480227/245362
def dedupe_stable(items: list[T]) -> list[T]:
    seen: set[T] = set()
    seen_add = seen.add
    return [item for item in items if not (item in seen or seen_add(item))]


def shallow_flatten(items: Iterable[Iterable[T]]) -> list[T]:
    return [item for sublist in items for item in sublist]


def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
    ):
        yield data[i : i + batch_size]


def tuplify(item: T | tuple[T, ...]) -> tuple[T, ...]:
    return item if isinstance(item, tuple) else (item,)

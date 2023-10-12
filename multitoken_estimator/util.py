from typing import TypeVar

T = TypeVar("T")


# based on https://stackoverflow.com/a/480227/245362
def dedupe_stable(items: list[T]) -> list[T]:
    seen: set[T] = set()
    seen_add = seen.add
    return [item for item in items if not (item in seen or seen_add(item))]

from multitoken_estimator.lib.util import dedupe_stable


def test_dedule_stable() -> None:
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert dedupe_stable(items) == items
    assert dedupe_stable(items + items) == items
    assert dedupe_stable(items + items + items) == items

from linear_relational_concepts.lib.PromptValidator import cache_key


def test_cache_key() -> None:
    assert cache_key("blah this is a prompt", "answer") == "474096ceddcbb81"

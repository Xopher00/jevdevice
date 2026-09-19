"""Pure functions only, no network, no device -- ui.py's contract."""

from __future__ import annotations

from jevdevice.elements import parse_actionable_elements
from jevdevice.ui import _ELEMENT_CACHE, _LRUCache


def test_lru_cache_evicts_oldest_past_capacity() -> None:
    cache = _LRUCache(maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["d"] = 4
    assert len(cache) == 3
    assert "a" not in cache
    assert cache.get("d") == 4


def test_lru_cache_get_refreshes_recency() -> None:
    cache = _LRUCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache.get("a")  # touch "a" so "b" becomes the oldest
    cache["c"] = 3
    assert "a" in cache
    assert "b" not in cache


def test_element_cache_is_bounded_at_module_scope() -> None:
    assert _ELEMENT_CACHE.maxsize == 500


def test_parse_actionable_elements_reads_clickable_nodes() -> None:
    dump_xml = (
        '<hierarchy>'
        '<node text="Send" resource-id="" content-desc="" clickable="true" bounds="[0,0][100,50]"/>'
        '<node text="" resource-id="" content-desc="" clickable="false" bounds="[0,50][100,100]"/>'
        '</hierarchy>'
    )
    elements = parse_actionable_elements(dump_xml)
    assert list(elements) == ["text='Send'"]
    assert (elements["text='Send'"].x, elements["text='Send'"].y) == (50, 25)

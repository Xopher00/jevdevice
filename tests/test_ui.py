"""Pure functions only, no network, no device -- ui.py's contract."""

from __future__ import annotations

from jevdevice.elements import describe_screen, parse_actionable_elements, parse_all_elements, parse_editable_elements
from jevdevice.ui import _ELEMENT_CACHE, _LRUCache, _no_editable_field_reasons


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


def test_describe_screen_returns_the_same_labels_as_parse_all_elements() -> None:
    dump_xml = (
        '<hierarchy>'
        '<node text="Send" resource-id="" content-desc="" clickable="true" bounds="[0,0][100,50]"/>'
        '<node text="Cancel" resource-id="" content-desc="" clickable="true" bounds="[0,50][100,100]"/>'
        '</hierarchy>'
    )
    assert describe_screen(dump_xml) == ["text='Send'", "text='Cancel'"]


def test_describe_screen_limit_truncates() -> None:
    nodes = "".join(
        f'<node text="item{i}" resource-id="" content-desc="" clickable="true" bounds="[0,{i}][100,{i + 10}]"/>'
        for i in range(5)
    )
    dump_xml = f"<hierarchy>{nodes}</hierarchy>"
    assert len(describe_screen(dump_xml, limit=2)) == 2


def test_describe_screen_goal_relevant_label_survives_truncation() -> None:
    nodes = "".join(
        f'<node text="noise{i}" resource-id="" content-desc="" clickable="true" bounds="[0,{i}][100,{i + 10}]"/>'
        for i in range(5)
    )
    nodes += '<node text="search field" resource-id="" content-desc="" clickable="true" bounds="[0,100][100,110]"/>'
    dump_xml = f"<hierarchy>{nodes}</hierarchy>"
    assert "text='search field'" in describe_screen(dump_xml, goal="type into the search field", limit=2)


def test_no_editable_field_reasons_names_a_real_facade_candidate() -> None:
    dump_xml = (
        '<hierarchy>'
        '<node text="" resource-id="fake_search_box" content-desc="Search" class="android.widget.LinearLayout" '
        'clickable="true" bounds="[0,0][100,50]"/>'
        '</hierarchy>'
    )
    reasons = _no_editable_field_reasons(dump_xml, "type into the search field")
    assert reasons[0] == "no real editable fields (EditText/AutoCompleteTextView) on screen"
    assert "fake_search_box" in reasons[1] or "Search" in reasons[1]


def test_no_editable_field_reasons_with_nothing_clickable_either() -> None:
    dump_xml = '<hierarchy><node text="hello" resource-id="" content-desc="" clickable="false" bounds="[0,0][100,50]"/></hierarchy>'
    reasons = _no_editable_field_reasons(dump_xml, "type into the search field")
    assert reasons == ("no real editable fields (EditText/AutoCompleteTextView) on screen",)


def test_parse_editable_elements_labels_an_unlabeled_field_from_its_ancestor() -> None:
    dump_xml = (
        '<hierarchy>'
        '<node text="" resource-id="com.example:id/peoplekit_autocomplete_bar_to" content-desc="" '
        'class="android.widget.RelativeLayout" clickable="false" bounds="[0,449][1080,617]">'
        '<node text="" resource-id="" content-desc="" class="android.widget.EditText" '
        'clickable="true" focused="true" bounds="[216,461][936,605]"/>'
        '</node>'
        '</hierarchy>'
    )
    elements = parse_editable_elements(dump_xml)
    assert list(elements) == ["EditText under resource-id='peoplekit_autocomplete_bar_to'"]
    label = "EditText under resource-id='peoplekit_autocomplete_bar_to'"
    assert (elements[label].x, elements[label].y) == (576, 533)


def test_parse_all_elements_unaffected_by_the_label_context_fallback() -> None:
    dump_xml = (
        '<hierarchy>'
        '<node text="" resource-id="parent" content-desc="" class="android.widget.RelativeLayout" '
        'clickable="false" bounds="[0,0][100,100]">'
        '<node text="" resource-id="" content-desc="" class="android.widget.EditText" '
        'clickable="true" bounds="[0,0][100,50]"/>'
        '</node>'
        '</hierarchy>'
    )
    assert list(parse_all_elements(dump_xml)) == ["resource-id='parent'"]

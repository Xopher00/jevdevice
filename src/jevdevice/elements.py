"""Parse the real, currently on-screen UI: a real `uiautomator2` accessibility-tree dump is a
real, runtime-discovered enumeration of every element on screen -- the same
"real enumeration -> semantic narrow -> deterministic action -> Jev verify" pattern app_launch.py
uses for packages and services.py uses for dumpsys services applies unchanged.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .matching import fuzzy_narrow
from .transport import AdbTransport


async def dump_screen(transport: AdbTransport) -> str:
    return await transport.dump_hierarchy()


def foreground_package(dump_xml: str) -> str | None:
    """Majority package among visible elements, not the single topmost window:
    a transient popup contributes few elements (or none by this name at all)
    and never wins the vote."""
    packages: dict[str, int] = {}
    for pkg in re.findall(r'package="([^"]+)"', dump_xml):
        if pkg not in {"android", "com.android.systemui"}:
            packages[pkg] = packages.get(pkg, 0) + 1
    return max(packages, key=packages.get) if packages else None


@dataclass
class Element:
    x: int
    y: int
    bounds: str  # real device-reported bounds string, e.g. "[166,1173][415,1615]" -- passed to the gate as evidence
    text: str = ""  # the node's own real current text, e.g. an EditText's existing content


def _nearest_ancestor_bounds(node, parent_of: dict, attr: str) -> str | None:
    """Custom-drawn menus often mark only a container `attr="true"` and leave its labeled
    text/icon children `attr="false"` -- confirmed live (a launcher long-press menu's "Missed
    calls" label is clickable="false" two levels under a clickable="true" real tap target)."""
    ancestor = parent_of.get(node)
    while ancestor is not None:
        if ancestor.attrib.get(attr) == "true":
            return ancestor.attrib.get("bounds")
        ancestor = parent_of.get(ancestor)
    return None


def _short_class(class_name: str) -> str:
    return class_name.rsplit(".", 1)[-1] or "element"


def _short_id(resource_id: str) -> str:
    return resource_id.rsplit("/", 1)[-1]


def _context_label(node, parent_of: dict, attrs: dict) -> str | None:
    """A real, interactable node with none of its own text/resource-id/content-desc (confirmed
    live: Gmail's To field is a real, focused, unlabeled EditText) still needs a real identity --
    borrow the nearest labeled ancestor's, since a layout container's own id is real context,
    never invented content."""
    ancestor = parent_of.get(node)
    while ancestor is not None:
        for key in ("resource-id", "content-desc", "text"):
            value = ancestor.attrib.get(key)
            if value:
                shown = _short_id(value) if key == "resource-id" else value
                return f"{_short_class(attrs.get('class', ''))} under {key}={shown!r}"
        ancestor = parent_of.get(ancestor)
    return None


def _parse_elements(dump_xml: str, is_match, ancestor_attr: str | None = None, label_context: bool = False) -> dict[str, Element]:
    """Every real node passing `is_match` (a real Android accessibility signal, never a
    per-app guess), described by whichever of its own real fields are non-empty. A labeled
    node that fails `is_match` but has a real ancestor matching `ancestor_attr` still counts,
    using that ancestor's bounds as the real tap target. `label_context` additionally keeps a
    matching node that has none of its own descriptive fields, labeled from a real ancestor."""
    root = ET.fromstring(dump_xml)
    parent_of = {child: parent for parent in root.iter() for child in parent}
    elements: dict[str, Element] = {}
    for node in root.iter("node"):
        attrs = node.attrib
        described = {k: v for k in ("text", "resource-id", "content-desc") if (v := attrs.get(k))}
        matched = is_match(attrs)
        if matched:
            bounds = attrs.get("bounds", "")
        elif ancestor_attr:
            bounds = _nearest_ancestor_bounds(node, parent_of, ancestor_attr)
        else:
            bounds = None
        if not bounds:
            continue
        if described:
            label = " ".join(f"{k}={v!r}" for k, v in described.items())
        elif label_context and matched:
            label = _context_label(node, parent_of, attrs)
            if label is None:
                continue
        else:
            continue
        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
        if not match:
            continue
        left, top, right, bottom = (int(n) for n in match.groups())
        elements[label] = Element((left + right) // 2, (top + bottom) // 2, bounds, attrs.get("text", ""))
    return elements


def parse_actionable_elements(dump_xml: str) -> dict[str, Element]:
    """Every real clickable node, plus a labeled node under a clickable container --
    Android's own signal for "this is tappable"."""
    return _parse_elements(dump_xml, lambda attrs: attrs.get("clickable") == "true", ancestor_attr="clickable")


# A subclass's own class name doesn't always contain "EditText" -- confirmed live:
# Settings' search bar is a real android.widget.AutoCompleteTextView (extends EditText).
_EDITABLE_CLASSES = ("EditText", "AutoCompleteTextView")


def parse_editable_elements(dump_xml: str) -> dict[str, Element]:
    """Every real node whose Android `class` is (a subclass of) EditText -- the
    OS's own signal for "this accepts text input", not a per-app guess."""
    return _parse_elements(dump_xml, lambda attrs: any(c in attrs.get("class", "") for c in _EDITABLE_CLASSES), label_context=True)


def parse_long_clickable_elements(dump_xml: str) -> dict[str, Element]:
    """Every real long-clickable node, plus a labeled node under a long-clickable
    container -- Android's own signal for "this supports long-press"."""
    return _parse_elements(dump_xml, lambda attrs: attrs.get("long-clickable") == "true", ancestor_attr="long-clickable")


def parse_all_elements(dump_xml: str) -> dict[str, Element]:
    """Every real described node, clickable or not -- a scroll target need not be tappable
    to prove it's visible."""
    return _parse_elements(dump_xml, lambda _attrs: True)


def describe_screen(dump_xml: str, *, goal: str | None = None, limit: int = 150) -> list[str]:
    """Compact real-element labels for a Jev state value, in place of raw XML (verbose,
    truncates blindly) -- reuses the same label shape narrow_and_pick already consumes
    everywhere else. When bounding is needed, order by goal-relevance first so the drop
    favors what's actually relevant, not whatever came last in the tree."""
    labels = list(parse_all_elements(dump_xml))
    if goal is not None and len(labels) > limit:
        labels = fuzzy_narrow(goal, labels, limit=len(labels))
    return labels[:limit]


def screen_summary(dump_xml: str, *, goal: str | None = None) -> dict:
    """One dump, everything a caller needs to ground a decision in the real current screen --
    no extra device round trip beyond the dump already taken."""
    return {
        "foreground_package": foreground_package(dump_xml),
        "editable_fields": list(parse_editable_elements(dump_xml)),
        "clickable_count": len(parse_actionable_elements(dump_xml)),
        "on_screen": describe_screen(dump_xml, goal=goal),
    }

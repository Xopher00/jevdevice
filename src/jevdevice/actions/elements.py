"""Parse the real, currently on-screen UI: a real `uiautomator2` accessibility-tree dump is a
real, runtime-discovered enumeration of every element on screen -- the same
"real enumeration -> semantic narrow -> deterministic action -> Jev verify" pattern app_launch.py
uses for packages and services.py uses for dumpsys services applies unchanged.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from jevdevice.budget import current_profile
from jevdevice.device import Device
from jevdevice.matching import fuzzy_narrow


async def dump_screen(device: Device) -> str:
    return await device.dump_hierarchy()


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
    """One real on-screen element.

    ACI-aligned observation keys, mapped onto what this parser already emits
    (role/text/bbox/interactable -- no fields invented for devices we don't
    have): `role` is the Android class noun (_CLASS_NOUNS: button, text field,
    toggle...); `text` the node's real text; `bounds` (+ the x/y center)
    is the device-reported bbox; `interactable` is not a stored field -- it is
    WHICH parse_* family selected the node (parse_actionable_elements =
    tappable, parse_long_clickable_elements = long-pressable,
    parse_editable_elements = text-input), Android's own signal, never a
    per-app guess."""
    x: int
    y: int
    bounds: str  # real device-reported bounds string, e.g. "[166,1173][415,1615]" -- passed to the gate as evidence
    text: str = ""  # the node's own real current text, e.g. an EditText's existing content
    description: str = ""  # natural-language phrasing of the same real fields, for fit questions
    short: str = ""  # short raw option label: the judge's Choice option text, ~1-3 tokens
    role: str = "element"  # ACI role: the class noun already phrased inside `description`


def _nearest_ancestor_bounds(node, parent_of: dict, attr: str) -> str | None:
    """Custom-drawn menus often mark only a container `attr="true"` and leave its labeled
    text/icon children `attr="false"` (e.g. a launcher long-press menu's "Missed calls"
    label is clickable="false" two levels under a clickable="true" real tap target)."""
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


_CLASS_NOUNS = (
    ("EditText", "text field"), ("AutoCompleteTextView", "text field"),
    ("Button", "button"), ("CheckBox", "toggle"), ("Switch", "toggle"), ("TextView", "text"),
)


def _class_noun(class_name: str) -> str:
    return next((n for pattern, n in _CLASS_NOUNS if pattern in class_name), "element")


def _natural_description(described: dict, attrs: dict) -> str:
    """A raw `text='X' resource-id='Y'` label in a fit question depresses the fit
    score versus the same facts phrased as a sentence."""
    noun = _class_noun(attrs.get("class", ""))
    clauses = []
    if "content-desc" in described:
        clauses.append(f"labelled {described['content-desc']!r}")
    if "text" in described:
        clauses.append(f"showing the text {described['text']!r}")
    if "resource-id" in described:
        clauses.append(f"identified as {_short_id(described['resource-id'])!r}")
    return f"the {noun} " + ", ".join(clauses) if clauses else f"the {noun}"


def _short_label(described: dict) -> str:
    """The element's own short raw label: its real text, real content-desc, or
    short resource-id -- the Choice option text, at ~1-3 tokens instead of the
    full label's ~10+ (an in-process head fits ~20 options)."""
    for key in ("text", "content-desc"):
        if key in described:
            return described[key]
    return _short_id(described["resource-id"])


def _context_identity(node, parent_of: dict, class_name: str) -> tuple[str, str] | None:
    """An unlabeled real node borrows the nearest labeled ancestor's real identity.
    Returns (short shown value, full 'Class under key=value' label) -- the short
    value is the option text, the full label the state-text description."""
    ancestor = parent_of.get(node)
    while ancestor is not None:
        for key in ("resource-id", "content-desc", "text"):
            value = ancestor.attrib.get(key)
            if value:
                shown = _short_id(value) if key == "resource-id" else value
                return shown, f"{_short_class(class_name)} under {key}={shown!r}"
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
            short = _short_label(described)
        elif label_context and matched:
            identity = _context_identity(node, parent_of, attrs.get("class", ""))
            if identity is None:
                continue
            shown, label = identity
            short = shown
        else:
            continue
        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
        if not match:
            continue
        left, top, right, bottom = (int(n) for n in match.groups())
        description = _natural_description(described, attrs) if described else ""
        elements[label] = Element((left + right) // 2, (top + bottom) // 2, bounds, attrs.get("text", ""), description, short,
                                  _class_noun(attrs.get("class", "")))
    return elements


def parse_actionable_elements(dump_xml: str) -> dict[str, Element]:
    """Every real clickable node, plus a labeled node under a clickable container --
    Android's own signal for "this is tappable". `label_context=True`: a clickable node with
    none of its own text/resource-id/content-desc is still a real tap target, not noise."""
    return _parse_elements(dump_xml, lambda attrs: attrs.get("clickable") == "true", ancestor_attr="clickable", label_context=True)


# A subclass's own class name doesn't always contain "EditText" -- e.g. a search
# bar can be a real android.widget.AutoCompleteTextView (extends EditText).
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


def count_unlabeled_interactive(dump_xml: str) -> dict[str, int]:
    """Free (no Jev) structural tally of real clickable/long-clickable nodes with none of
    text/resource-id/content-desc."""
    root = ET.fromstring(dump_xml)
    counts = {"clickable": 0, "clickable_unlabeled": 0, "long_clickable": 0, "long_clickable_unlabeled": 0}
    for node in root.iter("node"):
        attrs = node.attrib
        described = any(attrs.get(k) for k in ("text", "resource-id", "content-desc"))
        if attrs.get("clickable") == "true":
            counts["clickable"] += 1
            counts["clickable_unlabeled"] += not described
        if attrs.get("long-clickable") == "true":
            counts["long_clickable"] += 1
            counts["long_clickable_unlabeled"] += not described
    return counts


def short_options(elements: dict[str, Element]) -> dict[str, Element]:
    """Choice options for profiles with short_labels: short raw labels as the
    option text -- what actually costs head tokens -- while the rich
    descriptions stay in the state text. Values are the full Elements, so
    callers keep coordinates/bounds/description keyed by the option text the
    judge will answer with. Collisions (two real elements whose short labels
    match) get a '#2'-style suffix: an option list can't repeat itself."""
    result: dict[str, Element] = {}
    for label, element in elements.items():
        candidate, n = element.short or label, 1
        while candidate in result:
            n += 1
            candidate = f"{element.short or label} #{n}"
        result[candidate] = element
    return result


def describe_screen(dump_xml: str, *, goal: str | None = None, limit: int | None = None, telemetry: dict | None = None) -> list[str]:
    """Compact real-element labels for a Jev state value, in place of raw XML (verbose,
    truncates blindly) -- reuses the same label shape narrow_and_pick already consumes
    everywhere else. When bounding is needed, order by goal-relevance first so the drop
    favors what's actually relevant, not whatever came last in the tree.

    `telemetry`, when given a dict, is filled with what was cut (elements/bytes before
    and after) -- the caller passes it on to its ask() so the decision row records the
    truncation. No behavior change.

    `limit` is a named knob, never a bare int: the answering engine's profile
    screen_limit (budget.py). With no explicit limit the process engine's profile
    applies (JEV_ENGINE, the same env bootstrap reads)."""
    limit = current_profile().screen_limit if limit is None else limit
    labels = list(parse_all_elements(dump_xml))
    if telemetry is not None:
        telemetry["elements_before"] = len(labels)
        telemetry["bytes_before"] = len(dump_xml.encode("utf-8"))
    if goal is not None and len(labels) > limit:
        labels = fuzzy_narrow(goal, labels, limit=len(labels))
    kept = labels[:limit]
    if telemetry is not None:
        telemetry["elements_after"] = len(kept)
        telemetry["bytes_after"] = sum(len(label.encode("utf-8")) for label in kept)
    return kept


def screen_summary(dump_xml: str, *, goal: str | None = None, telemetry: dict | None = None, limit: int | None = None) -> dict:
    """One dump, everything a caller needs to ground a decision in the real current screen --
    no extra device round trip beyond the dump already taken. `telemetry`/`limit` pass
    through to describe_screen's truncation stats/profile knob."""
    return {
        "foreground_package": foreground_package(dump_xml),
        "editable_fields": list(parse_editable_elements(dump_xml)),
        "clickable_count": len(parse_actionable_elements(dump_xml)),
        "on_screen": describe_screen(dump_xml, goal=goal, telemetry=telemetry, limit=limit),
    }

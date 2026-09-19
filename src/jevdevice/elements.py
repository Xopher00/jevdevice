"""Parse the real, currently on-screen UI: a real `uiautomator2` accessibility-tree dump is a
real, runtime-discovered enumeration of every element on screen -- the same
"real enumeration -> semantic narrow -> deterministic action -> Jev verify" pattern app_launch.py
uses for packages and services.py uses for dumpsys services applies unchanged.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

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


def _parse_elements(dump_xml: str, is_match) -> dict[str, Element]:
    """Every real node passing `is_match` (a real Android accessibility signal,
    never a per-app guess), described by whichever of its own real fields are
    non-empty, mapped to the center of its real bounds."""
    elements: dict[str, Element] = {}
    for node in ET.fromstring(dump_xml).iter("node"):
        attrs = node.attrib
        if not is_match(attrs):
            continue
        described = {k: v for k in ("text", "resource-id", "content-desc") if (v := attrs.get(k))}
        if not described:
            continue
        bounds = attrs.get("bounds", "")
        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
        if not match:
            continue
        left, top, right, bottom = (int(n) for n in match.groups())
        label = " ".join(f"{k}={v!r}" for k, v in described.items())
        elements[label] = Element((left + right) // 2, (top + bottom) // 2, bounds, attrs.get("text", ""))
    return elements


def parse_actionable_elements(dump_xml: str) -> dict[str, Element]:
    """Every real clickable node -- Android's own signal for "this is tappable"."""
    return _parse_elements(dump_xml, lambda attrs: attrs.get("clickable") == "true")


def parse_editable_elements(dump_xml: str) -> dict[str, Element]:
    """Every real node whose Android `class` is (a subclass of) EditText -- the
    OS's own signal for "this accepts text input", not a per-app guess."""
    return _parse_elements(dump_xml, lambda attrs: "EditText" in attrs.get("class", ""))


def parse_long_clickable_elements(dump_xml: str) -> dict[str, Element]:
    """Every real long-clickable node -- Android's own signal for "this supports long-press"."""
    return _parse_elements(dump_xml, lambda attrs: attrs.get("long-clickable") == "true")


def parse_all_elements(dump_xml: str) -> dict[str, Element]:
    """Every real described node, clickable or not -- a scroll target need not be tappable
    to prove it's visible."""
    return _parse_elements(dump_xml, lambda _attrs: True)

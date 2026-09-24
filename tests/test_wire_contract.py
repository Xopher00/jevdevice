"""The FROZEN wire contract, pinned: the question models come from
typesymbolic (`jev.py` re-exports them), so these exact serialized shapes
guard the boundary -- a core-side model change that alters any byte here
must fail this test before it can silently change what the judge is asked.
"""

from __future__ import annotations

from jevdevice.jev import Choice, Noul, Score


def test_noul_wire_shape_is_unchanged() -> None:
    assert Noul(instructions="is this safe?").model_dump(mode="json", exclude_none=True) == {
        "type": "noul",
        "instructions": "is this safe?",
    }


def test_choice_wire_shape_keeps_undescribed_labels() -> None:
    # A None criteria value is a live-enumerated option with no description --
    # it must stay a JSON null on the wire.
    dumped = Choice(instructions="pick one", criteria={"pkg.a": None, "pkg.b": "described"}).model_dump(
        mode="json", exclude_none=True
    )
    assert dumped == {
        "type": "choice",
        "instructions": "pick one",
        "criteria": {"pkg.a": None, "pkg.b": "described"},
    }


def test_score_wire_shape_is_unchanged() -> None:
    assert Score(instructions="rate it", criteria=["low", "high"]).model_dump(mode="json", exclude_none=True) == {
        "type": "score",
        "instructions": "rate it",
        "criteria": ["low", "high"],
    }

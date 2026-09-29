"""
Check what `/context` does with a page's navigation links, without a database or a network.

    python -m scripts.check_context

Scoped to the navigation-links change rather than to all of B9: the endpoint's matching, caching
and session write are unchanged and remain unchecked here. What is new, and what this covers:

1. A snapshot carrying `links` is accepted. Both request models are `extra="forbid"`, so an
   undeclared key is a 400 and the extension's whole snapshot is lost — which is why the fold
   into `buttons` existed and why this is the check that lets it be removed.
2. A flagged navigation link does **not** raise the checkpoint. A portal masthead carries "Make a
   payment" on every page of the site, including the pages that have no payment on them, so
   counting one would put the panel into a checkpoint on a landing page and stop the Copilot
   before it has done anything. Only a control on this page gates this page.
3. A flagged *field* still raises it, by either signal. The carve-out is for links and nothing
   else, and a carve-out that quietly widened would mean writing into a password box.

`_checkpoint` is reached into directly: it is a pure function of the request, and the alternative
is a stubbed database session for a property that has nothing to do with one.
"""

from typing import Any

from pydantic import ValidationError

from src.context.constants import MAX_LINKS
from src.context.schemas import ContextRequest
from src.context.service import _checkpoint  # noqa: PLC2701  # pure, and the subject of check 2

URL = "https://portal.example.gov.ng/"

# A portal landing page: no form, and every route off it an ordinary <a href>. The masthead's
# payment link is flagged by the content script.
LINKS: list[dict[str, Any]] = [
    {"field_id": "g1-f1", "text": "Home", "href": "https://portal.example.gov.ng/"},
    {"field_id": "g1-f2", "text": "Renew Licence", "href": "https://portal.example.gov.ng/renew"},
    {
        "field_id": "g1-f3",
        "text": "Make a payment",
        "href": "https://remita.net/pay",
        "external": True,
        "sensitive": True,
    },
]


def request(**extra: Any) -> ContextRequest:
    """A well-formed snapshot, with room to vary one thing at a time."""
    return ContextRequest.model_validate({"tab_id": 1, "url": URL, **extra})


def expect(failures: list[str], condition: bool, description: str) -> None:  # noqa: FBT001
    if not condition:
        failures.append(description)


def check_links_are_accepted() -> list[str]:
    """A `links` key reaches the service instead of being refused as an unknown field."""
    failures: list[str] = []
    payload = request(links=LINKS)

    expect(failures, len(payload.links) == len(LINKS), "the links did not survive validation")
    expect(
        failures,
        payload.links[2].external and payload.links[2].sensitive,
        "a link lost the content script's own judgement about it",
    )
    expect(
        failures,
        payload.links[1].href == "https://portal.example.gov.ng/renew",
        "a link's address did not survive, so an answer cannot be grounded in where it goes",
    )

    try:
        request(links=[*LINKS] * (MAX_LINKS // len(LINKS) + 2))
        failures.append("a snapshot past the link cap was accepted")
    except ValidationError:
        pass

    return failures


def check_a_flagged_link_does_not_gate_the_page() -> list[str]:
    """The masthead's payment link is flagged, and the landing page is still usable."""
    failures: list[str] = []
    payload = request(
        links=LINKS,
        sensitive_flags=[{"field_id": "g1-f3", "reason": "payment"}],
    )
    checkpoint = _checkpoint(payload)

    expect(failures, not checkpoint.present, "a flagged navigation link raised the checkpoint")
    expect(failures, checkpoint.count == 0, f"the count was {checkpoint.count}, expected 0")

    return failures


def check_a_flagged_field_still_gates_the_page() -> list[str]:
    """Both of the content script's signals still work, on a field, exactly as before."""
    failures: list[str] = []

    inline = request(
        fields=[{"field_id": "f1", "label": "Password", "type": "password", "sensitive": True}],
        links=LINKS,
    )
    expect(failures, _checkpoint(inline).present, "an inline sensitive field stopped counting")

    flagged = request(
        fields=[{"field_id": "f1", "label": "One-Time Code", "type": "text"}],
        links=LINKS,
        sensitive_flags=[{"field_id": "f1", "reason": "otp"}],
    )
    expect(failures, _checkpoint(flagged).present, "a sensitive_flags entry stopped counting")

    both = request(
        fields=[
            {"field_id": "f1", "label": "Password", "type": "password", "sensitive": True},
            {"field_id": "f2", "label": "One-Time Code", "type": "text"},
        ],
        links=LINKS,
        sensitive_flags=[
            {"field_id": "f2", "reason": "otp"},
            # The masthead's link, on the same page as two real checkpoints.
            {"field_id": "g1-f3", "reason": "payment"},
        ],
    )
    counted = _checkpoint(both)
    expect(
        failures,
        counted.count == 2,
        f"the count was {counted.count}, expected 2 — the link was counted with the fields",
    )

    return failures


def check_a_page_with_no_links_is_unchanged() -> list[str]:
    """The carve-out cannot change a snapshot that has no links in it."""
    failures: list[str] = []
    payload = request(
        fields=[{"field_id": "f1", "label": "Password", "type": "password", "sensitive": True}],
        sensitive_flags=[{"field_id": "f1", "reason": "password"}],
    )
    checkpoint = _checkpoint(payload)

    expect(failures, checkpoint.present and checkpoint.count == 1, "a form page stopped counting")
    expect(failures, payload.links == [], "links defaulted to something other than empty")

    return failures


CHECKS = [
    ("a snapshot carrying links is accepted", check_links_are_accepted),
    ("a flagged navigation link does not gate the page", check_a_flagged_link_does_not_gate_the_page),
    ("a flagged field still gates the page, by either signal", check_a_flagged_field_still_gates_the_page),
    ("a page with no links is unaffected", check_a_page_with_no_links_is_unchanged),
]


def main() -> int:
    failures: list[str] = []

    print(f"checks ({len(CHECKS)}):")
    for name, check in CHECKS:
        try:
            result = check()
        except Exception as error:  # noqa: BLE001  # a raise is a failure like any other
            result = [f"raised {type(error).__name__}: {error}"]

        print(f"  {'FAIL' if result else 'ok  '}  {name}")
        failures += [f"{name}: {failure}" for failure in result]

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("\nAll context checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

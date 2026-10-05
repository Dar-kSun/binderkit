"""The portfolio may publish only numbers that `studies/` already contains.

`portfolio/README.md` states the rule this test enforces: `portfolio/` is a
derived view of `studies/`, and a portfolio that drifts from the study is worse
than no portfolio, because the whole claim of the repo is that its numbers are
reproducible.

Enforcement has three parts:

1. every entry in `numbers.json` still resolves to the same value from the
   `studies/` file its `source` field names;
2. every number displayed on the page carries a `data-n` attribute naming a key
   in `numbers.json`, and its text equals that key's rendered `display`;
3. no claim withdrawn in docs/SPEC.md section 8.7 appears anywhere in `portfolio/`,
   and the page depends on no external host.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PORTFOLIO = Path("portfolio")
NUMBERS = PORTFOLIO / "numbers.json"
INDEX = PORTFOLIO / "index.html"
SUMMARY = PORTFOLIO / "SUMMARY.md"
FIGURES = PORTFOLIO / "figures"

#: Figures copied from the studies, with the directory each came from.
EXPECTED_FIGURES = {
    "calibration.png": "studies/retrospective/figures",
    "paired_differences.png": "studies/retrospective/figures",
    "per_target_auroc.png": "studies/retrospective/figures",
    "auroc_pooled_vs_within.png": "studies/retrospective/figures",
    "binder_rate_by_target.png": "studies/retrospective/figures",
    "geometry_conditional.png": "studies/interface_geometry/figures",
    "geometry_per_target.png": "studies/interface_geometry/figures",
}

#: Withdrawn in docs/SPEC.md section 8.7. None may reappear on a published page.
WITHDRAWN = (
    "no published precedent",
    "nobody has checked",
    "nobody had checked",
    "geometry is redundant",
    "structural analysis is redundant",
    "fewer interface contacts",
)


@pytest.fixture(scope="module")
def numbers() -> dict:
    if not NUMBERS.is_file():
        pytest.skip("portfolio/numbers.json not built; run python -m portfolio.build")
    return json.loads(NUMBERS.read_text(encoding="utf-8"))["numbers"]


# ---------------------------------------------------------------------------
# 1. numbers.json still agrees with studies/
# ---------------------------------------------------------------------------


def test_every_number_still_resolves_to_its_source(numbers: dict) -> None:
    """Re-read each value from the study file its source field names."""
    from portfolio.build import resolve

    for key, rec in numbers.items():
        fresh = resolve(rec["source"])
        if isinstance(fresh, float):
            assert fresh == pytest.approx(rec["value"]), f"{key} drifted from {rec['source']}"
        else:
            assert fresh == rec["value"], f"{key} drifted from {rec['source']}"


def test_rebuilding_numbers_json_is_a_no_op(numbers: dict) -> None:
    """The committed file must be exactly what the generator produces."""
    from portfolio.build import build

    rebuilt = build()["numbers"]
    assert set(rebuilt) == set(numbers), "numbers.json is stale; re-run portfolio.build"
    for key in rebuilt:
        assert rebuilt[key]["display"] == numbers[key]["display"], key


def test_every_source_points_into_studies(numbers: dict) -> None:
    """Nothing may be sourced from the portfolio itself, or from thin air."""
    for key, rec in numbers.items():
        for part in rec["source"].replace("ratio:", "").split("|"):
            assert part.startswith("studies/"), f"{key} is not sourced from studies/"


# ---------------------------------------------------------------------------
# 2. the page shows only those numbers
# ---------------------------------------------------------------------------

DATA_N = re.compile(r'data-n="([^"]+)"[^>]*>([^<]*)<')


def test_page_numbers_match_numbers_json(numbers: dict) -> None:
    """Each `data-n` span's text must equal that key's rendered display."""
    html = INDEX.read_text(encoding="utf-8")
    found = DATA_N.findall(html)
    assert found, "no data-n annotated numbers found in index.html"
    for key, shown in found:
        assert key in numbers, f"index.html cites unknown number {key!r}"
        expected = numbers[key]["display"]
        assert shown.strip() == expected, (
            f"{key}: page shows {shown.strip()!r}, numbers.json says {expected!r}"
        )


def test_every_displayed_number_is_annotated() -> None:
    """A bare figure in the prose would escape the check above, so forbid it.

    Strips the CSS, the annotated values and the markup, then scans what a
    reader actually sees. Anything numeric left must be in the allowlist, and
    each entry there has a reason for not being a study result.
    """
    html = INDEX.read_text(encoding="utf-8")
    body = html.split("</style>", 1)[1]
    visible = DATA_N.sub(" ", body)
    visible = re.sub(r"<[^>]+>", " ", visible)
    visible = visible.replace("&nbsp;", " ")
    # "1. Two conclusions ..." is section numbering, not a quantity.
    visible = re.sub(r"(?<![\d.])\d+\.\s+(?=[A-Z])", " ", visible)

    allowed = {
        "95%",  # the confidence level itself, not a result
        "0.5",  # chance, as a concept
        "0.6",  # the inflection point, named in the calibration report
        "0.03",  # an illustrative magnitude in the method section
        "14",  # target count, stated beside an annotated count
        "2025",  # the year of the cited paper
        "3,766",  # the cited paper's corpus size
        "11.6%",  # the cited paper's binder rate, quoted from it
        "1.5",  # the published peripheral-trim width, a method parameter
        "1,440",  # the dataset's size, stated in the footer
        "1,189",  # study 2's n, also shown annotated
        "20",  # the competition's submission cap
    }
    for token in re.findall(r"\d[\d,]*\.?\d*%?", visible):
        assert token in allowed, (
            f"index.html shows the unannotated number {token!r}; "
            "wrap it in a data-n element or add it to the allowlist with a reason"
        )


def test_summary_quotes_only_published_numbers(numbers: dict) -> None:
    """Every decimal or percentage in SUMMARY.md must be one of ours."""
    if not SUMMARY.is_file():
        pytest.skip("SUMMARY.md not written yet")
    text = SUMMARY.read_text(encoding="utf-8")
    displays = {rec["display"] for rec in numbers.values()}
    displays |= {d.lstrip("+-") for d in displays}
    displays |= {d.rstrip("x") for d in displays}
    allowed = displays | {
        "95%",  # the confidence level itself, not a result
        "0.03",  # an illustrative magnitude, not a result
        "1.5",  # the published peripheral-trim width, a method parameter
        "1,440",
        "3,766",
        "2025",
        "1,320",
        "1,189",
        "1,235",
        "0.5",
        "0.6",
        "14",
        "20",
        "10",
        "13",
    }
    for token in re.findall(r"\d[\d,]*\.\d+%?|\d[\d,]*%", text):
        assert token in allowed, (
            f"SUMMARY.md quotes {token!r}, which is not in numbers.json. "
            "Add it to build.py with a source, or remove it."
        )


# ---------------------------------------------------------------------------
# 3. the page is honest, self-contained, and complete
# ---------------------------------------------------------------------------


def test_no_withdrawn_claim_appears_anywhere_in_the_portfolio() -> None:
    for path in PORTFOLIO.rglob("*"):
        if path.suffix.lower() not in {".html", ".md", ".json", ".py"}:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for phrase in WITHDRAWN:
            assert phrase not in text, f"{path}: withdrawn claim present: {phrase!r}"


def test_the_geometry_result_is_stated_in_its_narrow_form() -> None:
    """docs/SPEC.md section 8.5: the conflict with Overath et al. must be on the page."""
    html = INDEX.read_text(encoding="utf-8")
    assert "Overath" in html, "the conflicting published result must be named"
    assert "multiplied by" in html or "multiplicative" in html
    assert "additive" in html or "linear" in html


def test_page_names_no_hit_binder_or_validated_design() -> None:
    """There are none, so the page may not imply otherwise."""
    html = INDEX.read_text(encoding="utf-8").lower()
    assert "has been made or tested" in html or "no design" in html


def test_page_loads_nothing_from_an_external_host() -> None:
    """It must work from file:// and inside a static site unchanged."""
    html = INDEX.read_text(encoding="utf-8")
    for match in re.findall(r'(?:src|href)="([^"]+)"', html):
        if match.startswith("#"):
            continue
        if match.startswith("http"):
            # Links out are fine; loading resources is not.
            assert f'src="{match}"' not in html, f"external resource loaded: {match}"
            continue
        assert not match.startswith("//"), f"protocol-relative resource: {match}"
    assert "<script" not in html.lower() or "src=" not in html.lower().split("<script")[1][:200]


def test_page_has_a_dark_mode_and_a_viewport() -> None:
    html = INDEX.read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in html
    assert 'name="viewport"' in html


def test_all_seven_figures_are_present_and_identical_to_the_study_copies() -> None:
    """Copied, never regenerated: a regenerated figure can disagree with its report."""
    for name, source_dir in EXPECTED_FIGURES.items():
        copied = FIGURES / name
        original = Path(source_dir) / name
        assert copied.is_file(), f"missing portfolio figure {name}"
        assert original.is_file(), f"missing study figure {original}"
        assert copied.read_bytes() == original.read_bytes(), (
            f"{name} differs from {original}; copy it rather than regenerating it"
        )


def test_every_figure_on_disk_is_referenced_by_the_page() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for path in FIGURES.glob("*.png"):
        assert f"figures/{path.name}" in html, f"{path.name} is unused"


def test_portfolio_readme_states_it_is_derived() -> None:
    readme = PORTFOLIO / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8").lower()
    assert "derived" in text
    assert "studies/" in text


def test_no_banned_term_appears_in_the_portfolio() -> None:
    """docs/SPEC.md section 12.3 rule 9, applied to the one directory that gets published."""
    terms_file = Path(".private/banned_terms.txt")
    if not terms_file.is_file():
        pytest.skip(".private/banned_terms.txt not present")
    terms = [t.strip().lower() for t in terms_file.read_text(encoding="utf-8").splitlines()]
    terms = [t for t in terms if t]
    if not terms:
        pytest.skip("no banned terms configured")
    for path in PORTFOLIO.rglob("*"):
        if path.suffix.lower() not in {".html", ".md", ".json", ".py"}:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for term in terms:
            assert term not in text, f"{path}: banned term present"

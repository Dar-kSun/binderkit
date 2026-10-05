# portfolio/

A derived view of `studies/`, built to be dropped onto a personal website.
**`studies/` is canonical.** Nothing in here is a source of truth, and nothing
in here should be edited by hand except the prose.

| File | What it is |
|---|---|
| `index.html` | The page. One self-contained file: all CSS inline, figures as relative paths, no external hosts, no JavaScript. Opens from `file://` and drops into a static site unchanged. |
| `numbers.json` | Every headline figure, machine-readable, each with the `studies/` file and key it came from. **Generated — do not edit.** |
| `build.py` | Generates `numbers.json`. Run `python -m portfolio.build` after re-running either study. |
| `SUMMARY.md` | The same argument in plain language, for a reader who will not open the page. |
| `figures/` | Byte-for-byte copies of the seven PNGs from `studies/*/figures/`. Copied, never regenerated: a regenerated figure can silently disagree with the report beside it. |

## The rule this directory exists under

**No number may appear here that is not already in `studies/`.**

That is enforced, not promised. `tests/test_portfolio.py` asserts:

- every entry in `numbers.json` still resolves to the same value from the
  `studies/` file its `source` field names;
- every number shown on the page carries a `data-n` attribute naming a key in
  `numbers.json`, and its displayed text equals that key's rendered value — so
  editing a figure in the HTML without changing the study fails the suite;
- no number appears in the page's visible text without that annotation, apart
  from a short allowlist of structural values (the confidence level, a cited
  paper's year) each carrying its reason;
- the seven figures are byte-identical to the study copies;
- no claim withdrawn in docs/SPEC.md §8.7 appears anywhere in this directory;
- the page loads nothing from an external host.

A portfolio that drifts from the study it summarises is worse than no
portfolio, because the entire claim of this repository is that its numbers are
reproducible from one command.

## Rebuilding

```bash
python -m studies.retrospective.report      # study 1
python -m studies.interface_geometry.report # study 2
python -m portfolio.build                   # refresh numbers.json
pytest -q tests/test_portfolio.py           # fails if the page now disagrees
cp studies/*/figures/*.png portfolio/figures/
```

If a study's numbers move, the portfolio test fails until both `numbers.json`
and the prose around each figure have been updated. That failure is the point.

## What is deliberately not here

- Any claim of a hit, a binder, or a validated design. There are none.
- Any claim that this question has not been asked before, and any unqualified
  statement that interface geometry adds nothing. Both were withdrawn; see
  `studies/retrospective/CHANGEstats.md` §9 and §10 for the exact wording and
  why it did not survive.
- Anything generated. The page is written, the numbers are computed.

"""Ranking (docs/SPEC.md section 6.4).

Never ranks on a single metric: the organisers state that selection does not
rely on one, and the conditional objectives are poorly captured by any single
score. The scheme is hard filters, then the challenge's stated objective
priority order, then interface confidence as a tie-break, then diversity-aware
selection so the final list is not N variants of one backbone.

Every design gets a one-line plain-English reason built from its actual
numbers, and the full table including rejected designs is emitted, because that
table is the evidence of judgement.
"""

from __future__ import annotations

import logging

import pandas as pd

from binderkit.config import SCHEMAS, Config
from binderkit.metrics import liability_flags

log = logging.getLogger(__name__)


def apply_hard_filters(
    metrics: pd.DataFrame,
    novelty: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Return a frame with `filter_passed` and `filter_reason` per design.

    Order matters for the reason text: novelty first, because a novelty failure
    is a rule violation rather than a quality judgement.
    """
    df = metrics.merge(
        novelty[["design_id", "passed", "reason", "best_seq_identity", "batch_cluster"]].rename(
            columns={"passed": "novelty_passed", "reason": "novelty_reason"}
        ),
        on="design_id",
        how="left",
    )

    liab = liability_flags(df, cfg.liabilities)
    sc_floor = cfg.rank.min_self_consistency_rmsd

    reasons: list[str] = []
    passed: list[bool] = []
    for i, row in df.iterrows():
        why: list[str] = []
        ok = True
        if not bool(row.get("novelty_passed", False)):
            ok = False
            why.append(f"novelty gate: {row.get('novelty_reason', 'failed')}")
        if bool(liab.iloc[i]):
            ok = False
            why.append(
                "liability cap breached ("
                f"unpaired_cys={row['n_unpaired_cys']}, "
                f"glyc_sequons={row['n_glyc_sequons']}, "
                f"net_charge={row['net_charge']:.2f}, "
                f"lowcomplexity_run={row['max_lowcomplexity_run']})"
            )
        rmsd = row.get("rmsd_sc")
        if rmsd is not None and rmsd == rmsd and rmsd > sc_floor:
            ok = False
            why.append(f"self-consistency RMSD {rmsd:.2f} A exceeds the {sc_floor} A floor")
        passed.append(ok)
        reasons.append("; ".join(why) if why else "passed all hard filters")

    df["filter_passed"] = passed
    df["filter_reason"] = reasons
    log.info("hard filters: %d/%d designs passed", int(sum(passed)), len(df))
    return df


def score_objectives(
    designs: pd.DataFrame,
    context: dict,
    objective_names: tuple[str, ...],
) -> pd.DataFrame:
    """Run every registered objective over every design. `objectives` schema."""
    from binderkit.objectives import REGISTRY

    rows: list[dict[str, object]] = []
    for d in designs.itertuples():
        for obj_name in objective_names:
            obj = REGISTRY.get(obj_name)
            if obj is None:
                continue
            s = obj.score(d, context)
            rows.append(
                {
                    "design_id": d.design_id,
                    "objective": obj_name,
                    "value": s.value,
                    "uncertainty": s.uncertainty,
                    "reason": s.reason,
                }
            )
    return pd.DataFrame(rows, columns=SCHEMAS["objectives"])


def rank(
    filtered: pd.DataFrame,
    objectives: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Produce the full ranking table, rejected designs included.

    Sorting is lexicographic over the challenge's objective priority order,
    which is the honest reading of "ranked in this order": a design that wins on
    objective 1 outranks one that merely wins on objective 2, rather than the
    two being blended into a single weighted number.
    """
    wide = objectives.pivot_table(
        index="design_id", columns="objective", values="value", aggfunc="first"
    ).reset_index()
    df = filtered.merge(wide, on="design_id", how="left")

    present = [o for o in cfg.rank.objective_order if o in df.columns]
    # "affinity" is represented by interface confidence, which is the only
    # affinity proxy available; it is also the documented tie-break.
    df["affinity_proxy"] = df["ipsae"].fillna(df["iptm"]).fillna(0.0)

    sort_cols = [*present, "affinity_proxy"]
    ascending = [False] * len(sort_cols)

    eligible = df[df["filter_passed"]].sort_values(sort_cols, ascending=ascending)
    rejected = df[~df["filter_passed"]]

    # Diversity-aware pass: cap how many designs come from one batch cluster.
    chosen: list[str] = []
    per_cluster: dict[object, int] = {}
    cap = cfg.novelty.max_per_cluster
    for row in eligible.itertuples():
        cluster = getattr(row, "batch_cluster", None)
        n = per_cluster.get(cluster, 0)
        if cfg.rank.diversity_aware and n >= cap:
            continue
        per_cluster[cluster] = n + 1
        chosen.append(row.design_id)
        if len(chosen) >= cfg.submission.max_designs:
            break

    order = {d: i + 1 for i, d in enumerate(chosen)}

    rows: list[dict[str, object]] = []
    for row in eligible.itertuples():
        did = row.design_id
        included = did in order
        parts = [
            f"{o}={getattr(row, o, float('nan')):.3f}"
            for o in present
            if getattr(row, o, None) is not None
        ]
        reason = (
            f"{'selected' if included else 'eligible but not selected'}: "
            + ", ".join(parts)
            + f", affinity proxy (ipSAE) {row.affinity_proxy:.3f}"
            + f", batch cluster {getattr(row, 'batch_cluster', 'n/a')}"
        )
        if not included and cfg.rank.diversity_aware:
            reason += f" - cluster already contributed {cap} design(s)"
        rows.append(
            {
                "design_id": did,
                "rank": order.get(did, pd.NA),
                "included": included,
                "reason": reason,
            }
        )
    for row in rejected.itertuples():
        rows.append(
            {
                "design_id": row.design_id,
                "rank": pd.NA,
                "included": False,
                "reason": f"rejected: {row.filter_reason}",
            }
        )

    out = pd.DataFrame(rows, columns=SCHEMAS["ranking"])
    out = out.sort_values("rank", na_position="last").reset_index(drop=True)
    log.info(
        "rank: %d selected, %d eligible-not-selected, %d rejected",
        len(chosen),
        len(eligible) - len(chosen),
        len(rejected),
    )
    return out


def filter_cascade_counts(
    designs: pd.DataFrame,
    novelty: pd.DataFrame,
    filtered: pd.DataFrame,
    ranking: pd.DataFrame,
) -> list[tuple[str, int]]:
    """Counts surviving each stage, for the METHODS.md filter cascade."""
    return [
        ("designs generated", len(designs)),
        ("passed novelty gate", int(novelty["passed"].sum())),
        ("passed all hard filters", int(filtered["filter_passed"].sum())),
        ("selected for submission", int(ranking["included"].sum())),
    ]

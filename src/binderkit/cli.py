"""Command-line interface (docs/SPEC.md section 14).

binderkit compute
binderkit run --challenge 01-egfr [--tier auto|A|B|C] [--resume] [--force]
binderkit package --challenge 01-egfr
binderkit validate <submission.csv>
binderkit study retrospective
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from binderkit import compute, stages
from binderkit.config import Config
from binderkit.novelty import Reference, gate, read_fasta
from binderkit.objectives.ortholog import epitope_conservation
from binderkit.package import package
from binderkit.provenance import Provenance
from binderkit.rank import apply_hard_filters, rank, score_objectives
from binderkit.target import TargetSpec, fetch_uniprot_sequence, prepare
from binderkit.validate import validate_csv

log = logging.getLogger("binderkit")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------


def cmd_compute(args: argparse.Namespace) -> int:
    info = compute.detect()
    tier, reasons = compute.select_tier(info)
    md = compute.render_markdown(info, tier, reasons)
    out = Path("docs/COMPUTE.md")
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        log.info("wrote %s", out)
    else:
        sys.stdout.write(md)
    log.info("tier=%s", tier)
    return 0


def _challenge_dir(challenge_id: str) -> Path:
    return Path("challenges") / challenge_id


def _load_config(args: argparse.Namespace) -> Config:
    path = _challenge_dir(args.challenge) / "config.yaml"
    cfg = Config.from_yaml(path) if path.is_file() else Config(challenge_id=args.challenge)
    if getattr(args, "tier", None):
        cfg.tier = args.tier
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    info = compute.detect()
    detected, reasons = compute.select_tier(info)
    tier = cfg.resolve_tier(detected)
    cfg = cfg.apply_tier(tier)
    cfg.run_id = args.run_id

    prov = Provenance(run_id=cfg.run_id)
    prov.hardware = {
        "gpu": info.gpu_name,
        "vram_mib": info.vram_mib,
        "cpu_cores": info.cpu_cores,
        "ram_gb": info.ram_gb,
        "free_disk_gb": info.free_disk_gb,
        "tier_detected": detected,
        "tier_used": tier,
        "tier_reasons": reasons,
    }
    prov.seeds = {
        "generate": cfg.generate.seed,
        "sequence": cfg.sequence.seed,
        "global": cfg.seed,
    }
    log.info("tier detected=%s used=%s", detected, tier)
    for r in reasons:
        log.info("  tier: %s", r)

    cdir = _challenge_dir(cfg.challenge_id)
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cfg.work_dir / "cache"

    # --- target -------------------------------------------------------
    spec_path = cdir / "target_spec.json"
    if spec_path.is_file() and args.resume and not args.force:
        spec = TargetSpec.from_json(spec_path)
        log.info("resume: reusing %s", spec_path)
    else:
        with prov.time("target"):
            spec = prepare(cfg.target, cache)
        spec.to_json(spec_path)

    # Epitope conservation across orthologs - real, and it bounds objective 2.
    conservation: float | None = None
    species = ["human"]
    try:
        for sp, acc in cfg.target.orthologs.items():
            orth = fetch_uniprot_sequence(acc, cache)
            frac, per_res = epitope_conservation(
                spec.sequence_uniprot, orth, spec.epitope_residues_uniprot
            )
            conservation = frac
            species.append(sp)
            spec.conservation = {k: float(v) for k, v in per_res.items()}
            log.info("epitope conservation human/%s = %.1f%%", sp, 100 * frac)
        spec.to_json(spec_path)
    except Exception as exc:  # noqa: BLE001 - a network failure must not stop the run
        log.warning("ortholog conservation unavailable: %s", exc)
        prov.mark_skipped("ortholog_conservation", f"{type(exc).__name__}: {exc}")

    # --- generate / sequence / fold ------------------------------------
    def staged(name: str, fn):  # noqa: ANN001, ANN202
        if args.resume and not args.force:
            cached = stages.read_stage(cfg.work_dir, cfg.run_id, name)
            if cached is not None:
                return cached
        with prov.time(name):
            df = fn()
        stages.write_stage(df, cfg.work_dir, cfg.run_id, name)
        return df

    backbones = staged("backbones", lambda: stages.generate(spec, cfg, prov))
    designs = staged("designs", lambda: stages.design_sequences(backbones, cfg, prov))
    folds = staged("folds", lambda: stages.fold(designs, cfg, prov))
    folded = stages.collapse_folds(folds)

    # --- metrics -------------------------------------------------------
    from binderkit.metrics import compute_all

    ss_map = dict(zip(backbones["backbone_id"], backbones["ss_fractions"], strict=True))
    with prov.time("metrics"):
        metrics = compute_all(designs, folded, ss_map)
    stages.write_stage(metrics, cfg.work_dir, cfg.run_id, "metrics")

    # --- novelty -------------------------------------------------------
    refs: list[Reference] = [
        Reference(f"{spec.name}_target", spec.sequence_uniprot),
    ]
    known_path = cdir / "known_binders.fasta"
    if known_path.is_file():
        refs += read_fasta(known_path, is_known_binder=True)
        log.info("loaded %d known binder reference(s)", len(refs) - 1)
    else:
        log.warning(
            "no known_binders.fasta for %s; the known-binder arm of the "
            "novelty gate has nothing to screen against",
            cfg.challenge_id,
        )
        prov.mark_skipped(
            "known_binder_screen",
            f"{known_path} absent, so no known binders were screened against",
        )
    with prov.time("novelty"):
        nov = gate(designs, cfg.novelty, refs)
    stages.write_stage(nov, cfg.work_dir, cfg.run_id, "novelty")

    # --- objectives / rank ---------------------------------------------
    context = {
        "target_spec": spec,
        "epitope_conservation": conservation,
        "species": species,
        "per_species_scores": {},
    }
    with prov.time("objectives"):
        objectives = score_objectives(designs, context, cfg.rank.objective_order)
    filtered = apply_hard_filters(metrics, nov, cfg)
    ranking = rank(filtered, objectives, cfg)
    stages.write_stage(ranking, cfg.work_dir, cfg.run_id, "ranking")

    # --- package -------------------------------------------------------
    result = package(
        cfg,
        cdir / "submission",
        designs,
        metrics,
        nov,
        objectives,
        filtered,
        ranking,
        spec,
        prov,
    )
    for w in result.warnings:
        log.warning("validate: %s", w)
    for e in result.errors:
        log.error("validate: %s", e)
    log.info("submission valid=%s rows=%d", result.ok, result.n_rows)
    return 0 if result.ok else 1


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_csv(Path(args.path))
    for w in result.warnings:
        log.warning("%s", w)
    for e in result.errors:
        log.error("%s", e)
    print(f"{'PASSED' if result.ok else 'FAILED'} ({result.n_rows} rows)")
    return 0 if result.ok else 1


def cmd_study(args: argparse.Namespace) -> int:
    if args.which == "retrospective":
        from studies.retrospective.run_study import main as study_main

        return study_main(force_download=args.force)
    if args.which == "geometry":
        from studies.interface_geometry.report import main as geom_main

        return geom_main()
    log.error("unknown study %r", args.which)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="binderkit", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compute", help="probe hardware and set the tier")
    c.add_argument("--write", action="store_true", help="write docs/COMPUTE.md")
    c.set_defaults(func=cmd_compute)

    r = sub.add_parser("run", help="run the pipeline for a challenge")
    r.add_argument("--challenge", required=True)
    r.add_argument("--tier", choices=["auto", "A", "B", "C"], default=None)
    r.add_argument("--run-id", default="dev")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="validate a submission CSV")
    v.add_argument("path")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("study", help="run a study")
    s.add_argument("which", choices=["retrospective", "geometry"])
    s.add_argument("--force", action="store_true", help="re-download source data")
    s.set_defaults(func=cmd_study)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

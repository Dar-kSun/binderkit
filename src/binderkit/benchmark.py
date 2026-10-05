"""Measured per-stage cost, for the compute request (docs/CLUSTER_REQUEST.md).

A compute request built on estimates is easy to dismiss. This records what each
stage actually costs on this machine: wall time, peak RSS, and peak VRAM where a
GPU stage runs. Stages that cannot run here are recorded as such, with the
reason, rather than guessed at.

Every number this emits is marked `measured`. Figures taken from published
sources are marked `cited` and carry their source, and the two are never mixed.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MB, or None if unavailable.

    Uses psutil, which reports peak working set on Windows and current RSS
    elsewhere. The hand-rolled ctypes route through psapi/kernel32 silently
    returned None on this machine, and a benchmark that quietly reports nothing
    is worse than one that says it could not measure.
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        return round((peak if peak is not None else info.rss) / 1024**2, 1)
    except Exception:  # noqa: BLE001
        return None


def _peak_vram_mb() -> float | None:
    """Peak VRAM this process has allocated, MB, or None without a GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:  # noqa: BLE001
        return None


@dataclass
class StageBenchmark:
    """One measured, skipped or failed stage."""

    stage: str
    status: str = "measured"  # measured | unavailable | failed
    seconds: float = float("nan")
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    n_items: int = 0
    note: str = ""

    @property
    def seconds_per_item(self) -> float:
        return self.seconds / self.n_items if self.n_items else float("nan")


@dataclass
class BenchmarkSuite:
    """Collected measurements, serialisable into the compute request."""

    hardware: dict[str, Any] = field(default_factory=dict)
    stages: list[StageBenchmark] = field(default_factory=list)

    @contextmanager
    def measure(self, stage: str, n_items: int = 1):
        """Time a stage and record peak memory. Records failures rather than raising."""
        rec = StageBenchmark(stage=stage, n_items=n_items)
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as exc:  # noqa: BLE001
            rec.status = "failed"
            rec.note = f"{type(exc).__name__}: {exc}"
            log.warning("benchmark %s failed: %s", stage, rec.note)
        finally:
            rec.seconds = round(time.perf_counter() - t0, 3)
            rec.peak_rss_mb = _peak_rss_mb()
            rec.peak_vram_mb = _peak_vram_mb()
            self.stages.append(rec)
            log.info(
                "%-26s %8.2fs  %s items  rss=%s MB vram=%s",
                stage,
                rec.seconds,
                rec.n_items,
                rec.peak_rss_mb,
                rec.peak_vram_mb,
            )

    def unavailable(self, stage: str, reason: str) -> None:
        """Record a stage that cannot run here, and why."""
        self.stages.append(StageBenchmark(stage=stage, status="unavailable", note=reason))
        log.info("%-26s UNAVAILABLE: %s", stage, reason)

    def to_dict(self) -> dict:
        return {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": self.hardware,
            "stages": [
                {
                    "stage": s.stage,
                    "status": s.status,
                    "seconds": s.seconds,
                    "seconds_per_item": s.seconds_per_item,
                    "n_items": s.n_items,
                    "peak_rss_mb": s.peak_rss_mb,
                    "peak_vram_mb": s.peak_vram_mb,
                    "note": s.note,
                }
                for s in self.stages
            ],
        }

    def write(self, path: Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    def markdown_table(self) -> str:
        rows = [
            "| Stage | Status | Wall time | Per item | Peak RAM | Peak VRAM |",
            "|---|---|---|---|---|---|",
        ]
        for s in self.stages:
            if s.status != "measured":
                rows.append(f"| {s.stage} | **{s.status}** | - | - | - | {s.note} |")
                continue
            per = f"{s.seconds_per_item:.3f}s" if s.n_items > 1 else "-"
            vram = f"{s.peak_vram_mb:.0f} MB" if s.peak_vram_mb else "n/a (CPU)"
            rss = f"{s.peak_rss_mb:.0f} MB" if s.peak_rss_mb is not None else "not measured"
            rows.append(f"| {s.stage} | measured | {s.seconds:.1f}s | {per} | {rss} | {vram} |")
        return "\n".join(rows)


def run_suite(out: Path = Path("docs/benchmarks.json")) -> BenchmarkSuite:
    """Benchmark every stage that can run on this machine."""
    import pandas as pd

    from binderkit import compute as compute_mod
    from binderkit import search as seq_search
    from binderkit import stages as stage_mod
    from binderkit.config import Config
    from binderkit.geometry import compute_interface, identify_binder_chain, parse_mmcif
    from binderkit.provenance import Provenance
    from binderkit.target import TargetSpec

    suite = BenchmarkSuite()
    info = compute_mod.detect()
    tier, _ = compute_mod.select_tier(info)
    suite.hardware = {
        "gpu": info.gpu_name,
        "vram_mib": info.vram_mib,
        "cpu_cores": info.cpu_cores,
        "ram_gb": info.ram_gb,
        "free_disk_gb": info.free_disk_gb,
        "os": info.os_name,
        "python": info.python,
        "tier": tier,
        "torch": info.torch_version or "not installed",
        "cuda_available": info.cuda_available,
    }

    cfg = Config().apply_tier("C")
    prov = Provenance(run_id="bench")

    # --- target preparation (cached, so this measures parsing not network) ---
    spec_path = Path("challenges/01-egfr/target_spec.json")
    if spec_path.is_file():
        with suite.measure("target_prep (cached)", 1):
            TargetSpec.from_json(spec_path)
    else:
        suite.unavailable("target_prep", "no cached target spec")

    # --- fixture generation and sequence design ---
    spec = TargetSpec.from_json(spec_path) if spec_path.is_file() else None
    if spec is not None:
        with suite.measure("generate (fixture)", cfg.generate.n_per_length * 3) as rec:
            backbones = stage_mod.generate(spec, cfg, prov)
            rec.n_items = len(backbones)
        with suite.measure("sequence design (fixture)", 1) as rec:
            designs = stage_mod.design_sequences(backbones, cfg, prov)
            rec.n_items = len(designs)
        with suite.measure("co-fold (MOCK, not a prediction)", 1) as rec:
            folds = stage_mod.fold(designs, cfg, prov)
            rec.n_items = len(folds)

        from binderkit.metrics import compute_all

        with suite.measure("sequence metrics", 1) as rec:
            compute_all(
                designs,
                stage_mod.collapse_folds(folds),
                dict(zip(backbones.backbone_id, backbones.ss_fractions, strict=True)),
            )
            rec.n_items = len(designs)

        # --- real PDB database search ---
        if seq_search.available():
            seqs = dict(zip(designs.design_id, designs.sequence, strict=True))
            with suite.measure("novelty: MMseqs2 vs 1.1M PDB seqs", len(seqs)):
                seq_search.search(seqs)
        else:
            suite.unavailable("novelty: MMseqs2", "binary or database absent")

    # --- interface geometry on real coordinates ---
    struct_dir = Path("work/hf_cache/structures")
    cifs = sorted(struct_dir.glob("*.cif"))[:50] if struct_dir.is_dir() else []
    if cifs:
        summary = pd.read_csv("work/hf_cache/tables/design_summary.csv", low_memory=False)
        seq_by = dict(zip(summary.full_name, summary.sequence, strict=True))
        with suite.measure("interface geometry (real coords)", len(cifs)):
            for c in cifs:
                st = parse_mmcif(c)
                b, t = identify_binder_chain(st, seq_by.get(c.stem, ""))
                compute_interface(st, b, t, c.stem)
    else:
        suite.unavailable("interface geometry", "no cached structures")

    # --- the study itself ---
    summary_csv = Path("work/hf_cache/tables/design_summary.csv")
    if summary_csv.is_file():
        from studies.retrospective import stats as st_mod
        from studies.retrospective.run_study import build_table

        df = build_table(summary_csv)
        y = df["binder_final"].to_numpy(dtype=int)
        g = df["target"].to_numpy()
        s = df["ipsae_mean"].to_numpy(dtype=float)
        with suite.measure("paired bootstrap (2000 reps)", 2000):
            st_mod.paired_bootstrap_difference(
                y, s, df["binder_length"].to_numpy(dtype=float), g, n_boot=2000
            )
        with suite.measure("AUROC (rank-sum)", 20000):
            for _ in range(20000):
                st_mod.auroc(y, s)

    # --- GPU stages that cannot run here ---
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            suite.unavailable("GPU stages", "torch present but CUDA unavailable")
    except ImportError:
        suite.unavailable(
            "backbone generation (RFdiffusion)", "torch not installed; needs ~12-24 GB VRAM"
        )
        suite.unavailable(
            "sequence design (ProteinMPNN)", "torch not installed; CPU-capable once present"
        )
        suite.unavailable("co-folding (Boltz-2)", "torch not installed; needs ~16-24 GB VRAM")

    suite.write(out)
    log.info("wrote %s", out)
    return suite

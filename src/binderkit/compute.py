"""Hardware detection and tier selection (docs/SPEC.md section 1).

Two independent rules can force a lower tier: available VRAM, and the free-disk
guard. Both are evaluated and both are reported, so the chosen tier is never a
silent judgement call.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from binderkit.config import Tier

#: Below this much free disk, Tier A/B weight downloads are not attempted and
#: the tier drops by one (docs/SPEC.md section 1).
DISK_GUARD_GB = 50.0
#: Abort an in-progress download if it would leave less than this.
DISK_ABORT_FLOOR_GB = 10.0


@dataclass
class ComputeInfo:
    """What this machine actually has."""

    gpu_name: str | None = None
    vram_mib: int | None = None
    driver: str | None = None
    cpu_cores: int | None = None
    ram_gb: float | None = None
    free_disk_gb: float | None = None
    python: str = ""
    os_name: str = ""
    torch_version: str | None = None
    cuda_available: bool | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def vram_gib(self) -> float | None:
        return None if self.vram_mib is None else self.vram_mib / 1024.0


def _nvidia_smi() -> tuple[str | None, int | None, str | None]:
    """Query the first GPU. Returns (name, VRAM MiB, driver) or Nones."""
    if shutil.which("nvidia-smi") is None:
        return None, None, None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    line = out.stdout.strip().splitlines()
    if not line:
        return None, None, None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 3:
        return None, None, None
    try:
        return parts[0], int(float(parts[1])), parts[2]
    except ValueError:
        return parts[0], None, parts[2]


def _ram_gb() -> float | None:
    """Total RAM in GB.

    `free -g` from docs/SPEC.md section 1 does not exist in Git Bash on native
    Windows, so this uses a per-platform probe instead. The substitution is
    recorded in docs/COMPUTE.md.
    """
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return round(int(out.stdout.strip()) / 1024**3, 1)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    try:
        page_size = __import__("os").sysconf("SC_PAGE_SIZE")
        n_pages = __import__("os").sysconf("SC_PHYS_PAGES")
        return round(page_size * n_pages / 1024**3, 1)
    except (OSError, ValueError, AttributeError):
        return None


def detect(path: Path | None = None) -> ComputeInfo:
    """Probe the machine. Never raises; missing values stay None."""
    name, vram, driver = _nvidia_smi()
    info = ComputeInfo(
        gpu_name=name,
        vram_mib=vram,
        driver=driver,
        cpu_cores=__import__("os").cpu_count(),
        ram_gb=_ram_gb(),
        python=platform.python_version(),
        os_name=platform.platform(),
    )
    usage = shutil.disk_usage(Path(path or "."))
    info.free_disk_gb = round(usage.free / 1024**3, 1)

    try:
        import torch  # noqa: PLC0415 - optional, probed deliberately

        info.torch_version = torch.__version__
        info.cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        info.torch_version = None
        info.cuda_available = None
    return info


def tier_from_vram(info: ComputeInfo) -> Tier:
    """Tier implied by VRAM alone (docs/SPEC.md section 1 table)."""
    gib = info.vram_gib
    if gib is None:
        return "C"
    if gib >= 24.0:
        return "A"
    if gib >= 8.0:
        return "B"
    return "C"


def _drop_one(tier: Tier) -> Tier:
    return {"A": "B", "B": "C", "C": "C"}[tier]


def select_tier(info: ComputeInfo) -> tuple[Tier, list[str]]:
    """Choose the tier, applying both the VRAM table and the disk guard.

    Returns the tier and the human-readable reasons behind it. Both rules are
    always reported, even when they agree, so that a borderline machine does not
    look like a judgement call.
    """
    reasons: list[str] = []
    vram_tier = tier_from_vram(info)
    if info.vram_gib is None:
        reasons.append("No GPU detected by nvidia-smi -> Tier C on the VRAM rule.")
    else:
        reasons.append(
            f"VRAM {info.vram_mib} MiB = {info.vram_gib:.3f} GiB -> Tier {vram_tier} "
            "on the VRAM rule."
        )

    tier = vram_tier
    free = info.free_disk_gb
    if free is not None and free < DISK_GUARD_GB and vram_tier in ("A", "B"):
        tier = _drop_one(vram_tier)
        reasons.append(
            f"Free disk {free} GB is below the {DISK_GUARD_GB:.0f} GB guard, so the "
            f"tier drops from {vram_tier} to {tier} and Tier A/B weight downloads "
            "are not attempted."
        )
    elif free is not None and free < DISK_GUARD_GB:
        reasons.append(
            f"Free disk {free} GB is below the {DISK_GUARD_GB:.0f} GB guard; the tier "
            "is already C so there is nothing to drop."
        )
    else:
        reasons.append(f"Free disk {free} GB clears the {DISK_GUARD_GB:.0f} GB guard.")

    return tier, reasons


def disk_would_abort(required_gb: float, path: Path | None = None) -> bool:
    """True if downloading `required_gb` would breach the abort floor."""
    free = shutil.disk_usage(Path(path or ".")).free / 1024**3
    return (free - required_gb) < DISK_ABORT_FLOOR_GB


def render_markdown(info: ComputeInfo, tier: Tier, reasons: list[str]) -> str:
    """Render docs/COMPUTE.md content for `binderkit compute`."""
    lines = [
        "# Compute environment",
        "",
        f"**TIER: {tier}**",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| GPU | {info.gpu_name or 'none detected'} |",
        f"| VRAM | {info.vram_mib} MiB"
        + (f" = {info.vram_gib:.3f} GiB |" if info.vram_gib else " |"),
        f"| Driver | {info.driver or 'n/a'} |",
        f"| torch | {info.torch_version or 'not installed'} "
        f"(cuda_available={info.cuda_available}) |",
        f"| CPU cores | {info.cpu_cores} |",
        f"| RAM | {info.ram_gb} GB |",
        f"| Free disk | {info.free_disk_gb} GB |",
        f"| Python | {info.python} |",
        f"| OS | {info.os_name} |",
        "",
        "## Tier determination",
        "",
    ]
    lines += [f"{i}. {r}" for i, r in enumerate(reasons, 1)]
    return "\n".join(lines) + "\n"

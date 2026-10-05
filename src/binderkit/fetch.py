"""Network fetching that never blocks on a dead host.

Session 1 lost about 9.5 hours of wall clock to a stall, so every fetch here is
bounded: a hard timeout, three retries with backoff, then fall back to cache and
return a result that says what happened. Nothing in this module raises on a
network failure. Callers check `FetchResult.ok` and carry on.

Cache layout is `work/cache/<sha256-of-url>/payload` plus a `meta.json`
alongside it, so a cached entry is self-describing and a URL maps to exactly one
directory regardless of how ugly the URL is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 600  # 10 minutes, per the session brief
DEFAULT_RETRIES = 3
BACKOFF_BASE_S = 2.0
CACHE_ROOT = Path("work/cache")
USER_AGENT = "binderkit/0.1 (research; contact via repository)"


@dataclass
class FetchResult:
    """What happened, in enough detail to put in a report."""

    url: str
    path: Path | None = None
    ok: bool = False
    from_cache: bool = False
    status: str = ""
    size_bytes: int = 0
    attempts: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> str | None:
        """A `BLOCKED: network` marker for the journal, or None if fine."""
        return None if self.ok else f"BLOCKED: network - {self.url} - {self.status}"


def cache_dir_for(url: str) -> Path:
    """Deterministic cache directory for a URL."""
    return CACHE_ROOT / hashlib.sha256(url.encode()).hexdigest()[:32]


def cached_path(url: str) -> Path | None:
    """Return the cached payload for `url`, or None if not cached."""
    payload = cache_dir_for(url) / "payload"
    if payload.is_file() and payload.stat().st_size > 0:
        return payload
    return None


def head_size(url: str, timeout: int = 60) -> int | None:
    """Content-Length via HEAD, or None.

    Used to record a download's size before committing disk to it. Note that
    some hosts refuse HEAD (rest.uniprot.org answers 403), so None means
    "unknown", never "absent".
    """
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001 - probing only
        return None


def fetch(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    force: bool = False,
    label: str = "",
) -> FetchResult:
    """Fetch `url` into the cache. Never raises.

    Returns a `FetchResult`. On total failure, falls back to any cached copy and
    reports `from_cache=True`; if there is no cached copy either, `ok` is False
    and the caller marks the dependent work `BLOCKED: network`.
    """
    result = FetchResult(url=url)
    cdir = cache_dir_for(url)
    payload = cdir / "payload"

    if not force:
        hit = cached_path(url)
        if hit is not None:
            result.path, result.ok, result.from_cache = hit, True, True
            result.size_bytes = hit.stat().st_size
            result.status = "cache hit"
            log.info("cache hit  %-50s %s", label or url[-50:], _human(result.size_bytes))
            return result

    cdir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        result.attempts = attempt
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                data = resp.read()
            payload.write_bytes(data)
            result.path, result.ok = payload, True
            result.size_bytes = len(data)
            result.status = f"downloaded on attempt {attempt}"
            (cdir / "meta.json").write_text(
                json.dumps(
                    {
                        "url": url,
                        "label": label,
                        "size_bytes": len(data),
                        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.info("downloaded %-50s %s", label or url[-50:], _human(len(data)))
            return result
        except Exception as exc:  # noqa: BLE001 - every failure mode is handled the same
            msg = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            result.errors.append(msg)
            log.warning("fetch failed %s - %s", label or url, msg)
            if attempt < retries:
                time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))

    # Exhausted retries: fall back to cache if there is one.
    hit = cached_path(url)
    if hit is not None:
        result.path, result.ok, result.from_cache = hit, True, True
        result.size_bytes = hit.stat().st_size
        result.status = "network failed, served stale cache"
        log.warning("served stale cache for %s", label or url)
        return result

    result.status = f"failed after {retries} attempts, no cache"
    log.error("BLOCKED: network - %s", label or url)
    return result


def fetch_many(items: list[tuple[str, str]], **kwargs) -> dict[str, FetchResult]:
    """Fetch several (label, url) pairs. Returns label -> FetchResult.

    Keeps going past failures by construction, since `fetch` never raises.
    """
    out: dict[str, FetchResult] = {}
    for label, url in items:
        out[label] = fetch(url, label=label, **kwargs)
    return out


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n / 1:.1f}{unit}"
        n /= 1024.0
    return f"{n}B"


def summarise(results: dict[str, FetchResult]) -> str:
    """A table of what was fetched, for the journal."""
    lines = ["| item | status | size | attempts |", "|---|---|---|---|"]
    for label, r in results.items():
        size = f"{r.size_bytes / 1024 / 1024:.1f} MB" if r.size_bytes else "-"
        lines.append(f"| {label} | {r.status} | {size} | {r.attempts} |")
    return "\n".join(lines)


def materialize(dest_root: Path, prefix: str = "") -> dict[str, Path]:
    """Copy cached payloads to readable paths derived from their labels.

    The content-addressed cache stays the source of truth; this just gives the
    files names a human and pandas can use. Only entries whose label looks like
    a relative path are materialised, and an existing identical file is left
    alone so repeat calls are cheap.
    """
    import shutil

    out: dict[str, Path] = {}
    root = Path(dest_root)
    if not CACHE_ROOT.is_dir():
        return out
    for meta_file in CACHE_ROOT.glob("*/meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        label = str(meta.get("label", ""))
        if not label or label.startswith(("http://", "https://")) or ".." in label:
            continue
        if prefix and not label.startswith(prefix):
            continue
        payload = meta_file.parent / "payload"
        if not payload.is_file():
            continue
        target = root / label
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.stat().st_size != payload.stat().st_size:
            shutil.copy2(payload, target)
        out[label] = target
    return out

# Compute environment

**TIER: C** — no generation. CPU-only path, everything heavy mocked behind
interfaces, and all of §8. See [Tier determination](#tier-determination).

Probed 2026-10-04T22:31+0530. Reproduce with `binderkit compute`.

## Hardware

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8188 MiB = **7.996 GiB** |
| Driver | 595.97 |
| CUDA (via torch) | not determinable — torch not installed |
| CPU cores | 20 |
| RAM | 23.7 GB |
| **Free disk** | **46 GB** on `C:` (475 G total, 91% used) |
| OS | Windows 11 Home Single Language 10.0.26200, native |
| Shell | MINGW64 (Git Bash) 3.5.4 + PowerShell 7 |
| Python | 3.13.14 (`.venv`, pip 26.2.1) |

`free -g` from §1 is Linux-only and does not exist in Git Bash on native
Windows; RAM was read with `Get-CimInstance Win32_ComputerSystem` instead.
The substitution is recorded in `JOURNAL.md` under Decisions.

## Network

Reachable (checked with `curl`):

| Host | Status |
|---|---|
| pypi.org | 200 |
| huggingface.co | 200 |
| files.rcsb.org | 200 (verified a real 815 KB PDB download) |
| rest.uniprot.org | **GET 200**, HEAD 403 |
| proteinbase.com | 200 |
| github.com | 200 |

`rest.uniprot.org` rejects `HEAD` with 403 but serves `GET` normally — a
probe that only sends `HEAD` will wrongly conclude UniProt is down. Verified
by fetching `P00533.fasta` (EGFR) successfully.

Network access works, so §2 and §8 fetching are both viable and the §8.4
fallback study is not needed on network grounds.

## Tier determination

Tier C is determined **twice over, independently**:

1. **VRAM.** 8188 MiB is 7.996 GiB. §1 sets Tier B at ">=8 GB VRAM" and Tier C
   at "<8 GB". The card is 4 MiB short, so it reads as Tier C. On its own this
   is an uncomfortably thin margin to hang a night on.
2. **Disk guard, which settles it.** §1: "if free disk < 50 GB, do not attempt
   Tier A/B weight downloads; drop one tier and record why." Free disk is
   **46 GB**, under the 50 GB threshold. So even reading the card generously as
   Tier B, the guard drops it to Tier C.

Because the two rules agree, the tier does not depend on the 4 MiB margin and
no judgement call is being smuggled in here.

The disk guard is the binding constraint in practice: RFdiffusion, ProteinMPNN
and Boltz-2 weights together would not fit in 46 GB with the 10 GB abort
floor §1 requires, regardless of VRAM.

Tier is nonetheless a **config override**, not a hardcoded branch, so
`binderkit run --tier B` works if the author disagrees or frees disk. Per §0.2,
Tier C is "a perfectly good night": §8 is the headline deliverable and needs no
GPU.

## Time budget

8 h (§0.3 default; no other figure was supplied). Phase boundaries are
timestamped in `JOURNAL.md`.

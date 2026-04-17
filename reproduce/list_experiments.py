#!/usr/bin/env python3
"""
list_experiments.py — Inventory R³ paper reproduction targets.

Scans the autonomous-learning-library (ALL) sub-tree for:

  1. ``examples/experiment*.py``     — hand-written ALL example drivers.
  2. ``all/scripts/train_*.py``      — installed CLI entry points (``all-classic``,
                                       ``all-atari``, ``all-continuous`` …).
  3. ``all/r3/*.py`` (excluding tests) — the R³ runtime modules that are wired
                                       into the agents.
  4. ``all/presets/<domain>/<agent>.py`` — the agent presets (DQN, DDQN, PPO,
                                       Rainbow …) that the train scripts can
                                       launch.

For each scanned file we extract the leading docstring (if any), the default
``--frames`` from any ``argparse`` declaration, and any obvious environment /
agent hints. Everything is printed as a Markdown table to ``stdout`` so it can
be redirected into ``reproduce/EXPERIMENTS.md`` or pasted into a GitHub PR.

Usage
-----

  python reproduce/list_experiments.py            # full markdown report
  python reproduce/list_experiments.py --json     # raw JSON instead

The script intentionally has no third-party dependencies (only stdlib) so it
runs on the Jetson Orin venv and CI alike.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_ROOT = REPO_ROOT / "autonomous-learning-library"


@dataclass
class ScriptInfo:
    relpath: str
    category: str
    docstring: str = ""
    default_frames: Optional[str] = None
    envs: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "relpath": self.relpath,
            "category": self.category,
            "docstring": self.docstring,
            "default_frames": self.default_frames,
            "envs": self.envs,
            "agents": self.agents,
            "notes": self.notes,
        }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _module_docstring(path: Path) -> str:
    src = _read_text(path)
    if not src:
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    doc = ast.get_docstring(tree) or ""
    # Keep only the first non-empty line — the table cell stays compact.
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


_FRAMES_RE = re.compile(
    r"""['"]--frames['"][^)]*?default\s*=\s*([^,)\s]+)""",
    re.DOTALL,
)
_FRAMES_KW_RE = re.compile(
    r"""default_frames\s*=\s*([^,)\s]+)""",
    re.DOTALL,
)
_ENV_HINTS_RE = re.compile(
    r"""(?:CartPole-v[01]|Acrobot-v1|MountainCar(?:Continuous)?-v0|"""
    r"""LunarLander(?:Continuous)?-v[23]|PongNoFrameskip-v4|BreakoutNoFrameskip-v4|"""
    r"""SeaquestNoFrameskip-v4|Pendulum-v1|HalfCheetah-v[24]|Hopper-v[24]|Ant-v[24])""",
)
_AGENT_HINTS_RE = re.compile(
    r"""\b(dqn|ddqn|c51|rainbow|a2c|ppo|vac|vpg|vqn|vsarsa|sac|ddpg|td3)\b""",
    re.IGNORECASE,
)


def _scan_argparse(path: Path) -> Optional[str]:
    src = _read_text(path)
    if not src:
        return None
    m = _FRAMES_RE.search(src)
    if m:
        return m.group(1).strip()
    m = _FRAMES_KW_RE.search(src)
    if m:
        return m.group(1).strip()
    return None


def _scan_hints(path: Path) -> tuple[List[str], List[str]]:
    src = _read_text(path)
    if not src:
        return ([], [])
    envs = sorted(set(_ENV_HINTS_RE.findall(src)))
    agents = sorted({a.lower() for a in _AGENT_HINTS_RE.findall(src)})
    return envs, agents


def _iter_scripts() -> Iterable[ScriptInfo]:
    if not ALL_ROOT.exists():
        return

    # 1. examples/experiment*.py
    examples_dir = ALL_ROOT / "examples"
    if examples_dir.exists():
        for p in sorted(examples_dir.glob("experiment*.py")):
            envs, agents = _scan_hints(p)
            yield ScriptInfo(
                relpath=str(p.relative_to(REPO_ROOT)),
                category="example",
                docstring=_module_docstring(p),
                default_frames=_scan_argparse(p),
                envs=envs,
                agents=agents,
            )

    # 2. all/scripts/train_*.py
    scripts_dir = ALL_ROOT / "all" / "scripts"
    if scripts_dir.exists():
        for p in sorted(scripts_dir.glob("train_*.py")):
            envs, agents = _scan_hints(p)
            yield ScriptInfo(
                relpath=str(p.relative_to(REPO_ROOT)),
                category="cli-entry",
                docstring=_module_docstring(p),
                default_frames=_scan_argparse(p),
                envs=envs,
                agents=agents,
                notes=f"console_script: all-{p.stem.replace('train_', '')}",
            )

    # 3. all/r3/*.py (skip tests)
    r3_dir = ALL_ROOT / "all" / "r3"
    if r3_dir.exists():
        for p in sorted(r3_dir.glob("*.py")):
            if p.name.endswith("_test.py") or p.name == "__init__.py":
                continue
            yield ScriptInfo(
                relpath=str(p.relative_to(REPO_ROOT)),
                category="r3-module",
                docstring=_module_docstring(p),
            )

    # 4. agent presets
    presets_root = ALL_ROOT / "all" / "presets"
    if presets_root.exists():
        for domain_dir in sorted(p for p in presets_root.iterdir() if p.is_dir()):
            for p in sorted(domain_dir.glob("*.py")):
                if p.name == "__init__.py":
                    continue
                yield ScriptInfo(
                    relpath=str(p.relative_to(REPO_ROOT)),
                    category=f"preset:{domain_dir.name}",
                    docstring=_module_docstring(p),
                    agents=[p.stem],
                )


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def _render_markdown(scripts: List[ScriptInfo]) -> str:
    lines: List[str] = []
    lines.append("# R³ paper-reproduction script inventory")
    lines.append("")
    lines.append(f"Generated by `reproduce/list_experiments.py` over `{ALL_ROOT.relative_to(REPO_ROOT)}`.")
    lines.append("")

    # Group by category for readability.
    by_cat: dict[str, List[ScriptInfo]] = {}
    for s in scripts:
        by_cat.setdefault(s.category, []).append(s)

    for cat in sorted(by_cat):
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| Path | Default frames | Env hints | Agent hints | Description |")
        lines.append("|------|----------------|-----------|-------------|-------------|")
        for s in by_cat[cat]:
            desc = s.docstring or s.notes or ""
            lines.append(
                "| `{path}` | {frames} | {envs} | {agents} | {desc} |".format(
                    path=_md_escape(s.relpath),
                    frames=_md_escape(s.default_frames or "—"),
                    envs=_md_escape(", ".join(s.envs) or "—"),
                    agents=_md_escape(", ".join(s.agents) or "—"),
                    desc=_md_escape(desc),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args(argv)

    if not ALL_ROOT.exists():
        print(f"error: cannot find {ALL_ROOT}", file=sys.stderr)
        return 2

    scripts = list(_iter_scripts())
    if args.json:
        json.dump([s.to_dict() for s in scripts], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_render_markdown(scripts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

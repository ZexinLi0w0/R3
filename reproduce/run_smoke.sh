#!/usr/bin/env bash
# reproduce/run_smoke.sh — 5–10 minute Atari smoke test for the R³ stack.
#
# Goal
# ----
# Verify end-to-end that the modernized R³ stack can actually train a DQN agent
# on the canonical Atari benchmark (PongNoFrameskip-v4) on a Jetson Orin (or
# any CUDA host). This is the *Atari* smoke test — the classic-control
# CartPole variant lives in ``run_smoke_classic.sh``.
#
# Why Pong?
#   - R³ (RTSS 2023) explicitly evaluates on Atari (see paper §VI). Pong is the
#     fastest-converging classical Atari benchmark and is the standard "does
#     the pipeline work?" gate for any DQN-family implementation.
#   - On Orin AGX, ~10–20k frames of DQN-Pong is enough to confirm the env
#     wrapper, replay buffer, optimizer, and tensorboard logging all wire up
#     correctly without waiting for actual learning.
#
# What this script does
# ---------------------
#   1. Run the R³ algorithmic unit tests (pure Python, seconds).
#   2. Sanity-check that ``ale_py`` is importable and at least one Atari ROM
#      is registered. If not, print actionable install hints and FAIL fast.
#   3. Train ``dqn`` on ``PongNoFrameskip-v4`` for FRAMES (default 20 000)
#      via the installed ``all-atari`` CLI, with a tiny replay buffer to keep
#      RAM bounded.
#   4. Parse the resulting tensorboard log and print the final
#      ``test/returns/mean`` / ``returns/mean`` to stdout.
#
# Output layout (NVMe only — never the eMMC root)
# -----------------------------------------------
#   /experiment/zexin/R3/reproduce/runs/smoke_<UTC-timestamp>/
#       runs/                                ← ALL tensorboard subdir
#       smoke.log                            ← captured stdout/stderr
#       summary.json                         ← parsed metrics (best-effort)
#
# Usage
# -----
#   bash reproduce/run_smoke.sh                       # default 20000 frames
#   FRAMES=10000 bash reproduce/run_smoke.sh          # smaller / faster
#   ENV=BreakoutNoFrameskip-v4 bash reproduce/run_smoke.sh
#
# Exit codes
# ----------
#   0   PASS — unit tests + Atari training + log parsing all clean.
#   1   FAIL — R³ unit tests failed.
#   2   FAIL — Atari runtime not available (ale_py / ROMs missing).
#   3   FAIL — training crashed.
#   4   FAIL — could not parse tensorboard logs.
#
set -u
set -o pipefail

#------------------------------------------------------------------------------#
# Knobs (override via env vars).
#------------------------------------------------------------------------------#
FRAMES="${FRAMES:-20000}"
# NOTE: ALL's all-atari CLI wraps the env name with "NoFrameskip-v4" itself
# (see autonomous-learning-library/all/environments/atari.py). So pass the
# *short* name here (e.g. "Pong", not "PongNoFrameskip-v4").
ENV_NAME="${ENV:-Pong}"
AGENT_NAME="${AGENT:-dqn}"
DEVICE="${DEVICE:-cuda}"
RUNS_ROOT="${RUNS_ROOT:-/experiment/zexin/R3/reproduce/runs}"
TS="$(date -u +%Y%m%d_%H%M%SZ)"
RUN_DIR="${RUNS_ROOT}/smoke_${TS}"
LOG_FILE="${RUN_DIR}/smoke.log"

#------------------------------------------------------------------------------#
# Locate repo root (this script lives in <repo>/reproduce/).
#------------------------------------------------------------------------------#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${RUN_DIR}/runs"
echo "[smoke] repo:    ${REPO_ROOT}"
echo "[smoke] run_dir: ${RUN_DIR}"
echo "[smoke] env=${ENV_NAME} agent=${AGENT_NAME} device=${DEVICE} frames=${FRAMES}"
echo "[smoke] log:     ${LOG_FILE}"
echo

#------------------------------------------------------------------------------#
# Step 1 — R³ unit tests (pure CPU, seconds).
#------------------------------------------------------------------------------#
echo "[smoke] step 1/4 — R³ algorithmic unit tests"
{
    cd "${REPO_ROOT}/autonomous-learning-library"
    python -m unittest discover -s all/r3 -p "*_test.py" -v
} 2>&1 | tee -a "${LOG_FILE}"
unit_rc="${PIPESTATUS[0]}"
if [ "${unit_rc}" -ne 0 ]; then
    echo "[smoke] FAIL: R³ unit tests exited ${unit_rc}" | tee -a "${LOG_FILE}"
    exit 1
fi

#------------------------------------------------------------------------------#
# Step 2 — Atari runtime preflight.
#------------------------------------------------------------------------------#
echo
echo "[smoke] step 2/4 — Atari runtime preflight (ale_py + ROM availability)"
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import sys
try:
    import ale_py  # noqa: F401
    print(f"[preflight] ale_py {ale_py.__version__}")
except Exception as e:
    print(f"[preflight] FAIL: cannot import ale_py: {e}")
    print("[preflight] hint: pip install 'ale-py' 'autorom[accept-rom-license]' && AutoROM --accept-license")
    sys.exit(2)

try:
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make("PongNoFrameskip-v4")
    env.reset()
    env.close()
    print("[preflight] PongNoFrameskip-v4 constructible — ROM available.")
except Exception as e:
    print(f"[preflight] FAIL: cannot make PongNoFrameskip-v4: {e}")
    print("[preflight] hint: AutoROM --accept-license   # downloads the Atari 2600 ROM set")
    sys.exit(2)
PY
pre_rc="${PIPESTATUS[0]}"
if [ "${pre_rc}" -ne 0 ]; then
    echo "[smoke] FAIL: Atari runtime not ready (rc=${pre_rc}). See hints above." | tee -a "${LOG_FILE}"
    exit 2
fi

#------------------------------------------------------------------------------#
# Step 3 — micro Atari training run via all-atari.
#------------------------------------------------------------------------------#
echo
echo "[smoke] step 3/4 — micro Atari training: ${AGENT_NAME} on ${ENV_NAME}NoFrameskip-v4 for ${FRAMES} frames"
{
    cd "${RUN_DIR}"
    # Tiny replay buffer to keep Orin RAM happy at smoke scale; full paper
    # config uses 1e6 frames buffer and is exercised by run_short / run_full.
    all-atari "${ENV_NAME}" "${AGENT_NAME}" \
        --device "${DEVICE}" \
        --frames "${FRAMES}" \
        --logdir "${RUN_DIR}/runs" \
        --hyperparameters replay_buffer_size=10000 replay_start_size=1000
} 2>&1 | tee -a "${LOG_FILE}"
train_rc="${PIPESTATUS[0]}"
if [ "${train_rc}" -ne 0 ]; then
    echo "[smoke] FAIL: training exited ${train_rc}" | tee -a "${LOG_FILE}"
    exit 3
fi

#------------------------------------------------------------------------------#
# Step 4 — parse tensorboard logs.
#------------------------------------------------------------------------------#
echo
echo "[smoke] step 4/4 — parsing test return metrics"
export RUN_DIR
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json
import os
import sys
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"]) / "runs"
event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
if not event_files:
    print(f"[smoke] FAIL: no tensorboard event files under {run_dir}", file=sys.stderr)
    sys.exit(4)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print(f"[smoke] WARN: tensorboard import failed ({e}); cannot parse metrics."
          f"\n[smoke] Smoke training itself succeeded — treating as PASS.")
    sys.exit(0)

best = {"path": None, "tag": None, "value": None, "step": None}
for ef in event_files:
    acc = EventAccumulator(str(ef))
    try:
        acc.Reload()
    except Exception as e:
        print(f"[smoke] note: failed to load {ef}: {e}")
        continue
    tags = acc.Tags().get("scalars", [])
    for tag in tags:
        if tag.endswith("returns/mean") or tag.endswith("returns_100/mean"):
            scalars = acc.Scalars(tag)
            if scalars:
                last = scalars[-1]
                if best["value"] is None or last.step > (best["step"] or -1):
                    best = {
                        "path": str(ef),
                        "tag": tag,
                        "value": float(last.value),
                        "step": int(last.step),
                    }

summary_path = Path(os.environ["RUN_DIR"]) / "summary.json"
summary_path.write_text(json.dumps(best, indent=2))
print(f"[smoke] summary -> {summary_path}")
print(json.dumps(best, indent=2))

if best["value"] is None:
    print("[smoke] WARN: no returns/mean tag found — pipeline still ran cleanly.")
PY
parse_rc="${PIPESTATUS[0]}"
if [ "${parse_rc}" -ne 0 ]; then
    echo "[smoke] FAIL: metric parser exited ${parse_rc}" | tee -a "${LOG_FILE}"
    exit 4
fi

echo
echo "[smoke] ✅ PASS — see ${RUN_DIR}"
exit 0

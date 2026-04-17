#!/usr/bin/env bash
# reproduce/run_short.sh — 30–60 minute "short paper-validation" Atari run.
#
# Goal
# ----
# Train DQN on PongNoFrameskip-v4 for enough frames (default 250 000) that we
# expect to see early signs of learning (returns climbing out of the random
# baseline of ≈-21 toward ≈-15..-10) on a Jetson Orin AGX. This is *not* a
# full paper reproduction — that lives in ``run_full_paper.sh``. It IS a strong
# smoke test that the modernized R³ stack converges in the same direction as
# the published RTSS 2023 numbers.
#
# Designed to run fully detached
# ------------------------------
# The recommended invocation is:
#
#   tmux new-session -d -s r3-short \
#       "source /experiment/zexin/venvs/r3/bin/activate && \
#        cd /experiment/zexin/R3 && \
#        bash reproduce/run_short.sh 2>&1"
#
# The script itself also accepts ``--background`` which double-forks via
# ``setsid`` + ``nohup``, so it survives an SSH disconnect even without tmux.
#
# Output layout (NVMe only — never the eMMC root)
# -----------------------------------------------
#   /experiment/zexin/R3/reproduce/runs/short_<UTC-timestamp>/
#       runs/                    ← ALL tensorboard subdir
#       short.log                ← captured stdout/stderr
#       progress.json            ← parsed periodic snapshots (best-effort)
#       LATEST -> ...            ← convenience symlink the user can tail
#
# Resume / inspect
# ----------------
#   tmux attach -t r3-short
#   tail -F /experiment/zexin/R3/reproduce/runs/short_*/short.log | head
#   tensorboard --logdir /experiment/zexin/R3/reproduce/runs/short_<ts>/runs
#
set -u
set -o pipefail

#------------------------------------------------------------------------------#
# Knobs.
#------------------------------------------------------------------------------#
FRAMES="${FRAMES:-250000}"
# NOTE: ALL's all-atari CLI wraps the env name with "NoFrameskip-v4" itself,
# so pass the *short* name (e.g. "Pong", not "PongNoFrameskip-v4").
ENV_NAME="${ENV:-Pong}"
AGENT_NAME="${AGENT:-dqn}"
DEVICE="${DEVICE:-cuda}"
RUNS_ROOT="${RUNS_ROOT:-/experiment/zexin/R3/reproduce/runs}"
TS="$(date -u +%Y%m%d_%H%M%SZ)"
RUN_DIR="${RUNS_ROOT}/short_${TS}"
LOG_FILE="${RUN_DIR}/short.log"
LATEST_LINK="${RUNS_ROOT}/short_LATEST"

#------------------------------------------------------------------------------#
# Optional --background self-daemonize (when tmux is not used).
#------------------------------------------------------------------------------#
if [ "${1:-}" = "--background" ]; then
    shift
    mkdir -p "${RUN_DIR}"
    echo "[short] re-launching detached; logs -> ${LOG_FILE}"
    setsid nohup bash "$0" "$@" >"${LOG_FILE}" 2>&1 < /dev/null &
    echo "[short] background pid: $!"
    exit 0
fi

#------------------------------------------------------------------------------#
# Locate repo root.
#------------------------------------------------------------------------------#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${RUN_DIR}/runs"
ln -sfn "${RUN_DIR}" "${LATEST_LINK}" || true

echo "[short] repo:    ${REPO_ROOT}"
echo "[short] run_dir: ${RUN_DIR}"
echo "[short] env=${ENV_NAME} agent=${AGENT_NAME} device=${DEVICE} frames=${FRAMES}"
echo "[short] log:     ${LOG_FILE}"
echo "[short] symlink: ${LATEST_LINK} -> ${RUN_DIR}"
echo "[short] progress: tail -F ${LOG_FILE}"
echo

#------------------------------------------------------------------------------#
# Quick preflight (don't re-run the full unit suite — smoke owns that).
#------------------------------------------------------------------------------#
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import sys
try:
    import ale_py, gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make("PongNoFrameskip-v4")
    env.reset(); env.close()
    print("[short] preflight: ale_py + Pong ROM OK")
except Exception as e:
    print(f"[short] preflight FAIL: {e}")
    sys.exit(2)
PY
pre_rc="${PIPESTATUS[0]}"
if [ "${pre_rc}" -ne 0 ]; then
    echo "[short] FAIL: preflight" | tee -a "${LOG_FILE}"
    exit 2
fi

#------------------------------------------------------------------------------#
# Launch the training run.
#------------------------------------------------------------------------------#
echo "[short] starting training; this is expected to take 30–60 minutes on Orin AGX"
START_TS="$(date +%s)"
{
    cd "${RUN_DIR}"
    # Replay buffer kept at 100k for short run (paper uses 1e6 for full).
    # ``replay_start_size`` is bumped above the smoke value so the agent has
    # enough random data before learning begins.
    all-atari "${ENV_NAME}" "${AGENT_NAME}" \
        --device "${DEVICE}" \
        --frames "${FRAMES}" \
        --logdir "${RUN_DIR}/runs" \
        --hyperparameters \
            replay_buffer_size=100000 \
            replay_start_size=10000
} 2>&1 | tee -a "${LOG_FILE}"
train_rc="${PIPESTATUS[0]}"
END_TS="$(date +%s)"
echo "[short] training exit code: ${train_rc}; wall-clock: $((END_TS - START_TS))s" | tee -a "${LOG_FILE}"

if [ "${train_rc}" -ne 0 ]; then
    echo "[short] FAIL: training exited ${train_rc}" | tee -a "${LOG_FILE}"
    exit 3
fi

#------------------------------------------------------------------------------#
# Final metric parse.
#------------------------------------------------------------------------------#
export RUN_DIR
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json, os, sys
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"]) / "runs"
events = sorted(run_dir.rglob("events.out.tfevents.*"))
if not events:
    print("[short] WARN: no event files; skipping parse")
    sys.exit(0)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print(f"[short] WARN: tensorboard not available ({e}); skipping parse")
    sys.exit(0)

snapshots = []
for ef in events:
    acc = EventAccumulator(str(ef))
    try:
        acc.Reload()
    except Exception:
        continue
    for tag in acc.Tags().get("scalars", []):
        if tag.endswith("returns/mean") or tag.endswith("returns_100/mean"):
            for s in acc.Scalars(tag):
                snapshots.append({
                    "tag": tag, "step": int(s.step), "value": float(s.value),
                })

snapshots.sort(key=lambda s: (s["tag"], s["step"]))
out = Path(os.environ["RUN_DIR"]) / "progress.json"
out.write_text(json.dumps(snapshots, indent=2))
print(f"[short] wrote {out} ({len(snapshots)} samples)")

if snapshots:
    last = snapshots[-1]
    print(f"[short] last sample: tag={last['tag']} step={last['step']} value={last['value']:.3f}")
PY

echo
echo "[short] ✅ PASS — see ${RUN_DIR}"
exit 0

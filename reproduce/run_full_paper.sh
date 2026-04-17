#!/usr/bin/env bash
# reproduce/run_full_paper.sh — driver for the full RTSS 2023 R³ reproduction.
#
# This is the *catalog* + *driver* for every experiment that maps to a figure
# or table in the published paper:
#
#   Li et al., "R³: On-device Real-Time Deep Reinforcement Learning for
#   Autonomous Robotics," IEEE RTSS 2023. arXiv:2308.15039
#
# WHAT IT DOES
# ------------
#   * With ``--list``        : prints the catalog (env / agent / frames /
#                              estimated wall-time / target figure) and exits 0.
#   * With ``--dry-run``     : prints the exact ``all-atari`` invocations that
#                              would run, but does not actually launch them.
#   * With no flags          : runs every IN-SCOPE experiment serially.
#   * With ``--only NAME``   : runs only the named experiment from the catalog.
#
# Out-of-scope reproductions
# --------------------------
# The DonkeyCar / MUSHR-DL case study (paper §VII) requires a separate x86 PC
# running the Unity ``donkey_sim`` simulator; the Jetson is the *agent*, not
# the simulator. Single-Orin reproduction is therefore IMPOSSIBLE for those
# experiments and they are intentionally excluded here. See ``reproduce/README.md``
# section "Skipped Reproductions" for the full rationale and the manual setup
# you need if you want to run them yourself.
#
# BUDGET WARNING
# --------------
# Even the in-scope (Atari only) catalog below sums to multiple GPU-days on
# Orin AGX. The default ``--frames`` per experiment matches the paper's 10M
# frames/run × N seeds per env. Do not launch this casually; prefer ``--list``
# or ``--dry-run`` first, then pick one experiment with ``--only`` to start.
#
# Output layout (NVMe only — never the eMMC root)
# -----------------------------------------------
#   /experiment/zexin/R3/reproduce/runs/full_<UTC-timestamp>/
#       <experiment-name>/runs/   ← per-experiment tensorboard subdir
#       <experiment-name>/run.log ← per-experiment captured output
#       INDEX.md                  ← summary of all attempts
#
set -u
set -o pipefail

#------------------------------------------------------------------------------#
# Catalog
# -------
# Each entry is "name|env|agent|frames|seeds|figure|hours_estimate|notes".
# ``hours_estimate`` is rough Orin AGX wall-clock per seed.
# Add new experiments here as the codebase grows; ``--list`` will pick them up.
#------------------------------------------------------------------------------#
CATALOG=(
    "atari-pong-dqn|PongNoFrameskip-v4|dqn|10000000|3|Fig.6 / Tab.II|14|Canonical Atari baseline; converges to ~+20"
    "atari-breakout-dqn|BreakoutNoFrameskip-v4|dqn|10000000|3|Fig.6 / Tab.II|18|Atari benchmark; ~150 game-score by 10M"
    "atari-seaquest-dqn|SeaquestNoFrameskip-v4|dqn|10000000|3|Fig.6 / Tab.II|18|Atari benchmark; sparse reward"
    "atari-pong-rainbow|PongNoFrameskip-v4|rainbow|10000000|2|Fig.7 (algo ablation)|16|R³ vs vanilla on Rainbow"
    "atari-pong-c51|PongNoFrameskip-v4|c51|10000000|2|Fig.7 (algo ablation)|16|R³ vs vanilla on C51"
)

OUT_OF_SCOPE=(
    "donkeycar-sac : requires Unity DonkeySim on x86 PC; Orin acts only as RL agent. See README#skipped-reproductions."
    "donkeycar-ppo : same as above."
    "mushr-dl      : pre-trained .h5 + DonkeySim required; out of scope for this reproducer."
)

#------------------------------------------------------------------------------#
# Defaults / knobs.
#------------------------------------------------------------------------------#
DEVICE="${DEVICE:-cuda}"
RUNS_ROOT="${RUNS_ROOT:-/experiment/zexin/R3/reproduce/runs}"
TS="$(date -u +%Y%m%d_%H%M%SZ)"
ROOT_DIR="${RUNS_ROOT}/full_${TS}"
INDEX="${ROOT_DIR}/INDEX.md"

MODE="run"
ONLY=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --list)    MODE="list"; shift ;;
        --dry-run) MODE="dry";  shift ;;
        --only)    ONLY="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 64 ;;
    esac
done

#------------------------------------------------------------------------------#
# --list mode.
#------------------------------------------------------------------------------#
if [ "${MODE}" = "list" ]; then
    echo "# In-scope experiments (Atari, single-Orin reproducible)"
    printf "\n%-22s %-28s %-8s %-10s %-5s %-22s %-6s  %s\n" \
        "NAME" "ENV" "AGENT" "FRAMES" "SEEDS" "PAPER FIGURE" "HOURS" "NOTES"
    printf -- "-%.0s" {1..140}; echo
    for entry in "${CATALOG[@]}"; do
        IFS='|' read -r name env agent frames seeds fig hours notes <<<"${entry}"
        printf "%-22s %-28s %-8s %-10s %-5s %-22s %-6s  %s\n" \
            "${name}" "${env}" "${agent}" "${frames}" "${seeds}" "${fig}" "${hours}" "${notes}"
    done
    echo
    echo "# Out-of-scope (require external setup; NOT runnable here)"
    for s in "${OUT_OF_SCOPE[@]}"; do
        echo "  - ${s}"
    done
    exit 0
fi

#------------------------------------------------------------------------------#
# Run / dry-run.
#------------------------------------------------------------------------------#
mkdir -p "${ROOT_DIR}"
{
    echo "# R³ full-paper reproduction batch"
    echo
    echo "Started: $(date -u --iso-8601=seconds)"
    echo "Device : ${DEVICE}"
    echo "Mode   : ${MODE}${ONLY:+ (only=${ONLY})}"
    echo
    echo "| Name | Env | Agent | Frames | Seeds | Figure | Status |"
    echo "|------|-----|-------|--------|-------|--------|--------|"
} > "${INDEX}"

run_one() {
    local name="$1" env="$2" agent="$3" frames="$4" seeds="$5" fig="$6"
    local exp_dir="${ROOT_DIR}/${name}"
    mkdir -p "${exp_dir}/runs"
    local exp_log="${exp_dir}/run.log"
    local status="PENDING"

    for ((seed = 1; seed <= seeds; seed++)); do
        local cmd=(all-atari "${env}" "${agent}"
            --device "${DEVICE}"
            --frames "${frames}"
            --logdir "${exp_dir}/runs/seed_${seed}")
        echo "[full] ${name} seed=${seed}: ${cmd[*]}"
        if [ "${MODE}" = "dry" ]; then
            continue
        fi
        if "${cmd[@]}" 2>&1 | tee -a "${exp_log}"; then
            status="OK (last seed=${seed})"
        else
            status="FAIL (seed=${seed})"
            break
        fi
    done

    [ "${MODE}" = "dry" ] && status="DRY-RUN"
    echo "| ${name} | ${env} | ${agent} | ${frames} | ${seeds} | ${fig} | ${status} |" >> "${INDEX}"
}

for entry in "${CATALOG[@]}"; do
    IFS='|' read -r name env agent frames seeds fig hours notes <<<"${entry}"
    if [ -n "${ONLY}" ] && [ "${ONLY}" != "${name}" ]; then
        continue
    fi
    run_one "${name}" "${env}" "${agent}" "${frames}" "${seeds}" "${fig}"
done

echo
echo "[full] index -> ${INDEX}"
exit 0

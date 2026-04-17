# R³ Paper Reproduction Guide

This folder is the **single entry point** for reproducing the experiments from:

> Zexin Li, Aritra Samanta, Yufei Li, Andrea Soltoggio, Hyoseung Kim, Cong Liu.
> **R³: On-Device Real-Time Deep Reinforcement Learning for Autonomous Robotics.**
> *IEEE Real-Time Systems Symposium (RTSS) 2023*, pp. 131–144.
> [arXiv:2308.15039](https://arxiv.org/abs/2308.15039) · [DOI 10.1109/RTSS59052.2023.00021](https://doi.org/10.1109/RTSS59052.2023.00021)

If you only want to know whether the modernized stack works, run
`bash reproduce/run_smoke.sh`. Everything else here is a graduated path from
"5-minute smoke test" to "full multi-day paper reproduction".

---

## 1. System requirements

| Component        | Tested / recommended                                            |
|------------------|------------------------------------------------------------------|
| Hardware         | **NVIDIA Jetson AGX Orin** (also runs on x86 + CUDA GPU)         |
| OS               | Ubuntu 22.04 (JetPack 6.2)                                       |
| JetPack          | **6.2** (CUDA 12.6, cuDNN 9.x)                                   |
| Python           | 3.10 (ships with JetPack 6.2)                                    |
| PyTorch          | 2.8.x (Jetson AI Lab wheels — see main `README.md` quick-install)|
| Gymnasium        | ≥ 0.29                                                           |
| ALE / Atari ROMs | `ale-py` ≥ 0.10 + ROMs registered via `AutoROM --accept-license` |
| Disk             | All artifacts go on `/experiment` (NVMe). The eMMC root has < 10 GB free — never write logs/checkpoints to `~`. |

The complete install recipe lives in the project root [`README.md`](../README.md).
The reproduction scripts assume the venv at `/experiment/zexin/venvs/r3/`
already exists and has the modernized stack installed.

### One-liner environment activation (Jetson Orin)

```bash
source /experiment/zexin/venvs/r3/bin/activate
cd /experiment/zexin/R3
```

### Atari ROMs (one-time, only if the smoke preflight fails)

```bash
pip install 'ale-py' 'autorom[accept-rom-license]'
AutoROM --accept-license
```

---

## 2. Output convention

**Every** script in this folder writes under a single tree on the NVMe SSD:

```
/experiment/zexin/R3/reproduce/runs/
├── smoke_<UTC-timestamp>/    ← run_smoke.sh
├── short_<UTC-timestamp>/    ← run_short.sh
├── full_<UTC-timestamp>/     ← run_full_paper.sh
└── short_LATEST -> ...       ← convenience symlink
```

Override the root with `RUNS_ROOT=/some/other/path` if you really must, but
**never** point it at `~/` or `/home/...` on the Orin — the eMMC root will
fill up and brick the box. See [`AGENTS.md`](../AGENTS.md) for the full disk
layout rule.

---

## 3. The four scripts

| Script                       | Wall-clock         | Purpose                                                                  |
|------------------------------|--------------------|--------------------------------------------------------------------------|
| `run_smoke.sh`               | 5–10 min           | End-to-end pipeline gate on **Atari Pong** (≈20k frames). Exit 0 = green.|
| `run_smoke_classic.sh`       | 3–5 min            | Same idea but on CartPole-v1; useful when you don't have Atari ROMs yet. |
| `run_short.sh`               | 30–60 min          | "Does R³ start learning?" — DQN/Pong, 250k frames, daemonizable.         |
| `run_full_paper.sh`          | days               | Catalog + driver for all in-scope paper experiments. Use `--list` first. |
| `list_experiments.py`        | seconds            | Inventory of every experiment script and preset under ALL.               |

### 3.1 Smoke test — `run_smoke.sh`

```bash
bash reproduce/run_smoke.sh
# or
FRAMES=10000 bash reproduce/run_smoke.sh
ENV=Breakout bash reproduce/run_smoke.sh   # short name; ALL appends NoFrameskip-v4
```

What it checks, in order:

1. R³ algorithmic unit tests (`autonomous-learning-library/all/r3/*_test.py`).
2. ALE / ROM availability (Pong is constructible).
3. A real `all-atari` training run for `FRAMES` frames.
4. Tensorboard log parse: prints final `returns/mean`.

Exit codes: `0` PASS, `1` unit-test fail, `2` ALE/ROM missing, `3` training
crash, `4` log-parse failure.

### 3.2 Short paper validation — `run_short.sh`

```bash
# Recommended: detach via tmux so SSH disconnects don't kill the run.
tmux new-session -d -s r3-short \
    "source /experiment/zexin/venvs/r3/bin/activate && \
     cd /experiment/zexin/R3 && \
     bash reproduce/run_short.sh"

# Inspect progress without attaching to the tmux pane:
tail -F /experiment/zexin/R3/reproduce/runs/short_LATEST/short.log

# Attach to the tmux pane:
tmux attach -t r3-short
# (detach with Ctrl-b d)

# Tensorboard (from any host that can reach the Orin on port 6006):
tensorboard --logdir /experiment/zexin/R3/reproduce/runs/short_LATEST/runs --bind_all
```

The script also accepts `--background`, which `setsid+nohup`-daemonizes itself
without needing `tmux`. Use that if `tmux` is not available.

### 3.3 Full paper reproduction — `run_full_paper.sh`

This is a *catalog* and a *driver*. It is **not** intended to be run blindly —
the full catalog is multiple GPU-days. Always start with `--list`.

```bash
bash reproduce/run_full_paper.sh --list      # inventory + estimates
bash reproduce/run_full_paper.sh --dry-run   # print exact commands, do not run
bash reproduce/run_full_paper.sh --only atari-pong-dqn   # one experiment, all seeds
```

Each experiment in the catalog maps to a figure or table in the RTSS 2023
paper. The current (Atari-only) catalog:

| Name                  | Env                       | Agent   | Frames     | Seeds | Paper figure              |
|-----------------------|---------------------------|---------|------------|-------|---------------------------|
| `atari-pong-dqn`      | `Pong` (→ `PongNoFrameskip-v4`)         | dqn     | 10 000 000 | 3     | Fig. 6 / Table II         |
| `atari-breakout-dqn`  | `Breakout` (→ `BreakoutNoFrameskip-v4`) | dqn     | 10 000 000 | 3     | Fig. 6 / Table II         |
| `atari-seaquest-dqn`  | `Seaquest` (→ `SeaquestNoFrameskip-v4`) | dqn     | 10 000 000 | 3     | Fig. 6 / Table II         |
| `atari-pong-rainbow`  | `Pong`                                  | rainbow | 10 000 000 | 2     | Fig. 7 (algorithm ablation)|
| `atari-pong-c51`      | `Pong`                                  | c51     | 10 000 000 | 2     | Fig. 7 (algorithm ablation)|

### 3.4 Inventory — `list_experiments.py`

```bash
python reproduce/list_experiments.py        # markdown
python reproduce/list_experiments.py --json # machine-readable
```

Scans `autonomous-learning-library/{examples,all/scripts,all/r3,all/presets}`
and emits a per-category table with paths, default frame counts, env hints,
and agent hints.

---

## 4. Skipped reproductions

The original RTSS 2023 paper also contains an autonomous-driving case study
built on the [`gym-donkeycar`](https://github.com/tawnkramer/gym-donkeycar)
simulator. Those experiments are **out of scope** for this on-Orin reproducer
because they require additional hosts and assets that the Jetson alone cannot
provide:

- **DonkeyCar / DonkeySim experiments** require a separate x86 / Ubuntu PC
  running the Unity-based `donkey_sim` binary. The Jetson is the RL agent only;
  the simulator process is heavy and Linux-x86-only at the version pinned by
  the paper.
- **MUSHR-DL** depends on pre-trained `.h5` models *and* the same DonkeySim
  binary. There is no single-Orin path.

If you specifically need to reproduce those numbers, follow the legacy install
section at the bottom of the project [`README.md`](../README.md), set up the
PC + simulator, and use the scripts under `MUSHR-DL/`. They are intentionally
omitted from `run_full_paper.sh`.

---

## 5. R³ runtime modules vs. agent presets — current status

The R³ algorithmic core (memory budget, deadline, batch-size control, replay
buffer resize) lives in
[`autonomous-learning-library/all/r3/`](../autonomous-learning-library/all/r3/).
It is unit-tested but on the modernized branch it is **not yet wired** as a
default hook into the upstream agent presets (DQN, Rainbow, etc.). The
reproduction scripts therefore exercise:

- **Vanilla ALL** training to confirm the modernized stack converges normally,
  and
- The R³ unit tests to confirm the algorithmic equations match the paper.

Once the wiring PRs land on `main`, this guide will be updated to flip the
appropriate `--hyperparameters r3=true` flag (or equivalent) on each
reproduction. Until then, the catalog above represents the upper-bound DQN
baselines; R³'s memory/time savings are demonstrated by the unit suite.

---

## 6. Troubleshooting

| Symptom                                         | Likely cause / fix                                                                  |
|-------------------------------------------------|--------------------------------------------------------------------------------------|
| `cannot import ale_py`                          | `pip install 'ale-py' 'autorom[accept-rom-license]'` then `AutoROM --accept-license`|
| `Environment PongNoFrameskip-v4 doesn't exist`  | ROMs not installed; same fix as above.                                              |
| Out-of-memory on Orin                           | Lower `replay_buffer_size` via `--hyperparameters replay_buffer_size=50000`.        |
| Disk fills up                                   | You wrote to `~/` instead of `/experiment/`. Move data to `/experiment/zexin/...`.  |
| Run died on SSH disconnect                      | Use `tmux` (recommended) or `bash reproduce/run_short.sh --background`.             |
| `all-atari: command not found`                  | The venv is not activated; `source /experiment/zexin/venvs/r3/bin/activate`.        |

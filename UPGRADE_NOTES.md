# R³ Modernization: Upgrade Notes

## Background and Goals
The original RTSS 2023 implementation of R³ was built on a dependency stack that is now heavily outdated and unmaintained: PyTorch 1.13, Python 3.8, Gym 0.15.3, and JetPack 5.0.2. To keep the project usable on modern Jetson hardware (like Jetson AGX Orin and Orin Nano) which requires newer JetPack releases, this `upgrade/pytorch-2.5-gymnasium` branch modernizes the stack. The goal is to ensure security, compatibility, and ease of installation on contemporary NVIDIA JetPack 6.x platforms.

## Dependency Compatibility Matrix

| Component | Legacy Release (v1.0.0-legacy) | Modernized Branch (`upgrade/...`) |
|-----------|--------------------------------|-----------------------------------|
| Python | 3.8.10 | 3.10 |
| PyTorch | 1.13.0 | 2.8.x (`>=2.5`) |
| RL API | gym 0.15.3 | Gymnasium `>=0.29` |
| autonomous-learning-library| 0.8.1 | 0.9.1 |
| JetPack (Jetson) | 5.0.2 | 6.2 |

## Installation on JetPack 6.2 (Jetson AGX Orin / Orin Nano)
JetPack 6.2 comes with Python 3.10 by default.

1. **Create a virtual environment:**
   ```bash
   python3.10 -m venv ~/.venvs/r3
   source ~/.venvs/r3/bin/activate
   ```
2. **Install PyTorch (>=2.8) for JetPack 6.2:**
   NVIDIA hosts wheels via Jetson AI Lab. For CUDA 12.6, run:
   ```bash
   pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 "torch>=2.8" torchvision torchaudio
   ```
   > **Note:** the JP6 / CUDA 12.6 index on Jetson AI Lab currently publishes
   > `torch>=2.8` wheels only. The 2.5.x and 2.6.x lines were never built for
   > this index, so the previous `torch>=2.5,<2.6` constraint is impossible to
   > satisfy on a Jetson with this wheel source. We have relaxed the floor to
   > `>=2.5` and verified `2.8.x` works end-to-end.

   *(Check https://pypi.jetson-ai-lab.io/ for the latest URLs depending on your exact minor JetPack/CUDA version).*
3. **Install other dependencies:**
   ```bash
   # Install modernized ALL
   cd autonomous-learning-library && pip install -e . && cd ..
   
   # Install gym-donkeycar (upstream main branch is recommended)
   cd gym-donkeycar && pip install -e . && cd ..
   
   # Install MUSHR-DL requirements
   cd MUSHR-DL && pip install -r requirements.txt && cd ..
   ```

## Key API Changes
- **Gymnasium (`reset` and `step`):**
  Gymnasium introduced breaking changes compared to `gym 0.15`. 
  - `env.reset()` now returns `(observation, info)` instead of just `observation`.
  - `env.step(action)` now returns `(observation, reward, terminated, truncated, info)` instead of `(observation, reward, done, info)`.
  The upgraded `autonomous-learning-library` (v0.9.1) handles the new Gymnasium API internally.

- **PyTorch `torch.load`:**
  To mitigate security warnings and comply with PyTorch >=2.0, `torch.load()` calls have been updated to explicitly include `weights_only=False` where needed for legacy pickled model formats.

## Known Risks and Rollback
Since we are bumping multiple major versions across the stack (Python, PyTorch, Gym), some implicit behaviors or random seeds might differ slightly from the original paper's runs. The code in `autonomous-learning-library/all/r3/` remains unmodified structurally, but the underlying neural network execution relies on PyTorch 2.8 (Jetson AI Lab JP6/cu126 floor).

**Rollback Plan:**
If you encounter breaking issues, need to reproduce exact paper numbers, or are stuck on JetPack 5.x, you should roll back to the frozen legacy release.
👉 [v1.0.0-legacy Release](https://github.com/ZexinLi0w0/R3/releases/tag/v1.0.0-legacy)

## Known Issues

- **PyTorch 2.x `SourceChangeWarning` when loading legacy `MUSHR-DL/*.h5`
  checkpoints.** The `steering.h5`, `bezier.h5`, `bezier_0.5.h5`, and
  `bezier_1.5.h5` files were pickled with the legacy (non-zip) `torch.save`
  format and reference the `Net` / `Bezier` classes defined in
  `MUSHR-DL/pytorch_model.py`. They still load correctly under PyTorch 2.8 +
  `weights_only=False`, but PyTorch will emit a `SourceChangeWarning` for
  every stdlib `nn.*` module whose source has shifted between the original
  PyTorch 1.13 build and 2.8 (`Sequential`, `BatchNorm2d`, `Conv2d`, `ELU`,
  `MaxPool2d`, `Dropout`, `Linear`). The warnings are cosmetic; inference
  works. To re-pickle without warnings, retrain or call `torch.save` again
  on the loaded model on the new stack.
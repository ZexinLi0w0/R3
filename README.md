# RTSS'23 R^3
Official Implementation of RTSS 2023 paper [R^3: On-device Real-Time Deep Reinforcement Learning for Autonomous Robotics](https://arxiv.org/pdf/2308.15039).

> ## 🚧 Modernized Branch (`upgrade/pytorch-2.5-gymnasium`)
>
> **You are reading the modernization branch of R³.** It targets a contemporary stack:
>
> | Component | Modernized branch | Original (legacy) |
> |-----------|------------------|-------------------|
> | Python | **3.10** | 3.8.10 |
> | PyTorch | **2.8.x** (`>=2.5`) | 1.13.0 |
> | RL API | **Gymnasium `>=0.29`** | gym 0.15.3 |
> | autonomous-learning-library | **0.9.1** | 0.8.1 |
> | JetPack (Jetson) | **6.2** | 5.0.2 |
>
> **Why?** JetPack 5.0.2 / PyTorch 1.13 / gym 0.15 are unmaintained. The modernized branch upgrades the dependency stack so the project keeps running on current Jetson hardware (AGX Orin / Orin Nano) without security or compatibility holes.
>
> **What stayed the same?** All R³ research code under `autonomous-learning-library/all/r3/` is preserved bit-for-bit. The rest of `autonomous-learning-library/` is a drop-in replacement of upstream `cpnota/autonomous-learning-library` v0.9.1.
>
> **Need the exact paper artifact?** Use the [v1.0.0-legacy release](https://github.com/ZexinLi0w0/R3/releases/tag/v1.0.0-legacy) (frozen RTSS 2023 snapshot).
>
> **Quick install on JetPack 6.2 (Jetson AGX Orin / Orin Nano):**
> ```bash
> # 1. Create env (Python 3.10 ships with JetPack 6.2)
> python3.10 -m venv ~/.venvs/r3 && source ~/.venvs/r3/bin/activate
>
> # 2. Install PyTorch wheel from Jetson AI Lab (PyTorch 2.8+ on JP6 / CUDA 12.6)
> #    Browse https://pypi.jetson-ai-lab.io/jp6/cu126 for the wheel that matches
> #    your JetPack 6.2 / CUDA 12.6 build. Note: the JP6/cu126 index currently
> #    publishes torch >= 2.8 only; older 2.5/2.6 wheels are not available.
> pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
>     "torch>=2.8" torchvision torchaudio
>
> # 3. Install the rest
> cd autonomous-learning-library && pip install -e . && cd ..
> cd gym-donkeycar && pip install -e . && cd ..
> cd MUSHR-DL && pip install -r requirements.txt && cd ..
> ```
>
> 📖 Full migration story: see [`UPGRADE_NOTES.md`](./UPGRADE_NOTES.md).

# Overview
R^3 is a comprehensive solution designed to optimize on-device real-time DRL training by managing the intricate balance between timing and algorithm performance, specifically in memory-constrained environments. The approach involves co-optimizing two pivotal parameters of DRL training:

Evaluation: we implement R^3 based on Autonomous Learning Library(ALL) library which involves multiple well-known deep reinforcement learning algorithms and well-built benchmark.

Practical case study: We adopt gym-donkeycar simulator to conduct deep reinforcement learning via gym APIs on DonkeyCar simulator.

# Quick Start

For the modernized stack (PyTorch 2.8 + Gymnasium + Python 3.10 on JetPack 6.x),
follow the **Quick install** snippet in the Modernized Branch banner above, then
jump to step 2.

> ℹ️ The legacy x86 / JetPack 5.0.2 install instructions (Python 3.8, PyTorch
> 1.13, gym 0.15.3) have moved to the [Legacy installation
> (deprecated)](#legacy-installation-deprecated) appendix at the bottom of this
> README. They are kept for reproducibility of the RTSS 2023 paper artifact only.

2. Run the simulator on the x86 machine:
Example (on Ubuntu 20.04)
```
cd <path to DonkeySimLinux directory>
./donkey_sim.x86_64
```

3. Run the code to locally/remote control an autonomous driving car with a pre-trained steering model.
```bash
cd MUSHR-DL
python3 run_sim.py --dataset_name=testing --model=steering --test=True
python3 run_sim.py --dataset_name=testing --model=steering_dave2 --test=True --model_path=$MODEL_PATH
```

4. Installing Autonomous Learning Library
We need a specific version of `ale-py` to play the OpenAI Gym Atari games.
```
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
git checkout 069f8bd860b9da816cea58c5ade791025a51c105
# In setup.py turn the flags DSDL_SUPPORT and DSDL_DYNLOAD off
```
Install swig
```
conda install swig # In the RapidLearn conda environment
```
Now we can install the Autonomous Learning Library
```
git clone https://github.com/cpnota/autonomous-learning-library.git
cd autonomous-learning-library
pip install -e .[dev]
```

### Citation
Please cite our paper if you are inspired by R^3 in your work:
```
@inproceedings{li2023mathrm,
  title={$$\backslash$mathrm $\{$R$\}$\^{}$\{$3$\}$ $: On-Device Real-Time Deep Reinforcement Learning for Autonomous Robotics},
  author={Li, Zexin and Samanta, Aritra and Li, Yufei and Soltoggio, Andrea and Kim, Hyoseung and Liu, Cong},
  booktitle={2023 IEEE Real-Time Systems Symposium (RTSS)},
  pages={131--144},
  year={2023},
  organization={IEEE}
}
```

# Some useful tutorials for Deep Reinforcement Learning

[Train Donkey Car in Unity Simulator with Reinforcement Learning](https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html)

[Official version of Donkey Car](https://github.com/tawnkramer/gym-donkeycar)

[Autoencoder training of Doneky Car](https://github.com/araffin/aae-train-donkeycar)

[Learning to Drive in a Day(paper)](https://arxiv.org/abs/1807.00412)

[Learning to Drive in a Day(code)](https://github.com/r7vme/learning-to-drive-in-a-day)

[Learning to Drive Smoothly in Minutes(code)](https://github.com/araffin/learning-to-drive-in-5-minutes)

[Learning to Drive Smoothly in Minutes(blog)](https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)

[Autonomous Racing Robot With an Arduino, a Raspberry Pi and a Pi Camera](https://becominghuman.ai/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63)

[Racing Robot](https://github.com/sergionr2/RacingRobot)

[Applying Deep Learning to Autonomous Driving(MuSHR Car)](https://mushr.io/tutorials/deep_learning/)

[rl-baselines-zoo3](https://github.com/DLR-RM/rl-baselines3-zoo)

[Replicable-MARL](https://github.com/Replicable-MARL/MARLlib) This is a new Multi-agent DRL benchmark which is supported by Ray and RLib!

---

## Legacy installation (deprecated)

> ⚠️ The instructions in this section target the **original RTSS 2023 stack**
> (Python 3.8, PyTorch 1.13, gym 0.15.3, JetPack 5.0.2). They are kept here
> only for reproducibility of the published paper. JetPack 5.0.2 and PyTorch
> 1.13 are no longer maintained and are known to fail on current Jetson
> hardware (AGX Orin / Orin Nano on JP6).
>
> **For new work, use the modernized stack at the top of this README.**
>
> If you really need the original frozen artifact, check out the
> [`v1.0.0-legacy`](https://github.com/ZexinLi0w0/R3/releases/tag/v1.0.0-legacy)
> tag instead of running the snippets below on `main` / `upgrade/*`.

On x86 PC

Install Donkey simulator v22.11.06 from [Official version of Donkey Car](https://github.com/tawnkramer/gym-donkeycar).

gym-donkeycar recommend version: cd0cad6

Installation on PC
```bash
conda create -n rapidlearn python=3.8.10 # create a new conda environment named rapidlearn with python 3.8.10 (as on AGX)
conda activate rapidlearn
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # install pytorch 1.13.1 stable version
# cd gym-donkeycar-mushr
# python setup.py install # install the customized gym wrapper
cd gym-donkeycar
python setup.py install
cd ../MUSHR-DL
pip install -r requirements.txt # install the other dependencies
```

Installation on NVIDIA Jetson AGX

Software version: Jetpack 5.0.2

Hardware: NVIDIA Jetson AGX Orin & NVIDIA Jetson AGX Xavier

Software: Ubuntu 20.04.5 LTS; Python 3.8.10; PyTorch 1.13.0; Tensorflow 1.15.5+nv22.12; gym 0.15.3; gym-donkeycar 1.3.0

```bash
conda create -n rapidlearn python=3.8.10 # create a new conda environment named rapidlearn with python 3.8.10 (as on AGX)
conda activate rapidlearn
# Down load PyTorch wheel from https://developer.download.nvidia.com/compute/redist/jp/v502/
wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl
pip install torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl
# cd gym-donkeycar-mushr
# python setup.py install # install the customized gym wrapper
cd gym-donkeycar
python setup.py install
cd ../MUSHR-DL
pip install -r requirements.txt # install the other dependencies
```

# RTSS'23 R^3
Official Implementation of RTSS 2023 paper [R^3: On-device Real-Time Deep Reinforcement Learning for Autonomous Robotics](https://arxiv.org/pdf/2308.15039).

# Overview
R^3 is a comprehensive solution designed to optimize on-device real-time DRL training by managing the intricate balance between timing and algorithm performance, specifically in memory-constrained environments. The approach involves co-optimizing two pivotal parameters of DRL training:

Evaluation: we implement R^3 based on Autonomous Learning Library(ALL) library which involves multiple well-known deep reinforcement learning algorithms and well-built benchmark.

Practical case study: We adopt gym-donkeycar simulator to conduct deep reinforcement learning via gym APIs on DonkeyCar simulator.

# Quick Start
1. Install the dependencies.

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

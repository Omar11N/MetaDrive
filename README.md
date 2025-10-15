# MetaDrive RL Training with SAC Algorithm
Enhanced with Live Rendering, Video Recording, and Real-Time Visualization

## 🚗 Overview

This project implements a Soft Actor-Critic (SAC) reinforcement learning agent trained in the MetaDrive Simulator, an advanced driving simulation environment for autonomous driving research.
It provides a highly visual, interactive training loop — including live rendering, video recording of best episodes, and real-time training analytics — making it suitable for both research and demonstration purposes.

## ✨ Key Features

- 🎮 **Live Rendering**: Watch the agent drive and learn in real time.
- 📹 **Automatic Video Recording**: Save high-quality videos of top-performing episodes.
- 📊 **Live Training Visualization**: Real-time plots of rewards, success rate, and episode metrics.
- 🧠 **TensorBoard Integration**: Track learning curves, losses, and policy metrics.
- ⚙️ **Parallel Training (Multi-GPU Ready)**: Utilize multiple environments to accelerate learning.
- 💾 **Automatic Checkpointing**: Models saved periodically for safety and recovery.

## 🧩 Tech Stack

- **Simulation**: MetaDrive Simulator
- **RL Framework**: Stable-Baselines3 (SAC)
- **Visualization**: matplotlib, TensorBoard, imageio
- **Parallelism**: SubprocVecEnv / DummyVecEnv
- **Video Encoding**: imageio-ffmpeg

## 🛠️ Installation

### Prerequisites

Ensure you have Python 3.8+ and CUDA (optional, for GPU acceleration).

### Install Dependencies
```bash
pip install metadrive-simulator
pip install stable-baselines3[extra]
pip install tensorboard
pip install matplotlib
pip install imageio
pip install imageio-ffmpeg
```

## 🚀 Usage

### 1️⃣ Train the Agent (Basic)

Run a standard training session with parallel environments:
```bash
python train_metadrive_sac.py --n-envs 4
```

### 2️⃣ Watch Training Live (Single Environment)

Enable real-time visualization and rendering:
```bash
python train_metadrive_sac.py --live-render --n-envs 1
```

⚠️ **Note**: Live rendering works best with `--n-envs 1`.

### 3️⃣ Train Fast with Periodic Video Recording

Record every 50 episodes while using multiple environments:
```bash
python train_metadrive_sac.py --record-video --video-interval 50 --n-envs 4
```

### 4️⃣ Enable All Visualization Tools

Combine live plots, rendering, and recording:
```bash
python train_metadrive_sac.py --visualize --live-render --record-video --n-envs 1
```

## 📈 Monitoring Training

### TensorBoard

All training logs are automatically saved under `./metadrive_sac_training/logs`.
To monitor in real time:
```bash
tensorboard --logdir ./metadrive_sac_training/logs
```

### Live Metrics

The Visualization Callback provides:

- Reward progression
- Episode length distribution
- Rolling success rate
- Statistical summaries

Plots are updated dynamically and saved as `training_progress.png` upon completion.

## 🧪 Testing a Trained Model

After training, evaluate your agent:
```bash
python train_metadrive_sac.py --test ./metadrive_sac_training/sac_metadrive_final
```

**Options:**
```bash
--test-episodes N   # Number of test episodes (default: 10)
```

During testing:

- The agent drives in a rendered window.
- Success/failure, rewards, and episode statistics are logged.

## 📁 Directory Structure
```
metadrive_sac_training/
│
├── checkpoints/        # Periodic model checkpoints
├── logs/               # TensorBoard logs
├── videos/             # Recorded episode videos
├── training_progress.png  # Final training summary plot
└── sac_metadrive_final.zip # Final trained model
```

## ⚙️ Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--total-timesteps` | int | 1000000 | Total training steps |
| `--n-envs` | int | 8 | Number of parallel environments |
| `--live-render` | flag | - | Enable real-time environment rendering |
| `--record-video` | flag | - | Enable automatic video recording |
| `--video-interval` | int | 50 | Record every N episodes |
| `--visualize` | flag | - | Enable live matplotlib visualizations |
| `--plot-interval` | int | 200 | Update plot every N steps |
| `--device` | str | cuda | Device: cuda, cpu, or cuda:0 |
| `--test` | str | None | Path to model for evaluation |
| `--test-episodes` | int | 10 | Number of test episodes |

## 🧠 Architecture Overview

### Core Components

- **MetaDriveEnvWrapper** – Environment subclass that manages seeds and parallel envs.
- **VideoRecorderCallback** – Periodically records and saves agent performance videos.
- **VisualizationCallback** – Real-time training statistics, plotting, and analytics.
- **train_sac_agent()** – Central training loop integrating callbacks, TensorBoard, and checkpoints.
- **test_trained_agent()** – Evaluation routine for saved policies with rendering.

## 🖥️ Example Output

**During Training:**
```
[Episode 220] Reward: 45.3 | Length: 602 | Success: ✓
Avg Reward (100): 40.7 | Success Rate (100): 62.0%
📹 Video saved: ./videos/best_ep220_r45.mp4
```

**TensorBoard:**

- SAC losses
- Q-values
- Entropy coefficient (α)
- Reward curves

**Visual Plot:**

![Training Progress](training_progress.png)

## 🧾 License

This project is distributed under the MIT License.
See [LICENSE](LICENSE) for details.

## 👨‍💻 Authors & Credits

Developed by Omar Baiazid

Built using:
- [MetaDrive Simulator](https://github.com/metadriverse/metadrive)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

## 🧭 Next Steps

After training:
```bash
python train_metadrive_sac.py --test ./metadrive_sac_training/sac_metadrive_final
```

**Optional:**

- Convert trained SAC model to ONNX for deployment.
- Integrate reward shaping or curriculum learning.
- Explore multi-agent MetaDrive environments.

---

*"Teaching cars to drive — one policy gradient at a time."*

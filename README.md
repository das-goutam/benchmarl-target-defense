# BenchMARL Target Defense Environments

Multi-agent reinforcement learning environments for target defense scenarios using BenchMARL and VMAS.

## Overview

This repository contains custom VMAS environments for target defense scenarios where defenders must protect a target from attackers. The environments implement game-theoretic principles using Apollonius circles for optimal defense strategies.

## Environments

### 1. Target Defense Basic (`target_defense_basic.py`)
- Simple 1v1 or 3v1 defender-attacker scenario
- Defenders must detect attackers before they reach the target
- Uses sensing radius-based detection
- Apollonius optimization for reward calculation

### 2. Target Defense Smart (`target_defense_smart.py`)
- Enhanced version with smarter attacker behavior
- More sophisticated reward shaping
- Advanced spawning strategies

### 3. Target Defense Patrol (`target_defense_patrol.py`)
- Multi-agent patrol scenario
- Includes patrollers and pursuers with different roles
- Area-based spawning system

### 4. Target Defense Patrol End2End (`target_defense_patrol_End2End.py`)
- End-to-end learning version
- Integrated patrol and pursuit behaviors
- Heterogeneous agent speeds and capabilities

## Installation

### Prerequisites
```bash
# Install BenchMARL
pip install benchmarl

# Install VMAS
pip install vmas

# Install additional dependencies
pip install cvxpy torch torchrl tensordict
```

### Setup
1. Clone this repository:
```bash
git clone https://github.com/das-goutam/benchmarl-target-defense.git
cd benchmarl-target-defense
```

2. Copy the environment files to your BenchMARL installation:
```bash
# Find your BenchMARL installation path
python -c "import benchmarl; print(benchmarl.__path__[0])"

# Copy environments
cp environments/*.py <benchmarl_path>/environments/vmas/

# Copy YAML configs
cp yaml_configs/*.yaml <benchmarl_path>/conf/task/vmas/

# Copy Apollonius solver
cp apollonius_solver.py <your_project_directory>/
```

## Usage

### Basic Training Command

```bash
python -m benchmarl.run \
    task=vmas/target_defense_basic \
    algorithm=mappo \
    task.spawn_area_mode=true \
    task.spawn_area_width=0.1 \
    task.speed_ratio=0.3 \
    experiment.max_n_frames=1800000 \
    experiment.evaluation_interval=60000 \
    experiment.sampling_device=cuda \
    experiment.train_device=cuda \
    experiment.buffer_device=cuda \
    experiment.parallel_collection=true
```

### Environment-Specific Commands

#### Target Defense Basic (1v1)
```bash
python -m benchmarl.run \
    task=vmas/target_defense_basic \
    algorithm=mappo \
    task.num_defenders=1 \
    task.num_attackers=1 \
    task.speed_ratio=0.2 \
    task.sensing_radius=0.15 \
    experiment.max_n_frames=1000000
```

#### Target Defense Basic (3v1)
```bash
python -m benchmarl.run \
    task=vmas/target_defense_basic \
    algorithm=mappo \
    task.num_defenders=3 \
    task.num_attackers=1 \
    task.speed_ratio=0.3 \
    task.sensing_radius=0.15 \
    experiment.max_n_frames=1800000
```

#### Target Defense Smart
```bash
python -m benchmarl.run \
    task=vmas/target_defense_smart \
    algorithm=mappo \
    task.spawn_area_mode=true \
    task.spawn_area_width=0.1 \
    experiment.max_n_frames=2000000
```

#### Target Defense Patrol
```bash
python -m benchmarl.run \
    task=vmas/target_defense_patrol \
    algorithm=mappo \
    task.num_patrollers=1 \
    task.num_pursuers=2 \
    task.num_attackers=1 \
    task.patroller_sensing_radius=0.35 \
    experiment.max_n_frames=2000000
```

#### Target Defense Patrol End2End
```bash
python -m benchmarl.run \
    task=vmas/target_defense_patrol_end2end \
    algorithm=mappo \
    task.num_patrollers=1 \
    task.num_pursuers=2 \
    experiment.max_n_frames=2500000
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `task.num_defenders` | Number of defender agents | 3 |
| `task.num_attackers` | Number of attacker agents | 1 |
| `task.speed_ratio` | Attacker speed / Defender speed | 0.2 |
| `task.sensing_radius` | Detection radius for defenders | 0.15 |
| `task.spawn_area_mode` | Enable area-based spawning | false |
| `task.spawn_area_width` | Width of spawn area | 0.1 |
| `task.use_apollonius` | Use Apollonius optimization | true |
| `experiment.max_n_frames` | Total training frames | 1000000 |
| `experiment.evaluation_interval` | Evaluation frequency | 60000 |

## Apollonius Solver

The `apollonius_solver.py` module implements game-theoretic optimal defense using Apollonius circles:

```python
from apollonius_solver import solve_apollonius_optimization

result = solve_apollonius_optimization(
    attacker_pos=[0.5, 0.8],
    defender_positions=[[0.2, 0.2], [0.5, 0.1], [0.8, 0.2]],
    nu=5.0  # defender_speed / attacker_speed
)

print(f"Min Y-coordinate: {result['min_y_coordinate']}")
print(f"Defender payoff: {result['defender_payoff']}")
```

### Features:
- Computes Apollonius circle centers and radii
- Solves convex optimization problem for minimal attacker penetration
- Supports multiple defenders with heterogeneous speeds
- Numerical stability for small speed ratios

## Results

### Training Performance

See the `plots/` directory for visualization of training results:

- **3v1_comparison_20250927_122218.png**: Three defenders vs one attacker training curves
- **1v1_comparison_20250927_205404.png**: One defender vs one attacker comparison
- **evaluation_1000_episodes.png**: Evaluation results over 1000 episodes

### Key Findings

1. **3v1 Scenario**: Defenders achieve 85%+ detection rate after 1.8M frames
2. **1v1 Scenario**: Single defender achieves 60-70% detection rate
3. **Patrol Scenario**: Heterogeneous agents show emergent coordination

## Project Structure

```
benchmarl-target-defense/
├── environments/
│   ├── target_defense_basic.py
│   ├── target_defense_smart.py
│   ├── target_defense_patrol.py
│   └── target_defense_patrol_End2End.py
├── yaml_configs/
│   ├── target_defense_basic.yaml
│   ├── target_defense_smart.yaml
│   ├── target_defense_patrol.yaml
│   └── target_defense_patrol_End2End.yaml
├── plots/
│   ├── 3v1_comparison_20250927_122218.png
│   ├── 1v1_comparison_20250927_205404.png
│   └── evaluation_1000_episodes.png
├── apollonius_solver.py
└── README.md
```

## Advanced Usage

### Custom Training Script

```python
from benchmarl.experiment import Experiment
from benchmarl.algorithms import MappoConfig

# Create experiment
experiment = Experiment(
    task="vmas/target_defense_basic",
    algorithm=MappoConfig.get_from_yaml(),
    seed=0,
    config={
        "task": {
            "num_defenders": 3,
            "speed_ratio": 0.3,
            "sensing_radius": 0.15
        },
        "experiment": {
            "max_n_frames": 1800000,
            "evaluation_interval": 60000
        }
    }
)

# Run training
experiment.run()
```

### Evaluation Only

```bash
python -m benchmarl.run \
    task=vmas/target_defense_basic \
    algorithm=mappo \
    experiment.mode=evaluation \
    experiment.checkpoint_path=/path/to/checkpoint
```

## Citation

If you use these environments in your research, please cite:

```bibtex
@misc{benchmarl_target_defense,
  author = {Goutam Das},
  title = {BenchMARL Target Defense Environments},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/das-goutam/benchmarl-target-defense}
}
```

## References

- [BenchMARL](https://github.com/facebookresearch/BenchMARL): Multi-Agent Reinforcement Learning benchmark
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator): Vectorized Multi-Agent Simulator
- [TorchRL](https://github.com/pytorch/rl): PyTorch reinforcement learning library

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: [your-email@example.com]
- GitHub: [@das-goutam](https://github.com/das-goutam)

## Acknowledgments

This work builds upon the BenchMARL and VMAS frameworks and implements game-theoretic principles for multi-agent defense scenarios.

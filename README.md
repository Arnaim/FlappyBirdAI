# Flappy Bird AI — Neuroevolution with NEAT

An AI-powered Flappy Bird implementation where neural networks learn to play the game through **NeuroEvolution of Augmenting Topologies (NEAT)**.

The project explores evolutionary neural networks, progressive difficulty training, and performance evaluation by comparing a baseline model against a progressively trained model.

## Overview

This project combines a Flappy Bird game built with Pygame and an AI agent trained using NEAT.

Instead of using traditional supervised learning, the AI learns through an evolutionary process. A population of neural networks is evaluated, and better-performing networks are selected and evolved over multiple generations.

The project also implements progressive difficulty training, allowing the AI to evolve across increasingly challenging game levels.

## Features

- Flappy Bird game implemented with Pygame.
- AI-controlled bird using NEAT.
- Neural network with four input features and one output.
- Evolutionary training with a population of 100 genomes.
- Progressive training across five difficulty levels.
- Randomized pipe spacing and gap positions.
- Model evaluation using identical random seeds.
- CSV-based evaluation results.
- Jupyter notebook for visual analysis and comparison.

## How the AI Works

The AI uses a feed-forward neural network with:

**Input features:**
1. Bird's vertical position.
2. Bird's vertical velocity.
3. Distance to the next pipe.
4. Center position of the next pipe gap.

**Output:**

The network produces an output value. If the output exceeds the jump threshold of `0.5`, the bird jumps.

### NEAT Training

NEAT evolves neural networks by modifying their connection weights and structures.

The training process uses:
- Population size: 100
- Generations: 50 per training run
- Fitness-based selection
- Mutation and crossover
- Survival-based fitness rewards

Fitness rewards include surviving frames and passing pipes, while collisions receive a penalty.

## Progressive Difficulty Training

The progressive training approach evolves the population through five difficulty levels.

| Level | Speed | Gap Size |
|---|---:|---:|
| 1 | 5.0 | 165 |
| 2 | 5.5 | 158 |
| 3 | 6.0 | 150 |
| 4 | 6.5 | 144 |
| 5 | 7.2 | 138 |

Each stage runs for 10 generations, giving a total of 50 generations.

## Model Evaluation

The baseline model and progressive model were evaluated using the same 100 random seeds at Level 5.

| Metric | Baseline | Progressive |
|---|---:|---:|
| Average pipes | 33.90 | 39.03 |
| Minimum pipes | 1 | 1 |
| Maximum pipes | 68 | 69 |

**Average improvement:** 5.13 pipes (approximately 15.1%).

| Outcome | Number of tests |
|---|---:|
| Progressive model wins | 31 |
| Baseline model wins | 13 |
| Ties | 56 |

These results show that the progressive model achieved a higher average score on this evaluation set.

The comparison uses paired random seeds, meaning both models encounter the same seeded game scenarios.

## Visual Analysis

The project includes a Jupyter notebook for analyzing the evaluation results.

The notebook generates:
- Average performance comparison.
- Score distribution histograms.
- Box plots.
- Paired seed-by-seed comparisons.
- Per-seed score differences.
- Win/loss/tie distribution.
- Survival-time comparisons.

Notebook: `FlappyBirdAI_Level5_VSCode_Analysis.ipynb`

## Installation

### Requirements

- Python 3.14 (or a compatible Python version)
- Pygame
- NEAT-Python
- NumPy
- Matplotlib
- Pillow
- Pandas
- IPython / Jupyter kernel

### Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/FlappyBirdAI.git
cd FlappyBirdAI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Play with the trained AI

```bash
python play.py
```

### Train the baseline model

```bash
python -m training.train
```

### Train the progressive model

```bash
python -m training.train_progressive
```

### Evaluate the progressive and baseline models

```bash
python evaluate_progressive.py
```

Evaluation results are saved to:

```text
results/level_5_comparison.csv
```

### Analyze evaluation results

Open `FlappyBirdAI_Level5_VSCode_Analysis.ipynb` in VS Code with the Python and Jupyter extensions installed.

Select the project's Python virtual environment and run the notebook cells.

## Project Structure

```text
FlappyBirdAI/
├── assets/
├── config/
│   └── neat_config.txt
├── game/
│   ├── bird.py
│   ├── pipe.py
│   ├── game.py
│   └── difficulty.py
├── training/
│   ├── train.py
│   └── train_progressive.py
├── saved_models/
├── results/
│   └── level_5_comparison.csv
├── play.py
├── evaluate.py
├── evaluate_progressive.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pygame
- NEAT-Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Limitations and Future Improvements

- Improve consistency across more difficult levels.
- Experiment with additional neural network inputs.
- Record generation-by-generation training statistics in future training runs.
- Improve generalization across unseen game scenarios.
- Add gameplay recordings and visual demonstrations.

## Author

**Naimur Rahman Arnab**

GitHub: [Arnaim](https://github.com/Arnaim)


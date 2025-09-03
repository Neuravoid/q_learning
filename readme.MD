## Q-Learning on FrozenLake (Gymnasium)

A concise, reproducible implementation of tabular Q-Learning on the FrozenLake-v1 environment using Gymnasium. The notebook walks through environment setup, hyperparameter selection, epsilon-greedy exploration, training, evaluation, and simple visualizations.

### Key Features
- **Environment**: `FrozenLake-v1` with `is_slippery=False` for deterministic transitions
- **Algorithm**: Tabular Q-Learning with epsilon-greedy policy
- **Reproducibility**: Fixed random seeds where applicable
- **Visualization**: Training curves and final policy/value inspection

### Repository Structure
- `final.ipynb`: End-to-end notebook (setup, training, evaluation, plots)
- `readme.MD`: Project overview and usage guide (this file)

### Requirements
- Python 3.9+ (3.10/3.11 recommended)
- Windows PowerShell or any terminal

Python packages (install via pip):
- `gymnasium`
- `numpy`
- `matplotlib`
- `tqdm` (optional, if progress bars are used)

### Quickstart (Windows PowerShell)
```powershell
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install gymnasium numpy matplotlib tqdm

# 3) Launch Jupyter (if you prefer the classic UI)
pip install notebook
jupyter notebook final.ipynb

# Or use JupyterLab
pip install jupyterlab
jupyter lab final.ipynb
```

If you use VS Code or another IDE, you can open `final.ipynb` directly.

### How to Use the Notebook
1. Open `final.ipynb`.
2. Run the cells from top to bottom.
3. Training output will show progress and final results; plots visualize learning dynamics.

### Algorithm Overview
Q-Learning updates the action-value function Q(s,a) off-policy using bootstrapping:
\[ Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') − Q(s,a) ] \]

- **α (alpha)**: learning rate
- **γ (gamma)**: discount factor
- **ε (epsilon)**: exploration rate for epsilon-greedy action selection

Typical training loop steps:
- Initialize Q-table to zeros
- For each episode: iterate over steps until terminal state
  - Select action via epsilon-greedy
  - Step the environment to get (s', r, done)
  - Update Q(s,a) using the Bellman error
  - Decay epsilon over time (optional)

### Default Configuration (as used in the notebook)
- Environment: `FrozenLake-v1`, `is_slippery=False`
- Initialization: zeroed Q-table sized |S| × |A|
- Policy: epsilon-greedy with linear or exponential decay
- Training: multiple episodes with capped steps per episode
- Seeding: environment and RNG seeds set for reproducibility when possible

Note: `is_slippery=False` makes the environment deterministic, which helps learning converge faster in a didactic setting. Setting it to `True` increases stochasticity and difficulty.

### Results & Evaluation
- The notebook prints training progress (start/end) and shows plots, e.g., average rewards per episode.
- After training, you can inspect the learned policy by choosing the argmax action per state from the Q-table.

### Troubleshooting
- If you see rendering issues, ensure `matplotlib` is installed and the kernel was restarted after installation.
- If Gymnasium is not found, re-run `pip install gymnasium` in the active environment and restart the kernel.
- If execution is slow, reduce the number of episodes or disable progress bars.

### References
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
- Gymnasium documentation: `https://gymnasium.farama.org/`


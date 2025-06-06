# AIGC Stackelberg Game Base Code

This repository provides a simple Python implementation for experiments
described in the imaginary paper **"Optimizing AIGC Services Using Learning-based
Stackelberg Game in Vehicular Metaverses"**.

The code implements a small simulated environment and a tabular
Q-learning agent to model the leader--follower interaction. It is meant as a
starting point for further research and experimentation.

## Requirements

The scripts run with the standard Python 3.11 environment provided in this
repository. No external dependencies beyond the standard library are used.

## Structure

- `src/environment.py` – Stackelberg game environment.
- `src/agent.py` – Q-learning agent for the leader.
- `src/train.py` – example training loop.
- `tests/` – basic unit tests.

## Usage

Run a short training session:

```bash
python3 src/train.py
```

Run tests:

```bash
pytest -q
```

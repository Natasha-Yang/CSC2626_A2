# Q3 — Diffusion Policy + Behavior Cloning (Robomimic "square", image)

A minimal DDPM-based Diffusion Policy (`diffusion_policy.py`) and an explicit
behavior-cloning baseline (`bc.py`) for the Robomimic "square" nut-assembly
task, sharing a dataloader (`utils/dataset.py`) and network backbone
(`networks/`). See the module docstrings for the algorithmic details.

## Setup

The shared repo-root `.venv` (see `../README.md`) already has everything
pip-installable via `requirements.txt`, including `robosuite==1.4.1` and a
`mujoco==3.1.6` pin (robosuite doesn't work reliably against newer mujoco in
this stack — see the comment in `../requirements.txt`).

`robomimic` needs a from-source install, since none of its released PyPI
versions are a clean fit here (0.3.0 hard-imports legacy `mujoco_py`; 0.5.0
pulls in unrelated `transformers`/`diffusers`):

```bash
git clone https://github.com/ARISE-Initiative/robomimic.git /tmp/robomimic_src
cd /tmp/robomimic_src && git checkout v0.4.0
cd /path/to/A2_completed
uv pip install -p .venv --no-deps /tmp/robomimic_src
```

## Dataset

Download the "square" task's proficient-human demonstrations (image
observations, delta/relative actions, ~2.6GB):

```bash
mkdir -p Q3/data/robomimic/datasets/square/ph
curl -L -o Q3/data/robomimic/datasets/square/ph/image.hdf5 \
  https://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/image.hdf5
```

(This is robomimic's own per-task hosting — the CleanDiffuser-recommended
zip bundles all 5 tasks and is ~85GB, far more than needed here.)

## Running

```bash
cd Q3
python diffusion_policy.py    # trains + evaluates the DDPM policy
python bc.py                  # trains + evaluates the BC baseline
```

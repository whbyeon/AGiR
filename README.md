# AGiR: Mitigating Gift Over-Reliance in Mixed-Motive Games

Official code for **"AGiR: Mitigating Gift Over-Reliance in Mixed-Motive Games"**.

AGiR (Adaptive Gift Refusal) is an individual-level safeguard against gift over-reliance
in mixed-motive games that adaptively regulates incoming gifts based on the agent's
reward-gift orientation.

## Installation

```bash
git clone https://github.com/whbyeon/AGiR.git
cd AGiR
conda create -n agir python=3.10
conda activate agir
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"   # GPU
export PYTHONPATH="$PWD"
```

## Training

### Unfair Cleanup, Harvest, and Coin Game

Experiments in Section 5 of the paper. Example: LASE + AGiR on Unfair Cleanup-Hard.

```bash
bash algorithms/LASE/scripts/agir_lase_cleanup_hard.sh
```

### Two-agent Cleanup

Experiments in Section 6.2 of the paper. Example: LASE + AGiR in the mismatch scenario.

```bash
bash algorithms/LASE/scripts/agir_lase_2a_mismatch.sh
```

### Where things are

- `algorithms/LASE/` — LASE (without the imagination module) and LASE + AGiR.
- `algorithms/RX/` — Reward Exchange (RX) and RX + AGiR.
- `algorithms/IPPO/` — IPPO with individual reward (IPPO-I), IPPO with common reward (IPPO-C) and
  inequity aversion (IA). IPPO-C and IA are selected through the environment flags
  `ENV_KWARGS.shared_rewards` and `ENV_KWARGS.inequity_aversion`.
- `socialjax/environments/` — the environments. `cleanup/clean_up_4a_{Alow,Amid}.py`,
  `cleanup/clean_up_2a_*.py`, `common_harvest/harvest_open_unfair.py` and `coin_game/coin_game.py` are the ones used in the paper.

## Citation

TBD

## License

See [LICENSE](LICENSE). The environments are based on
[SocialJax](https://github.com/cooperativex/SocialJax), licensed under Apache 2.0.

# Mod-p Verifier Sandbox

Separate first experiment for the small algorithmic setup suggested by Serguei:

- train a tiny Transformer from scratch on part of a modular addition table;
- evaluate held-out addition accuracy `a`;
- apply a configurable noisy verifier with `(alpha, beta)` where:
  - `alpha = Pr[V=1 | candidate is correct]`
  - `beta = Pr[V=1 | candidate is wrong]`
- measure post-verifier accepted accuracy `a_prime`;
- compare it to the conditional-accuracy formula:

```text
a_prime = alpha * a / (alpha * a + beta * (1 - a))
```

This folder is intentionally separate from `game24_composition/`.

## Run

```bash
cd modp_verifier_sandbox
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_modp_verifier.sh
```

Fast smoke run:

```bash
EPOCHS=5 P=17 bash run_modp_verifier.sh
```

Outputs are saved under:

- `outputs/metrics.json`
- `outputs/metrics.csv`
- `outputs/predictions.csv`
- `outputs/accuracy_curve.csv`
- `checkpoints/epoch_*.pt`
- `checkpoints/final.pt`

Generated outputs and checkpoints are ignored by git.

## Scope

Implemented now: mod-p addition only.

Not implemented yet: S_5 multiplication, Countdown/Game24 nano-transformer, RL, tree search, noisy self-training loops.

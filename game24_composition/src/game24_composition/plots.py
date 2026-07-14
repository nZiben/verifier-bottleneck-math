"""Plotting helpers for exploration sweeps."""

from __future__ import annotations

from pathlib import Path


def plot_passk_curve(results, out_path):
    import matplotlib.pyplot as plt

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for result in results:
        pass_at_k = result["pass_at_k"]
        xs = sorted(int(k) for k in pass_at_k)
        ys = [pass_at_k[str(k)] for k in xs]
        label = f"T={result['temperature']}, top_p={result['top_p']}, n={result['num_samples']}"
        plt.plot(xs, ys, marker="o", label=label)

    plt.xscale("log", base=2)
    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.title("AB symbolic Game24 exploration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

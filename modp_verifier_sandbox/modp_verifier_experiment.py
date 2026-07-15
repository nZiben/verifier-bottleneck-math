from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyTransformer(nn.Module):
    def __init__(self, p, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.p = p
        self.plus_token = p
        self.eq_token = p + 1
        self.embed = nn.Embedding(p + 2, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 4, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, p)

    def forward(self, x):
        h = self.embed(x) + self.pos
        h = self.encoder(h)
        return self.head(h[:, -1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=31)
    parser.add_argument("--train_frac", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--alpha_beta", nargs="+", default=["1.0,0.0", "0.9,0.1", "0.8,0.2"])
    args = parser.parse_args()
    run(args)


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = make_modp_table(args.p)
    train_idx, test_idx = split_indices(len(y), args.train_frac, args.seed)
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    test_x, test_y = x[test_idx].to(device), y[test_idx].to(device)

    model = TinyTransformer(
        p=args.p,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    curve = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, opt, train_loader, device)
        train_acc = accuracy(model, x[train_idx].to(device), y[train_idx].to(device))
        test_acc = accuracy(model, test_x, test_y)
        curve.append({"epoch": epoch, "loss": loss, "train_accuracy": train_acc, "test_accuracy": test_acc})
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", model, args, epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    save_checkpoint(checkpoint_dir / "final.pt", model, args, args.epochs)
    preds, correct = predict(model, test_x, test_y)
    base_accuracy = sum(correct) / len(correct)
    verifier_rows = verifier_sweep(correct, parse_alpha_beta(args.alpha_beta), args.seed)

    metrics = {
        "task": "mod_p_addition",
        "p": args.p,
        "train_frac": args.train_frac,
        "num_train": len(train_idx),
        "num_test": len(test_idx),
        "model": {
            "type": "TinyTransformer",
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
        },
        "epochs": args.epochs,
        "seed": args.seed,
        "device": device,
        "base_accuracy_a": base_accuracy,
        "verifier": verifier_rows,
    }
    write_json(out_dir / "metrics.json", metrics)
    write_rows(out_dir / "metrics.csv", verifier_rows)
    write_rows(out_dir / "accuracy_curve.csv", curve)
    write_predictions(out_dir / "predictions.csv", x[test_idx].tolist(), test_y.cpu().tolist(), preds, correct)
    print(f"wrote {out_dir} and {checkpoint_dir}")


def make_modp_table(p):
    rows, labels = [], []
    plus_token, eq_token = p, p + 1
    for a in range(p):
        for b in range(p):
            rows.append([a, plus_token, b, eq_token])
            labels.append((a + b) % p)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def split_indices(n, train_frac, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = max(1, min(n - 1, int(round(n * train_frac))))
    return torch.tensor(idx[:n_train]), torch.tensor(idx[n_train:])


def train_epoch(model, opt, loader, device):
    model.train()
    total_loss, total = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(yb)
        total += len(yb)
    return total_loss / total


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    return (model(x).argmax(dim=-1) == y).float().mean().item()


@torch.no_grad()
def predict(model, x, y):
    model.eval()
    pred = model(x).argmax(dim=-1)
    correct = (pred == y).cpu().tolist()
    return pred.cpu().tolist(), [bool(item) for item in correct]


def parse_alpha_beta(items):
    pairs = []
    for item in items:
        alpha, beta = item.split(",", 1)
        pairs.append((float(alpha), float(beta)))
    return pairs


def verifier_sweep(correct, pairs, seed):
    rng = random.Random(seed + 1)
    a = sum(correct) / len(correct)
    rows = []
    for alpha, beta in pairs:
        accepted_correct = 0
        accepted_total = 0
        for is_correct in correct:
            accept_prob = alpha if is_correct else beta
            if rng.random() < accept_prob:
                accepted_total += 1
                accepted_correct += int(is_correct)
        measured = accepted_correct / accepted_total if accepted_total else None
        predicted = predicted_a_prime(a, alpha, beta)
        rows.append(
            {
                "alpha": alpha,
                "beta": beta,
                "base_accuracy_a": a,
                "accepted_total": accepted_total,
                "accepted_rate": accepted_total / len(correct),
                "measured_a_prime": measured,
                "predicted_a_prime": predicted,
                "abs_error": abs(measured - predicted) if measured is not None and predicted is not None else None,
            }
        )
    return rows


def predicted_a_prime(a, alpha, beta):
    denom = alpha * a + beta * (1.0 - a)
    return None if denom == 0 else (alpha * a) / denom


def save_checkpoint(path, model, args, epoch):
    torch.save({"epoch": epoch, "args": vars(args), "model_state_dict": model.state_dict()}, path)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path, inputs, gold, preds, correct):
    rows = [
        {"a": row[0], "b": row[2], "gold": target, "pred": pred, "correct": is_correct}
        for row, target, pred, is_correct in zip(inputs, gold, preds, correct)
    ]
    write_rows(path, rows)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

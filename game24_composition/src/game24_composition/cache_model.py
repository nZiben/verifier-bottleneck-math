"""Download a Hugging Face model once and reuse the local path."""

from __future__ import annotations

import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    for attempt in range(1, args.retries + 1):
        try:
            snapshot_download(repo_id=args.model_name, local_dir=args.out_dir)
            print(f"Cached {args.model_name} at {args.out_dir}")
            return
        except Exception:
            if attempt == args.retries:
                raise
            time.sleep(30 * attempt)


if __name__ == "__main__":
    main()

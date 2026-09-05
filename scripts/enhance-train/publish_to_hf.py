# -*- coding: utf-8 -*-
"""Publish the editor model and its training corpus to Hugging Face.

Nothing here runs automatically: publishing is public and hard to take back, so
this is a script you read and then run yourself.

    pip install huggingface_hub
    hf auth login                      # or: huggingface-cli login

    python publish_to_hf.py --user YOUR_NAME --dry-run    # see what would happen
    python publish_to_hf.py --user YOUR_NAME --dataset    # push the corpus
    python publish_to_hf.py --user YOUR_NAME --model      # push the model

Repos are created **private** by default. Publish, look at the rendered card,
then flip to public in the web UI when you are happy with it.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

CARDS = pathlib.Path(__file__).resolve().parent / "cards"

MODEL_DIR = pathlib.Path("D:/dev/handy-models")
CORPUS_DIR = pathlib.Path("D:/dev/handy-train-alpaca")

DEFAULT_MODEL_REPO = "handy-editor-lfm2.5-350m"
DEFAULT_DATASET_REPO = "handy-dictation-editing"

# Local file -> name in the repo. The checkpoint numbers are meaningful only to
# the run that produced them; a published artefact should say what it is.
# Only quants that passed *both* evaluations. Q2_K_L is deliberately absent: it
# scores 62/68 on the suite, which looks survivable, and 47.9% on the held-out
# set, which is half the edits wrong. Nothing goes in this dict on the strength
# of the suite alone.
MODEL_FILES = {
    "LiquidAI_LFM2.5-350M_1788615152.Q4_K_M.gguf": "handy-editor-350m-Q4_K_M.gguf",
    "LiquidAI_LFM2.5-350M_1788615152.Q8_0.gguf": "handy-editor-350m-Q8_0.gguf",
    "LiquidAI_LFM2.5-350M_1788615152.F16.gguf": "handy-editor-350m-F16.gguf",
}

# The chat files are the ones this model was trained on; the Alpaca files are the
# same examples for anyone fine-tuning a base checkpoint.
DATASET_FILES = [
    "handy_editor.jsonl",
    "handy_editor_alpaca.jsonl",
    "handy_eval.jsonl",
    "handy_eval_alpaca.jsonl",
    "manifest.json",
]


def check(paths: list[pathlib.Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"  missing: {p}", file=sys.stderr)
        raise SystemExit("nothing was uploaded")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    ap.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    ap.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    ap.add_argument("--model", action="store_true", help="publish the model")
    ap.add_argument("--dataset", action="store_true", help="publish the corpus")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public instead of private")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (args.model or args.dataset):
        raise SystemExit("pick --model, --dataset, or both")

    model_id = f"{args.user}/{args.model_repo}"
    dataset_id = f"{args.user}/{args.dataset_repo}"

    plan: list[tuple[str, pathlib.Path, str, str]] = []
    if args.dataset:
        check([CORPUS_DIR / f for f in DATASET_FILES] + [CARDS / "DATASET_CARD.md"])
        plan.append(("dataset", CARDS / "DATASET_CARD.md", "README.md", dataset_id))
        for f in DATASET_FILES:
            plan.append(("dataset", CORPUS_DIR / f, f, dataset_id))
    if args.model:
        check([MODEL_DIR / f for f in MODEL_FILES] + [CARDS / "MODEL_CARD.md"])
        plan.append(("model", CARDS / "MODEL_CARD.md", "README.md", model_id))
        for src, dst in MODEL_FILES.items():
            plan.append(("model", MODEL_DIR / src, dst, model_id))

    for kind, src, dst, repo in plan:
        size = src.stat().st_size / 1e6
        print(f"  {kind:8} {repo}/{dst:34} <- {src.name}  ({size:.1f} MB)")

    # The cards ship with placeholders that would otherwise be published as-is.
    for card in (CARDS / "MODEL_CARD.md", CARDS / "DATASET_CARD.md"):
        if card.exists() and "TODO" in card.read_text(encoding="utf-8"):
            print(f"\n  ! {card.name} still contains TODO placeholders")

    if args.dry_run:
        print("\ndry run; nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    for repo_id, repo_type in ((model_id, "model"), (dataset_id, "dataset")):
        if any(p[3] == repo_id for p in plan):
            api.create_repo(repo_id, repo_type=repo_type,
                            private=not args.public, exist_ok=True)
            print(f"repo ready: {repo_id} ({repo_type})")

    for kind, src, dst, repo in plan:
        print(f"uploading {dst} -> {repo}")
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=repo, repo_type=kind)

    print("\ndone. Review the rendered cards, then make the repos public.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

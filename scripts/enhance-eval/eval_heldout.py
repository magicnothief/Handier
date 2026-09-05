"""Score a model against the held-out eval set built by `enhance-train`.

The 68-case suite in `suite.py` is the behavioural exam, but the training corpus
now draws on the same signal vocabulary, so a fine-tune's score there is no
longer fully independent. `handy_eval.jsonl` is held out by *source* — real rows
from a split training never reads, synthetic rows from a different seed with
every training input excluded — so it is the number to trust for a fine-tune.

Usage:
  python eval_heldout.py --model PATH [--eval D:/dev/handy-train-v2/handy_eval.jsonl]
                         [--limit N] [--prompt FILE]
"""

import argparse
import json
import random
import re
import statistics
import sys
import time

from bench import ALPACA_INSTRUCTION, ALPACA_PREAMBLE, Sidecar


def norm(text: str) -> str:
    """Compare on words, not on incidental spacing or case."""
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", text)


def content_words(text: str) -> set[str]:
    return set(norm(text).split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="D:/dev/handy-train-v2/handy_eval.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="sample this many rows")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--prompt", help="system prompt file; omit for a promptless model")
    ap.add_argument("--verify-prompt",
                    help="separate prompt for verify rows, as the app sends")
    ap.add_argument("--label", default="held-out")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--show", type=int, default=6, help="example failures to print")
    ap.add_argument("--alpaca", action="store_true",
                    help="send the Alpaca prompt as the user turn, for a fine-tune "
                         "trained on instruction/input/output")
    ap.add_argument("--switch", action="store_true",
                    help="append /no_think, for a stock reasoning model")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    if args.limit and args.limit < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]

    system = open(args.prompt, encoding="utf-8").read().strip() if args.prompt else ""
    if args.alpaca:
        system = ""

    def wrap(text: str) -> str:
        """Assemble the Alpaca prompt, matching training exactly.

        Split across the system and user slots instead, the same model repeats
        itself until the token budget runs out -- so this is not cosmetic.
        """
        if not args.alpaca:
            return text
        return (
            f"{ALPACA_PREAMBLE}### Instruction:\n{ALPACA_INSTRUCTION}"
            f"\n\n### Input:\n{text}\n\n### Response:\n"
        )
    # A prompted stock model needs the *verify* prompt on verify rows. The app
    # sends two different prompts for the two jobs; handing it the editor prompt
    # for a "Original:/Edited:" row makes it edit the pair instead of judging it,
    # which scores 0 and says nothing about the model. A promptless fine-tune
    # needs neither, and tells the tasks apart from the user turn.
    verify_system = (
        open(args.verify_prompt, encoding="utf-8").read().strip()
        if args.verify_prompt else system
    )
    sc = Sidecar(args.model, think=False, cpu=args.cpu, switch=args.switch)

    verify_ok = verify_n = 0
    # A verify model that always says SAME scores well on an unbalanced set, so
    # the two labels are tracked apart.
    per_label = {"SAME": [0, 0], "CHANGED": [0, 0]}
    edit_exact = edit_n = 0
    edit_f1: list[float] = []
    failures: list[str] = []
    times: list[float] = []

    for row in rows:
        msgs = row["messages"]
        user, want = msgs[-2]["content"], msgs[-1]["content"]
        is_verify = want in ("SAME", "CHANGED")
        t0 = time.time()
        out, _ = sc.gen(
            verify_system if is_verify else system,
            user if is_verify else wrap(user),
            budget=320,
        )
        times.append((time.time() - t0) * 1000)

        if is_verify:
            verify_n += 1
            per_label[want][1] += 1
            got = out.strip().upper()
            # Anything that is not one of the two labels counts as wrong; the
            # host treats an unparseable verdict as "changed" and throws the
            # edit away, so a chatty answer is a real failure, not a near miss.
            if got.startswith(want):
                verify_ok += 1
                per_label[want][0] += 1
            elif len(failures) < args.show:
                failures.append(f"  VERIFY want={want} got={out[:70]!r}\n    {user[:110]}")
            continue

        edit_n += 1
        if norm(out) == norm(want):
            edit_exact += 1
        else:
            got_w, want_w = content_words(out), content_words(want)
            inter = len(got_w & want_w)
            p = inter / len(got_w) if got_w else 0.0
            r = inter / len(want_w) if want_w else 0.0
            edit_f1.append(2 * p * r / (p + r) if p + r else 0.0)
            if len(failures) < args.show:
                failures.append(
                    f"  EDIT\n    in   {user[:110]}\n    want {want[:110]}\n    got  {out[:110]}"
                )
    for _ in range(edit_exact):
        edit_f1.append(1.0)

    sc.close()

    print(f"\n=== {args.label} :: {args.model.split('/')[-1]} ===")
    print(f"  rows {len(rows)}   median {int(statistics.median(times))}ms")
    if edit_n:
        print(f"  edit    exact {edit_exact}/{edit_n} ({100*edit_exact/edit_n:.1f}%)   "
              f"mean word-F1 {statistics.mean(edit_f1):.3f}")
    if verify_n:
        print(f"  verify  {verify_ok}/{verify_n} ({100*verify_ok/verify_n:.1f}%)   "
              + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in per_label.items()))
    for f in failures:
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

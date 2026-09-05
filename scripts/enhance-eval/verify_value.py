"""Does the verify pass earn its place, given how good the editor now is?

The held-out eval scores the verifier on *synthesised* corruptions, which it was
trained on. That answers "can it spot the damage we generate", not the question
the app actually poses: "when this editor makes a real mistake, does the verifier
catch it, and how many good edits does it destroy in the process?"

So: run the editor for real, label each output against the reference, then ask the
verifier about the editor's own text. Four outcomes matter.

    correct edit -> SAME      kept, right call
    correct edit -> CHANGED   FALSE REJECT: user loses a good edit
    wrong edit   -> CHANGED   CAUGHT: user saved from a bad paste
    wrong edit   -> SAME      MISSED

At a low editor error rate the verifier sees mostly good edits, so even a small
false-reject rate can cost more than it saves. That is the whole question.

The deterministic triage in `enhance/verify.rs` is replayed alongside, because
`Auto` only refers an edit to the model when that triage flags it -- if the triage
alone separates the classes, the second inference pass is not buying anything.

Usage:
  python verify_value.py --model PATH [--limit N]
"""

import argparse
import json
import random
import re
import sys

from bench import Sidecar
from eval_heldout import content_words, norm

# --- replica of enhance/verify.rs triage ------------------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in",
    "on", "at", "for", "with", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "our", "their",
}


def content_sequence(text: str) -> list[str]:
    return [w for w in norm(text).split() if w not in STOPWORDS]


def reorders_shared_words(original: str, edited: str) -> bool:
    """True when both sides use the same content words in a different order."""
    a, b = content_sequence(original), content_sequence(edited)
    shared = set(a) & set(b)
    if len(shared) < 2:
        return False
    return [w for w in a if w in shared] != [w for w in b if w in shared]


def adds_content_words(original: str, edited: str) -> bool:
    before = set(content_sequence(original))
    return any(w not in before for w in content_sequence(edited))


def needs_verification(original: str, edited: str) -> bool:
    return reorders_shared_words(original, edited) or adds_content_words(original, edited)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="D:/dev/handy-train-v2/handy_eval.jsonl")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--f1-floor", type=float, default=0.97,
                    help="word-F1 above which a non-exact edit counts as correct")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    edits = [r for r in rows if r["messages"][-1]["content"] not in ("SAME", "CHANGED")]
    random.Random(args.seed).shuffle(edits)
    edits = edits[: args.limit]

    sc = Sidecar(args.model, think=False, switch=False)

    cells = {("ok", "SAME"): 0, ("ok", "CHANGED"): 0,
             ("bad", "SAME"): 0, ("bad", "CHANGED"): 0}
    triage_cells = {("ok", True): 0, ("ok", False): 0,
                    ("bad", True): 0, ("bad", False): 0}
    false_rejects, catches = [], []

    for row in edits:
        src = row["messages"][-2]["content"]
        want = row["messages"][-1]["content"]
        out, _ = sc.gen("", src)

        if norm(out) == norm(want):
            quality = "ok"
        else:
            got_w, want_w = content_words(out), content_words(want)
            inter = len(got_w & want_w)
            p = inter / len(got_w) if got_w else 0.0
            r = inter / len(want_w) if want_w else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            # A reference from real speech is not always better than the model's
            # answer, so near-identical output is not counted as an error.
            quality = "ok" if f1 >= args.f1_floor else "bad"

        verdict, _ = sc.gen("", f"Original: {src}\nEdited: {out}")
        verdict = "CHANGED" if verdict.strip().upper().startswith("CHANGED") else "SAME"
        cells[(quality, verdict)] += 1
        triage_cells[(quality, needs_verification(src, out))] += 1

        if quality == "ok" and verdict == "CHANGED" and len(false_rejects) < args.show:
            false_rejects.append(f"    in   {src[:100]}\n    out  {out[:100]}")
        if quality == "bad" and verdict == "CHANGED" and len(catches) < args.show:
            catches.append(f"    in   {src[:100]}\n    want {want[:100]}\n    out  {out[:100]}")

    sc.close()

    ok = cells[("ok", "SAME")] + cells[("ok", "CHANGED")]
    bad = cells[("bad", "SAME")] + cells[("bad", "CHANGED")]
    n = ok + bad
    print(f"\n=== verify value :: {args.model.split('/')[-1]} ===")
    print(f"  {n} edits   editor correct {ok} ({100*ok/n:.1f}%)   wrong {bad} ({100*bad/n:.1f}%)")
    print("\n  LLM verify pass:")
    print(f"    caught      {cells[('bad','CHANGED')]:4d} of {bad} bad edits")
    print(f"    missed      {cells[('bad','SAME')]:4d}")
    print(f"    FALSE REJECT{cells[('ok','CHANGED')]:4d} of {ok} good edits")
    saved, lost = cells[("bad", "CHANGED")], cells[("ok", "CHANGED")]
    print(f"    net         {saved - lost:+d} edits  (saved {saved}, destroyed {lost})")

    print("\n  deterministic triage alone (no second inference pass):")
    print(f"    flags       {triage_cells[('bad',True)]:4d} of {bad} bad edits")
    print(f"    flags       {triage_cells[('ok',True)]:4d} of {ok} good edits (would be referred)")

    if catches:
        print("\n  examples caught:")
        for c in catches:
            print(c)
    if false_rejects:
        print("\n  examples destroyed:")
        for f in false_rejects:
            print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

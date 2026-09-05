"""Check a built corpus before spending GPU hours on it.

Every check here corresponds to a bug that reached a trained model at least
once. Run it on the directory `build_dataset.py` wrote:

    python validate_dataset.py D:/dev/handy-train-v2

Exits non-zero if anything is wrong, so it can gate a training run.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

# A verify request is told apart from an edit request by its opening word, so
# an edit whose transcript happens to start this way is ambiguous to the model.
VERIFY_OPENER = re.compile(r"^\s*Original:", re.I)
DOUBLED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.I)
# "that that" and "had had" are ordinary English; a doubled article or pronoun
# in a *target* is a stutter the model is being taught to keep.
STUTTER_WORDS = {"the", "a", "an", "to", "i", "we", "it", "is", "of", "and", "in"}

# Only the hesitation sounds. The hedges ("kind of", "sort of", "I mean",
# "you know", "basically") are excluded on purpose: they are filler *and*
# ordinary English, and the annotators of the real-speech sources kept the
# ordinary uses. Flagging them made this check fire on "I mean what I say" --
# which is one of the eval suite's preservation cases -- and on "kind of
# anxious". A word-level check cannot tell the two senses apart, so it should
# not try.
#
# "uh" is matched with a trailing boundary that excludes "uh-huh", which is a
# backchannel word rather than a hesitation and is meant to survive.
FILLERS = [r"um", r"uh(?!-)", r"erm", r"uhm", r"my bad"]


def load(path: pathlib.Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path.name}:{n}: not valid JSON ({e})")
    return rows


def as_messages(row: dict) -> list[dict] | None:
    """Return the row as a chat turn list, whatever format it is written in.

    The corpus ships in two shapes -- chatml for an instruct model with a chat
    template, Alpaca for a base model without one -- and every check below is
    about the content, not the container. Normalising here means the Alpaca
    files are gated by exactly the same rules rather than skipped, which is what
    happened when this only understood `messages`.
    """
    msgs = row.get("messages")
    if isinstance(msgs, list):
        return msgs
    if "output" in row and "input" in row:
        # `instruction` is a constant across the file, so it carries no
        # example-level information and is deliberately not turned into a
        # system turn: doing so would make every row look like it had one.
        return [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    return None


def check(path: pathlib.Path, suite_inputs: set[str]) -> list[str]:
    rows = load(path)
    problems: list[str] = []
    roles = collections.Counter()
    editor_rows: list[dict] = []
    verify_rows: list[dict] = []

    for i, row in enumerate(rows):
        msgs = as_messages(row)
        if not isinstance(msgs, list) or len(msgs) < 2:
            problems.append(f"row {i}: neither a 'messages' array nor an Alpaca row")
            continue
        # Downstream checks index `row["messages"]`, so give an Alpaca row the
        # same shape rather than special-casing every one of them.
        row = dict(row, messages=msgs)
        if not all(isinstance(m, dict) and "role" in m and "content" in m for m in msgs):
            problems.append(f"row {i}: a message is missing 'role' or 'content'")
            continue
        roles[tuple(m["role"] for m in msgs)] += 1
        if msgs[-1]["role"] != "assistant":
            problems.append(f"row {i}: last turn is {msgs[-1]['role']}, not assistant")
        if not str(msgs[-1]["content"]).strip():
            problems.append(f"row {i}: empty assistant turn")
        (verify_rows if VERIFY_OPENER.match(str(msgs[-2]["content"])) else editor_rows).append(row)

    if any("instruction" in r for r in rows):
        blank = sum(1 for r in rows if not str(r.get("instruction", "")).strip())
        if blank:
            problems.append(f"{blank} Alpaca rows have an empty 'instruction'")
        distinct = {str(r.get("instruction", "")) for r in rows}
        if len(distinct) > 2:
            problems.append(
                f"{len(distinct)} distinct instructions; the host sends exactly one"
            )

    print(f"\n== {path.name} ==")
    print(f"  rows            {len(rows)}")
    print(f"  role patterns   {dict(roles)}")
    print(f"  looks like edit {len(editor_rows)}   looks like verify {len(verify_rows)}")

    # A verify example must answer with exactly one of the two labels; anything
    # else means the host's parser will fall back to "changed" and discard a
    # good edit.
    labels = collections.Counter(
        str(r["messages"][-1]["content"]).strip() for r in verify_rows
    )
    stray = {k: v for k, v in labels.items() if k not in ("SAME", "CHANGED")}
    if stray:
        problems.append(f"verify rows with a label that is not SAME/CHANGED: {stray}")
    if verify_rows:
        print(f"  verify labels   {dict(labels)}")

    # An edit target that is a bare label means an example landed in the wrong
    # corpus.
    mislabelled = sum(
        1 for r in editor_rows if str(r["messages"][-1]["content"]).strip() in ("SAME", "CHANGED")
    )
    if mislabelled:
        problems.append(f"{mislabelled} edit rows answer with a verify label")

    if editor_rows:
        targets = [str(r["messages"][-1]["content"]) for r in editor_rows]

        q = sum(1 for t in targets if t.rstrip().endswith("?"))
        share = 100.0 * q / len(targets)
        print(f"  question-shaped {q} ({share:.1f}%)")
        # disfl_qa is all questions; letting it dominate taught a fine-tune to
        # answer everything with one.
        if share > 35:
            problems.append(
                f"{share:.1f}% of edit targets are questions; cap disfl_qa harder"
            )

        stutters = [
            t for t in targets
            if (m := DOUBLED_WORD.search(t)) and m.group(1).lower() in STUTTER_WORDS
        ]
        print(f"  stutter targets {len(stutters)} ({100.0 * len(stutters) / len(targets):.2f}%)")
        if len(stutters) > 0.01 * len(targets):
            problems.append(
                f"{len(stutters)} targets keep a doubled function word, e.g. {stutters[0]!r}"
            )

        leaked = collections.Counter()
        for t in targets:
            low = t.lower()
            for f in FILLERS:
                if re.search(r"\b" + f + r"\b", low):
                    leaked[f] += 1
        if leaked:
            print(f"  filler in target {dict(leaked.most_common(5))}")
        if sum(leaked.values()) > 0.01 * len(targets):
            problems.append(f"fillers survive into {sum(leaked.values())} targets")

        malformed = [
            t for t in targets
            if re.search(r",\s*[.,]|\s+[,.]|,\s*$", t)
            or re.search(r"\b(and|but|or|the|of|a|an|with|from|because)\s*[,.]?\s*$", t, re.I)
        ]
        print(f"  malformed       {len(malformed)} "
              f"({100.0 * len(malformed) / len(targets):.2f}%)")
        if len(malformed) > 0.002 * len(targets):
            problems.append(
                f"{len(malformed)} targets end mid-clause or carry a segmentation "
                f"artefact, e.g. {malformed[0]!r}"
            )

        # The eval suite must not be trainable, or its score means nothing.
        hit = [
            r for r in editor_rows
            if re.sub(r"[^a-z0-9 ]", "", str(r["messages"][-2]["content"]).lower()).strip()
            in suite_inputs
        ]
        print(f"  suite leakage   {len(hit)}")
        if hit:
            problems.append(f"{len(hit)} rows reproduce an eval-suite input verbatim")

    return problems


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = pathlib.Path(sys.argv[1])

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "enhance-eval"))
    try:
        import suite  # type: ignore

        suite_inputs = {
            re.sub(r"[^a-z0-9 ]", "", c[1].lower()).strip() for c in suite.ALL
        }
    except Exception as e:
        print(f"! eval suite unavailable ({e}); skipping the leakage check")
        suite_inputs = set()

    files = sorted(out.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl files in {out}")

    problems: list[str] = []
    for path in files:
        problems += [f"{path.name}: {p}" for p in check(path, suite_inputs)]

    print()
    if problems:
        print(f"FAILED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK — {len(files)} files pass every check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

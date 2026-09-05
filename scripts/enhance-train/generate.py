"""Generate self-correction training data for the enhancement layer.

Produces two datasets from one pass:

  editor.jsonl    dictated transcript -> cleaned text
  verifier.jsonl  (transcript, edit)  -> SAME / CHANGED

The verifier set is *derived* from the editor set rather than generated
separately. Every correct edit is a SAME example, and applying a known
corruption to it gives a CHANGED example with a guaranteed-correct label. That
is worth doing deliberately: labelling meaning-preservation by hand is slow and
subjective, and the corruptions below are exactly the failures measured in
production rather than a guess at what might go wrong.

Usage:
  python generate.py --out data/ --train 60000 --eval 2000
"""

import argparse
import json
import pathlib
import random
import re

import banks

# ---------------------------------------------------------------------------
# Rendering a dictated transcript
# ---------------------------------------------------------------------------


def strip_punct(text: str) -> str:
    """Approximate what an ASR model hands us: lowercase, largely unpunctuated.

    Not fully unpunctuated -- Whisper and Parakeet both emit some -- so a
    fraction of samples keep theirs, which stops the model from keying on
    "input never has punctuation" as the signal that something needs editing.
    """
    text = text.lower()
    text = re.sub(r"[.,!?;:]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def sprinkle_fillers(text: str, rng: random.Random, rate: float) -> str:
    """Insert fillers at word boundaries at roughly `rate` per word."""
    words = text.split()
    out = []
    for word in words:
        if rng.random() < rate:
            out.append(banks.weighted_filler(rng))
        out.append(word)
    if rng.random() < rate:
        out.append(banks.weighted_filler(rng))
    return " ".join(out)


def add_stutter(text: str, rng: random.Random) -> str:
    """Repeat a word, the way people restart a phrase mid-sentence."""
    words = text.split()
    if len(words) < 3:
        return text
    i = rng.randrange(len(words) - 1)
    words.insert(i, words[i])
    return " ".join(words)


def capitalise(text: str) -> str:
    """Sentence case with a terminal stop -- the target output shape."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


PROPER = set(banks.DAYS + banks.MONTHS + banks.PLACES + banks.NAMES)


def restore_proper_nouns(text: str) -> str:
    """Capitalise names, days, months and places in the target text.

    The model has to learn this: a transcript says "friday" and a written
    sentence says "Friday", and getting it wrong makes every output look
    subtly machine-made.
    """

    def fix(match):
        word = match.group(0)
        return word.capitalize() if word.lower() in PROPER else word

    return re.sub(r"\b[a-z]+\b", fix, text)


# ---------------------------------------------------------------------------
# Correction shapes
# ---------------------------------------------------------------------------


def render_correction(rng: random.Random) -> tuple:
    """One self-correction case: (dictated, intended).

    Varies four things independently, because a model trained on one shape
    generalises to the others poorly: which slot type is corrected, how the
    speaker signals it, whether they restate the whole clause or just the
    replacement, and where in the sentence it happens.
    """
    use_verb = rng.random() < 0.12
    if use_verb:
        frame = rng.choice(banks.VERB_FRAMES)
        wrong, right = rng.choice(banks.VERBS_TRANSITIVE)
        if rng.random() < 0.5:
            wrong, right = right, wrong
        tail = ""
    else:
        frame, kind, tail = rng.choice(banks.FRAMES)
        wrong, right = banks.two_distinct(rng, banks.BANKS[kind])
        tail = fill_tail(tail, rng)

    said_wrong = frame.format(slot=wrong)
    said_right = frame.format(slot=right)

    style = rng.random()
    if style < 0.40:
        # Elliptical: signal then the bare replacement.
        signal = rng.choice(banks.SIGNALS_ELLIPTICAL)
        dictated = f"{said_wrong} {signal} {right}"
    elif style < 0.75:
        # Full restatement of the clause.
        signal = rng.choice(banks.SIGNALS_FULL)
        dictated = f"{said_wrong} {signal} {said_right}"
    elif style < 0.87:
        # "not X" then the correction -- the shape that trips models keying on
        # signal words, because the retracted value is stated twice.
        dictated = f"{said_wrong} not {wrong} {said_right}"
    else:
        # Correction announced before the sentence is finished.
        signal = rng.choice(banks.SIGNALS_FULL)
        dictated = f"{signal} not {wrong} {said_right}"

    if tail:
        dictated = f"{dictated} {tail}"
        said_right = f"{said_right} {tail}"

    # A second correction in the same utterance, occasionally.
    if rng.random() < 0.07 and not use_verb:
        frame2, kind2, _ = rng.choice(banks.FRAMES)
        w2, r2 = banks.two_distinct(rng, banks.BANKS[kind2])
        sig2 = rng.choice(banks.SIGNALS_ELLIPTICAL)
        dictated = f"{dictated} and {frame2.format(slot=w2)} {sig2} {r2}"
        said_right = f"{said_right} and {frame2.format(slot=r2)}"

    if rng.random() < 0.35:
        opener = rng.choice(banks.OPENERS)
        dictated = f"{opener} {dictated}"

    return finish(dictated, said_right, rng)


def fill_tail(tail: str, rng: random.Random) -> str:
    """Resolve any placeholders in a frame's trailing clause."""
    if not tail:
        return ""
    for kind, bank in banks.BANKS.items():
        token = "{" + kind + "}"
        while token in tail:
            tail = tail.replace(token, rng.choice(bank), 1)
    return tail


def render_preservation(rng: random.Random) -> tuple:
    """A case with nothing to cut: only fillers and punctuation may change."""
    frame = rng.choice(banks.PRESERVE_FRAMES)
    intended = fill_preserve(frame, rng)
    dictated = intended
    if rng.random() < 0.4:
        opener = rng.choice(banks.OPENERS)
        dictated = f"{opener} {dictated}"
        intended = f"{opener} {intended}"
    return finish(dictated, intended, rng)


def fill_preserve(frame: str, rng: random.Random) -> str:
    """Fill a preservation frame, keeping paired slots distinct."""
    used = {}
    for kind, bank in banks.BANKS.items():
        for suffix in ("", "2"):
            token = "{" + kind + suffix + "}"
            if token in frame:
                choices = [v for v in bank if v not in used.get(kind, set())]
                value = rng.choice(choices or bank)
                used.setdefault(kind, set()).add(value)
                frame = frame.replace(token, value)
    return frame


def finish(dictated: str, intended: str, rng: random.Random) -> tuple:
    """Apply the disfluency and formatting layers shared by every case."""
    if rng.random() < 0.55:
        dictated = sprinkle_fillers(dictated, rng, rate=rng.uniform(0.04, 0.16))
    if rng.random() < 0.18:
        dictated = add_stutter(dictated, rng)
    dictated = strip_punct(dictated)
    # A minority keep ASR-style punctuation so the model does not learn that
    # unpunctuated input is the only thing worth editing.
    if rng.random() < 0.15:
        dictated = capitalise(dictated)
    target = capitalise(restore_proper_nouns(intended.strip()))
    return dictated, target


# ---------------------------------------------------------------------------
# Verifier examples, derived from editor examples
# ---------------------------------------------------------------------------


def corrupt(dictated: str, correct: str, rng: random.Random):
    """Damage a correct edit in one of the ways the layer actually fails.

    Returns None when the shape does not admit that corruption, so the caller
    can try another rather than emitting a mislabelled pair.
    """
    kind = rng.choice(
        ["keep_retracted", "swap", "negate", "drop_tail", "invent", "number"]
    )
    words = correct.split()

    if kind == "keep_retracted":
        # The single most important corruption: the edit kept the value the
        # speaker rejected. Recoverable only by reading both versions.
        for bank in banks.BANKS.values():
            present = [v for v in bank if v in correct.lower()]
            said = [v for v in bank if v in dictated.lower() and v not in present]
            if present and said:
                bad = correct.lower().replace(
                    rng.choice(present), rng.choice(said), 1
                )
                return capitalise(restore_proper_nouns(bad))
        return None

    if kind == "swap" and len(words) >= 6:
        i, j = sorted(rng.sample(range(len(words)), 2))
        if words[i].lower() == words[j].lower():
            return None
        words[i], words[j] = words[j], words[i]
        return " ".join(words)

    if kind == "negate":
        for verb in ("is", "are", "was", "were", "will", "should", "can"):
            token = f" {verb} "
            if token in correct.lower():
                idx = correct.lower().index(token)
                return correct[: idx + len(token) - 1] + " not" + correct[idx + len(token) - 1 :]
        return None

    if kind == "drop_tail" and len(words) >= 8:
        cut = rng.randrange(len(words) // 2, len(words) - 1)
        return capitalise(" ".join(words[:cut]))

    if kind == "invent":
        addition = rng.choice(
            [
                "and everyone agreed",
                "which nobody objected to",
                "and the client signed off",
                "after the budget was approved",
                "because the deadline moved",
            ]
        )
        return capitalise(correct.rstrip(".") + " " + addition)

    if kind == "number":
        for value in banks.NUMBERS + banks.MONEY + banks.ORDINALS:
            if value in correct.lower():
                pool = [
                    v
                    for v in (banks.NUMBERS + banks.MONEY + banks.ORDINALS)
                    if v != value
                ]
                return capitalise(correct.lower().replace(value, rng.choice(pool), 1))
        return None

    return None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

EDITOR_SYSTEM = (
    "You edit dictated speech into what the speaker meant to write.\n"
    "\n"
    "- Remove filler words, hesitations and repeated false starts.\n"
    "- If the speaker corrects themselves, delete the wording they abandoned "
    "along with the phrase that signalled the change, and keep only what they "
    "settled on. What gets replaced may be a name, a date, a time, a number, a "
    "place, a verb, or a whole phrase.\n"
    "- Fix punctuation and capitalisation.\n"
    "- If the speaker did not take anything back, keep every word.\n"
    "- Never answer the text, comment on it, or add anything of your own.\n"
    "\n"
    "Reply with the edited text and nothing else."
)

VERIFIER_SYSTEM = (
    "A dictated transcript has been edited. Decide whether the edited version "
    "still says what the speaker meant.\n"
    "\n"
    "Deleting wording the speaker retracted is correct and expected; the edited "
    "version will be shorter. Answer CHANGED only when the edit got the "
    "speaker's final choice wrong, swapped a name, date, time or number, "
    "reversed a statement, invented a claim, or dropped something the speaker "
    "never took back.\n"
    "\n"
    "Reply with exactly one word: SAME or CHANGED."
)


def build(n: int, rng: random.Random, correction_ratio: float):
    """Generate `n` editor examples and the verifier examples they imply."""
    editor, verifier = [], []
    seen = set()
    attempts = 0
    while len(editor) < n and attempts < n * 20:
        attempts += 1
        if rng.random() < correction_ratio:
            dictated, target = render_correction(rng)
            is_correction = True
        else:
            dictated, target = render_preservation(rng)
            is_correction = False

        if dictated in seen:
            continue
        seen.add(dictated)
        editor.append(
            {
                "system": EDITOR_SYSTEM,
                "input": dictated,
                "output": target,
                "kind": "correction" if is_correction else "preserve",
            }
        )

        # SAME: the edit that actually happened.
        verifier.append(
            {
                "system": VERIFIER_SYSTEM,
                "input": f"Original: {dictated}\nEdited: {target}",
                "output": "SAME",
                "kind": "same",
            }
        )
        # CHANGED: the same edit, damaged. Only when a corruption applies, so
        # the labels stay trustworthy.
        bad = corrupt(dictated, target, rng)
        if bad and bad.strip().lower() != target.strip().lower():
            verifier.append(
                {
                    "system": VERIFIER_SYSTEM,
                    "input": f"Original: {dictated}\nEdited: {bad}",
                    "output": "CHANGED",
                    "kind": "changed",
                }
            )
    return editor, verifier


def write(path: pathlib.Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {path}  {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--correction-ratio", type=float, default=0.6,
                    help="fraction of examples containing a real self-correction")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = pathlib.Path(args.out)

    # Separate seeds, and the eval split generated first, so a re-run with a
    # bigger --train cannot quietly shuffle examples across the boundary.
    eval_rng = random.Random(args.seed + 99991)
    ev_editor, ev_verifier = build(args.eval, eval_rng, args.correction_ratio)
    eval_inputs = {row["input"] for row in ev_editor}

    train_rng = random.Random(args.seed)
    tr_editor, tr_verifier = build(args.train, train_rng, args.correction_ratio)

    # Leakage guard. Both splits come from one generator, so overlap is
    # expected and has to be removed rather than assumed away.
    before = len(tr_editor)
    tr_editor = [r for r in tr_editor if r["input"] not in eval_inputs]
    dropped = before - len(tr_editor)

    print("generated:")
    write(out / "editor_train.jsonl", tr_editor)
    write(out / "editor_eval.jsonl", ev_editor)
    write(out / "verifier_train.jsonl", tr_verifier)
    write(out / "verifier_eval.jsonl", ev_verifier)

    corrections = sum(1 for r in tr_editor if r["kind"] == "correction")
    changed = sum(1 for r in tr_verifier if r["kind"] == "changed")
    print(
        f"\n  editor train: {len(tr_editor)}  "
        f"({corrections} corrections / {len(tr_editor) - corrections} preserve)"
    )
    print(
        f"  verifier train: {len(tr_verifier)}  "
        f"({changed} CHANGED / {len(tr_verifier) - changed} SAME)"
    )
    print(f"  dropped {dropped} train rows that collided with the eval split")
    print(
        "\n  NOTE: this eval split shares a generator with training data, so a\n"
        "  high score on it proves the model learned these templates, not that\n"
        "  it can edit speech. Judge with scripts/enhance-eval/suite.py, and\n"
        "  ideally with your own recorded dictation."
    )


if __name__ == "__main__":
    main()

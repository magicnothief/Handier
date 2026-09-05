"""Build the fine-tuning corpus for the enhancement layer.

Produces an editor corpus, and optionally a verifier corpus:

  editor   raw transcript -> cleaned transcript
  verifier (original, edited) -> SAME / CHANGED

`--editor-only` skips the verifier half, and is the recommended setting: the
verify pass measured net-negative against a trained editor -- 0 of 10 real
mistakes caught, one correct edit destroyed -- so the whole budget is better
spent on editing. See ../enhance-eval/README.md.

Sources
-------
Real, human-annotated:
  * google-research-datasets/disfl_qa (CC-BY-4.0) -- 11.8K questions where an
    annotator inserted a *contextually plausible* retraction, e.g. "What is the
    second level of territorial division in Poland no make that the basic unit
    of territorial division in Warsaw?". The distractor is drawn from the source
    passage, which makes these considerably harder than a random swap, and the
    signal phrases are the same family the app targets.
  * nyralabs/disfluency_speech_english (Apache-2.0) -- 5K verbatim/intended
    transcript pairs from real speech, covering fillers, repetitions and
    cutoffs, which disfl_qa does not contain.
  * amaai-lab/DisfluencySpeech (Apache-2.0) -- 5K Switchboard utterances
    published at three levels of cleanup; the verbatim and fully-cleaned
    columns are used as an editor pair. These are *declarative*, which matters:
    disfl_qa is entirely questions, and a fine-tune on it alone learns that the
    answer is always a question.

Synthetic, generated here:
  * retractions over templated dictation-shaped sentences and over sentences
    lifted from disfl_qa's Wikipedia passages
  * filler and stutter injection
  * negatives -- text that must come back untouched

Why synthesise at all when real data exists: no public dataset contains the
negative half. Every disfluency corpus is a corpus of things to remove, so a
model trained only on them learns that editing is always correct. The parent
project measured over-eager cutting as the dominant failure, so a quarter of
this corpus is text where the right answer is to change almost nothing.

Usage
-----
    python build_dataset.py --out D:/dev/handy-train-editor --n 90000         --system-mode none --editor-only

    python build_dataset.py --out D:/dev/handy-train --n 90000 --no-hub
        (skip the Hub downloads; synthetic only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "enhance-eval"))

import taxonomy as tx  # noqa: E402

# ---------------------------------------------------------------------------
# Holdout
# ---------------------------------------------------------------------------
# The eval suite must not appear in training. A previous round of this project
# reported a model as working when the smoke test's cases were verbatim the
# few-shot examples in its own prompt -- it measured memorisation and called it
# capability. Exact inputs from the suite are dropped here, and the README says
# plainly that the suite is now in-distribution and a fresh held-out set is
# needed to claim an improvement.
def load_suite_inputs() -> set[str]:
    try:
        import suite  # type: ignore
    except Exception as e:  # pragma: no cover - the suite is optional here
        print(f"  ! could not import the eval suite ({e}); holdout disabled")
        return set()
    return {normalise(text) for _id, text, *_rest in suite.ALL}


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    out = (text or "").lower()
    out = "".join(c if (c.isalnum() or c.isspace() or c == "'") else " " for c in out)
    return " ".join(out.split())


# ---------------------------------------------------------------------------
# The prompt the model is trained against
# ---------------------------------------------------------------------------
# Deliberately the *shipped* prompt, dumped from the Rust source, so a
# fine-tuned model is a drop-in replacement and no change to prompt.rs is
# needed to try it. `--short-prompt` trains against a terse instruction
# instead, which is faster at inference but requires editing prompt.rs to match.
SHORT_EDITOR_PROMPT = "Edit the dictated transcript into what the speaker meant to write."
SHORT_VERIFY_PROMPT = "Did the edit change what the speaker meant? Answer SAME or CHANGED."


def load_prompt(path: Path | None, fallback: str) -> str:
    if path and path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


# ---------------------------------------------------------------------------
# ASR shaping
# ---------------------------------------------------------------------------
def to_asr(text: str, rng: random.Random) -> str:
    """Make a clean sentence look like something that came out of an ASR model.

    Both shapes are produced across the corpus rather than one: Whisper emits
    punctuation and capitals, Parakeet and friends often do not, and the app
    supports both. A model trained on only one shape gets brittle about the
    other.
    """
    if rng.random() < 0.5:
        return text.strip()
    out = text.lower()
    out = re.sub(r"[.,;:!?]", "", out)
    return " ".join(out.split())


def stutter(text: str, rng: random.Random) -> str:
    """Repeat a word, the way people do when they hesitate."""
    words = text.split()
    if len(words) < 4:
        return text
    i = rng.randrange(0, min(4, len(words)))
    return " ".join(words[:i] + [words[i]] + words[i:])


def sprinkle_fillers(text: str, rng: random.Random, n: int) -> str:
    words = text.split()
    for _ in range(n):
        filler = rng.choice(tx.FILLERS)
        i = rng.randrange(0, len(words) + 1)
        words[i:i] = filler.split()
    out = " ".join(words)
    if rng.random() < 0.5:
        out = f"{rng.choice(tx.LEADING_FILLERS)} {out}"
    return out


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------
def make_retraction(rng: random.Random) -> tuple[str, str]:
    """A sentence where the speaker replaces one entity with another.

    Returns (raw, clean). The clean side keeps only the second choice, which is
    the whole point of the feature.
    """
    frame, kind = rng.choice(tx.FRAMES)
    wrong, right = tx.pick_two(rng, tx.ENTITY_KINDS[kind])
    head, tail = rng.choice(tx.HEADS), rng.choice(tx.TAILS)

    if rng.random() < 0.2:
        signal = rng.choice(tx.REPLACEMENT_SIGNALS)
        # "book it for four people make that six people"
        raw_mid = f"{frame.format(slot=wrong)} {signal} {right}"
    else:
        signal = rng.choice(tx.RETRACTION_SIGNALS)
        if rng.random() < 0.5:
            # Repeat the frame: "the meeting is on friday no wait it's on saturday"
            raw_mid = f"{frame.format(slot=wrong)} {signal} {frame.format(slot=right)}"
        else:
            raw_mid = f"{frame.format(slot=wrong)} {signal} {right}"

    raw = f"{head}{raw_mid}{tail}"
    clean = f"{head}{frame.format(slot=right)}{tail}"

    # Fillers and stutters ride along on some of them, because in real
    # dictation a correction rarely arrives on its own.
    if rng.random() < 0.45:
        raw = sprinkle_fillers(raw, rng, rng.randint(1, 2))
    if rng.random() < 0.15:
        raw = stutter(raw, rng)

    return to_asr(raw, rng), tx.sentence_case(clean)


def make_double_retraction(rng: random.Random) -> tuple[str, str]:
    """Two corrections in one utterance, or one correction applied twice."""
    if rng.random() < 0.5:
        a_raw, a_clean = make_retraction(rng)
        b_raw, b_clean = make_retraction(rng)
        raw = f"{a_raw} and {b_raw}"
        clean = f"{a_clean.rstrip('.')} and {b_clean[0].lower()}{b_clean[1:]}"
        return raw, tx.sentence_case(clean)

    # "friday no wait saturday no actually sunday" -- only the last survives.
    frame, kind = rng.choice(tx.FRAMES)
    pool = tx.ENTITY_KINDS[kind]
    if len(pool) < 3:
        return make_retraction(rng)
    first, second, third = rng.sample(pool, 3)
    s1, s2 = rng.choice(tx.RETRACTION_SIGNALS), rng.choice(tx.RETRACTION_SIGNALS)
    raw = f"{frame.format(slot=first)} {s1} {second} {s2} {third}"
    return to_asr(raw, rng), tx.sentence_case(frame.format(slot=third))


def make_long_retraction(rng: random.Random, sentences: list[str]) -> tuple[str, str]:
    """A multi-sentence dictation whose correction reaches back several sentences.

    The corpus was otherwise entirely single utterances -- 162 tokens at the
    longest -- while the app sends whole paragraphs in one request and never
    chunks them. A model trained only on one-liners has never seen the case it
    will actually meet.

    Long range is the point, not just length. A retraction at the end of a
    paragraph has to be matched against an entity introduced sentences earlier,
    so the wrong wording cannot be found by looking at the neighbouring clause.
    Everything the speaker did not take back has to survive intact, which also
    makes these the strictest preservation examples in the corpus.
    """
    frame, kind = rng.choice(tx.FRAMES)
    wrong, right = tx.pick_two(rng, tx.ENTITY_KINDS[kind])

    def filler_sentence() -> str:
        if sentences and rng.random() < 0.5:
            return rng.choice(sentences).rstrip(".")
        other, other_kind = rng.choice(tx.FRAMES)
        return other.format(slot=rng.choice(tx.ENTITY_KINDS[other_kind]))

    # One head, used on both sides. Drawing it twice made the raw open with
    # "discussed" and the target with "Quick note," -- a difference the model
    # would have to invent.
    head = rng.choice(tx.HEADS)
    opening = f"{head}{frame.format(slot=wrong)}"
    middle = [filler_sentence() for _ in range(rng.randint(1, 4))]

    signal = rng.choice(tx.RETRACTION_SIGNALS)
    if rng.random() < 0.5:
        correction = f"{signal} {frame.format(slot=right)}"
    else:
        correction = f"{signal} {right}"

    raw_parts = [opening] + middle + [correction]
    # The corrected sentence is restated in place; the abandoned wording and the
    # signal phrase both disappear, and the sentences between them do not move.
    clean_parts = [f"{head}{frame.format(slot=right)}"] + middle

    def join(parts: list[str]) -> str:
        # Every sentence gets capitalised, not just the first: the target is the
        # written form, and a paragraph that runs "February. trial runs..." would
        # teach the model to drop capitals after a full stop.
        cleaned = [p.strip().rstrip(".").strip() for p in parts]
        return ". ".join(tx.sentence_case(p).rstrip(".") for p in cleaned if p) + "."

    raw = ". ".join(p.strip().rstrip(".").strip() for p in raw_parts if p.strip()) + "."
    clean = join(clean_parts)

    if rng.random() < 0.6:
        raw = sprinkle_fillers(raw, rng, rng.randint(1, 3))
    if rng.random() < 0.25:
        raw = stutter(raw, rng)
    return to_asr(raw, rng), tx.sentence_case(clean)


# ---------------------------------------------------------------------------
# Targeted generators
# ---------------------------------------------------------------------------
# Each of these exists because the trained editor got that exact pattern wrong.
# They are worth more than raw volume: at a ~1% real error rate the model already
# has the common shapes, and adding another 45K template rows mostly teaches it
# what it knows. See ../enhance-eval/README.md for the measurements.

# Adjuncts that open a sentence and must survive. The model dropped
# "Between Bingen and Bonn," from an otherwise correct edit.
# Adjectives that qualify a document, so the correction has a same-sentence
# distractor to be applied to by mistake.
SLOT_QUALIFIERS = [
    "quarterly", "monthly", "annual", "final", "draft", "revised", "internal",
    "updated", "original", "interim", "preliminary", "consolidated",
]

LEADING_ADJUNCTS = [
    "between the two offices", "after the review", "before the deadline",
    "in the second quarter", "on the client call", "at the London site",
    "during the migration", "under the new policy", "for the audit",
    "across both teams", "since the last release", "by the end of the month",
    "without the vendor", "apart from the invoice", "beyond the pilot",
]


def make_leading_adjunct(rng: random.Random, sentences: list[str]) -> tuple[str, str]:
    """A sentence-initial phrase that carries content and must not be cut.

    The measured failure was a silent deletion: the edit was otherwise perfect
    and simply lost the opening phrase, which no length check would flag because
    the result still reads as a fluent sentence.
    """
    adjunct = rng.choice(LEADING_ADJUNCTS)
    frame, kind = rng.choice(tx.FRAMES)
    body = frame.format(slot=rng.choice(tx.ENTITY_KINDS[kind]))
    clean = f"{adjunct}, {body}{rng.choice(tx.TAILS)}"
    raw = clean
    if rng.random() < 0.7:
        raw = sprinkle_fillers(raw, rng, rng.randint(1, 2))
    if rng.random() < 0.3:
        raw = stutter(raw, rng)
    return to_asr(raw, rng), tx.sentence_case(clean)


def make_false_start(rng: random.Random, sentences: list[str]) -> tuple[str, str]:
    """An abandoned pronoun or article before the real subject.

    "Yeah, we you'd have to just sit and wait" -- the "we" is a false start and
    goes; the model kept it. Distinct from a stutter, because the abandoned word
    is not a repeat of what follows, so a duplicate-word rule cannot catch it.
    """
    frame, kind = rng.choice(tx.FRAMES)
    body = frame.format(slot=rng.choice(tx.ENTITY_KINDS[kind]))
    clean = f"{body}{rng.choice(tx.TAILS)}"
    abandoned = rng.choice(
        ["we", "i", "they", "it", "he", "she", "you", "the", "a", "and", "so"]
    )
    raw = f"{abandoned} {clean}"
    if rng.random() < 0.5:
        raw = f"{rng.choice(['yeah', 'okay', 'right', 'well'])} {raw}"
        clean = clean
    if rng.random() < 0.5:
        raw = sprinkle_fillers(raw, rng, 1)
    return to_asr(raw, rng), tx.sentence_case(clean)


def make_initial_retraction(rng: random.Random, sentences: list[str]) -> tuple[str, str]:
    """The retraction opens the utterance, before the thing being corrected.

    "no wait not friday the meeting is on saturday" -- the abandoned value is
    named inside the signal phrase rather than in a clause of its own, and the
    model kept it as "Not Friday, the meeting is on Saturday."
    """
    frame, kind = rng.choice(tx.FRAMES)
    wrong, right = tx.pick_two(rng, tx.ENTITY_KINDS[kind])
    signal = rng.choice(tx.RETRACTION_SIGNALS)
    shape = rng.random()
    if shape < 0.4:
        raw = f"{signal} not {wrong} {frame.format(slot=right)}"
    elif shape < 0.7:
        raw = f"{signal} {frame.format(slot=right)} not {wrong}"
    else:
        raw = f"not {wrong} {signal} {frame.format(slot=right)}"
    clean = frame.format(slot=right)
    if rng.random() < 0.4:
        raw = sprinkle_fillers(raw, rng, 1)
    return to_asr(raw, rng), tx.sentence_case(clean)


def make_slot_confusion(rng: random.Random, sentences: list[str]) -> tuple[str, str]:
    """Two slots of different kinds; the correction must land on the right one.

    The measured failure: "send the quarterly report to the finance team,
    actually send it to the audit team" came back as "send the audit report to
    the finance team". The replacement was applied to the wrong slot, which
    silently changes what the speaker asked for.
    """
    kinds = [k for k in ("name", "team", "place", "room")
             if len(tx.ENTITY_KINDS.get(k, [])) >= 3]
    if not kinds:
        return make_retraction(rng)
    # The kept slot is an adjective, not an entity: entity banks contain values
    # that already carry an article ("the annex"), which produced "send the the
    # annex report", and a date reads oddly as a document qualifier.
    keep = rng.choice(SLOT_QUALIFIERS)
    thing = rng.choice(["report", "invoice", "summary", "deck", "draft", "spec"])
    fix_kind = rng.choice(kinds)
    wrong, right = tx.pick_two(rng, tx.ENTITY_KINDS[fix_kind])

    def article(value: str) -> str:
        return value if value.startswith("the ") else value

    body = f"send the {keep} {thing} to {article(wrong)}"
    signal = rng.choice(tx.RETRACTION_SIGNALS)
    if rng.random() < 0.5:
        raw = f"{body} {signal} send it to {article(right)} instead"
    else:
        raw = f"{body} {signal} to {article(right)}"
    clean = f"send the {keep} {thing} to {article(right)}"
    if rng.random() < 0.5:
        raw = sprinkle_fillers(raw, rng, rng.randint(1, 2))
    return to_asr(raw, rng), tx.sentence_case(clean)


def long_form_from_speech(df, rng: random.Random, want: int, splits: set[str]):
    """Real multi-sentence passages, rebuilt from consecutive source utterances.

    DisfluencySpeech publishes one utterance per row, but the audio filename is a
    running index over the source corpus, so rows with adjacent indices were
    adjacent speech. Joining a run of them recovers a genuine paragraph of
    conversational dictation -- real hesitations, real repairs, real cleanup --
    rather than a synthesised approximation of one.

    Runs are cut wherever the index is not contiguous. That does not guarantee a
    single speaker or topic, because the published corpus does not mark call
    boundaries; a window can therefore change subject partway. That is harmless
    here and arguably useful: the target keeps every sentence, so those windows
    become long examples whose correct answer is to delete nothing.
    """
    if df is None or len(df) == 0:
        return []

    rows = df[df["split"].isin(splits)].sort_values("idx")
    runs: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    previous = None
    for idx, a, c in zip(rows["idx"], rows["transcript_a"], rows["transcript_c"]):
        if previous is not None and idx != previous + 1:
            if len(current) > 1:
                runs.append(current)
            current = []
        current.append((str(a), str(c)))
        previous = idx
    if len(current) > 1:
        runs.append(current)

    out: list[tuple[str, str]] = []
    for run in runs:
        i = 0
        while i < len(run) - 1 and len(out) < want:
            size = rng.randint(2, 6)
            window = run[i : i + size]
            i += size
            if len(window) < 2:
                break
            raws = [tidy_target(a) for a, _ in window]
            cleans = [tidy_target(c) for _, c in window]
            if not all(raws) or not all(cleans):
                continue
            # One unusable sentence would teach a dangling clause in the middle
            # of an otherwise good paragraph, which is harder to spot than a
            # short bad row.
            if not all(target_is_usable(c) for _, c in window):
                continue
            out.append((" ".join(raws), " ".join(cleans)))
        if len(out) >= want:
            break
    rng.shuffle(out)
    return out[:want]


def make_filler_only(rng: random.Random, sentences: list[str]) -> tuple[str, str]:
    """Nothing retracted; only hesitation noise to strip."""
    clean = rng.choice(sentences) if sentences and rng.random() < 0.5 else None
    if clean is None:
        frame, kind = rng.choice(tx.FRAMES)
        entity = rng.choice(tx.ENTITY_KINDS[kind])
        clean = f"{rng.choice(tx.HEADS)}{frame.format(slot=entity)}{rng.choice(tx.TAILS)}"
    raw = sprinkle_fillers(clean, rng, rng.randint(1, 3))
    if rng.random() < 0.3:
        raw = stutter(raw, rng)
    return to_asr(raw, rng), tx.sentence_case(clean)


def make_negative(rng: random.Random, sentences: list[str]) -> tuple[str, str]:
    """Text whose correct output is itself, modulo casing and punctuation.

    Half use retraction vocabulary innocently -- the adversarial case that
    matters -- and half are ordinary clean sentences.
    """
    if rng.random() < 0.55:
        template = rng.choice(tx.INNOCENT_SIGNAL_USES)
        d1, d2 = tx.pick_two(rng, tx.DAYS)
        n1, n2 = tx.pick_two(rng, tx.NAMES)
        num1, num2 = tx.pick_two(rng, tx.SMALL_NUMBERS)
        m1, m2 = tx.pick_two(rng, tx.MONEY)
        o1, o2 = tx.pick_two(rng, tx.ORDINALS)
        clean = template.format(
            day=d1, day2=d2, name=n1, name2=n2, num=num1, num2=num2,
            money=m1, money2=m2, ordinal=o1, ordinal2=o2,
            month=rng.choice(tx.MONTHS), place=rng.choice(tx.PLACES),
            room=rng.choice(tx.ROOMS), time=rng.choice(tx.TIMES),
            team=rng.choice(tx.TEAMS),
        )
    elif sentences and rng.random() < 0.6:
        clean = rng.choice(sentences)
    else:
        frame, kind = rng.choice(tx.FRAMES)
        entity = rng.choice(tx.ENTITY_KINDS[kind])
        clean = f"{frame.format(slot=entity)}{rng.choice(tx.TAILS)}"

    raw = clean
    # A negative may still carry fillers; removing those is correct, and
    # keeping the content is what is being taught.
    if rng.random() < 0.4:
        raw = sprinkle_fillers(clean, rng, 1)
    return to_asr(raw, rng), tx.sentence_case(clean)


# ---------------------------------------------------------------------------
# Hub sources
# ---------------------------------------------------------------------------
def fetch_disfl_qa(cache: Path) -> list[tuple[str, str, str]]:
    """(disfluent, fluent, context) triples from disfl_qa's train + validation."""
    import pandas as pd

    base = "https://huggingface.co/datasets/google-research-datasets/disfl_qa/resolve/refs%2Fconvert%2Fparquet/default"
    rows: list[tuple[str, str, str]] = []
    for split in ("train", "validation"):
        target = cache / f"disfl_qa_{split}.parquet"
        if not target.exists():
            url = f"{base}/{split}/0000.parquet"
            print(f"  downloading disfl_qa/{split} ...")
            pd.read_parquet(url).to_parquet(target)
        df = pd.read_parquet(target)
        for _, r in df.iterrows():
            rows.append(
                (str(r["disfluent question"]), str(r["original question"]), str(r["context"]))
            )
    print(f"  disfl_qa: {len(rows)} pairs")
    return rows


DISFLUENCY_SPEECH_SPLITS = (("train", 3), ("validation", 1), ("test", 1))

# How many real multi-sentence passages to build. Capped rather than unbounded:
# each one consumes several utterances that would otherwise be single examples,
# and the windows overlap in subject matter, so more of them buys less than the
# row count suggests.
LONG_FORM_REAL_TARGET = 1200


def fetch_disfluency_speech(cache: Path):
    """(verbatim, cleaned) pairs from amaai-lab/DisfluencySpeech (Apache-2.0).

    5K utterances re-recorded from Switchboard, published with three transcripts
    at increasing levels of cleanup. `transcript_a` keeps every filler and
    repair; `transcript_c` is the intended sentence. That is precisely this
    layer's task, on real conversational speech.

    Worth more than its size suggests. disfl_qa is 11.8K *questions* lifted from
    SQuAD, and a model trained on it alone learns that the answer is always a
    question -- measured on a fine-tune of LFM2.5-350M, which answered "we we
    need to to check the logs" with "What is the name of the log file?". These
    rows are declarative, which is what dictation actually looks like.

    The published targets are not consistently sentence-cased, so they are
    lightly normalised rather than used raw; teaching the model to emit a
    lowercase sentence start would undo the punctuation half of the task.

    The audio filename is a running index over the source corpus, and it is kept
    because it is the only ordering information published. Rows whose indices are
    adjacent were adjacent utterances, which is what makes real multi-sentence
    passages possible -- see `long_form_from_speech`.
    """
    target = cache / "disfluency_speech_text.parquet"
    if target.exists():
        import pandas as pd

        df = pd.read_parquet(target)
        print(f"  disfluency_speech: {len(df)} rows (cached)")
        return df

    try:
        import pandas as pd
        import pyarrow.parquet as pq

        base = (
            "https://huggingface.co/datasets/amaai-lab/DisfluencySpeech/"
            "resolve/refs%2Fconvert%2Fparquet/default"
        )
        frames = []
        for split, files in DISFLUENCY_SPEECH_SPLITS:
            for i in range(files):
                url = f"{base}/{split}/{i:04d}.parquet"
                print(f"  reading DisfluencySpeech/{split}/{i:04d} (text columns only) ...")
                # `audio.path` projects the struct field without pulling the
                # 16 kHz audio beside it; the train split alone is 1.3 GB.
                table = pq.read_table(
                    url,
                    columns=["audio.path", "transcript_a", "transcript_c"],
                    filesystem=_http_fs(),
                )
                frame = table.to_pandas()
                frame.columns = ["path", "transcript_a", "transcript_c"]
                frame["split"] = split
                frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df["idx"] = (
            df["path"].str.replace(".wav", "", regex=False).astype("int64")
        )
        df = df.sort_values("idx").reset_index(drop=True)
        df.to_parquet(target)
        print(f"  disfluency_speech: {len(df)} rows")
        return df
    except Exception as e:
        print(f"  ! DisfluencySpeech unavailable ({e}); continuing without it")
        import pandas as pd

        return pd.DataFrame(columns=["path", "transcript_a", "transcript_c", "split", "idx"])


# Leading and trailing turn markers that survive the published transcripts.
_EDGE_JUNK = re.compile(r"^[\s\-]+|[\s\-]+$")


def tidy_spoken(text: str) -> str:
    """Sentence-case a published transcript without rewording it."""
    text = _EDGE_JUNK.sub("", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # A comma cannot precede a new sentence; the source has ", It's like" where
    # a full stop belongs, and copying that would teach the wrong punctuation.
    text = re.sub(r",\s+([A-Z])", lambda m: ". " + m.group(1), text)
    return text[0].upper() + text[1:]


def fetch_nyra(cache: Path) -> list[tuple[str, str]]:
    """(verbatim, intended) pairs, text columns only.

    The published parquet carries 16 kHz audio and runs to about a gigabyte, so
    the two text columns are projected out rather than pulling the whole file.
    """
    target = cache / "nyra_text.parquet"
    if target.exists():
        import pandas as pd

        df = pd.read_parquet(target)
        pairs = list(zip(df["verbatim_transcript"], df["intended_transcript"]))
        print(f"  nyralabs: {len(pairs)} pairs (cached)")
        return pairs

    try:
        import pyarrow.parquet as pq
        import pandas as pd

        base = "https://huggingface.co/datasets/nyralabs/disfluency_speech_english/resolve/refs%2Fconvert%2Fparquet/default"
        frames = []
        for split, files in (("train", 2), ("validation", 1), ("test", 1)):
            for i in range(files):
                url = f"{base}/{split}/{i:04d}.parquet"
                print(f"  reading nyralabs/{split}/{i:04d} (text columns only) ...")
                table = pq.read_table(
                    url,
                    columns=["verbatim_transcript", "intended_transcript"],
                    filesystem=_http_fs(),
                )
                frames.append(table.to_pandas())
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(target)
        pairs = list(zip(df["verbatim_transcript"], df["intended_transcript"]))
        print(f"  nyralabs: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  ! nyralabs unavailable ({e}); continuing without it")
        return []


def _http_fs():
    """An fsspec HTTP filesystem for pyarrow.

    pyarrow used to accept an `https://` string directly; since 21.x it raises
    `Unrecognized filesystem type in URI` instead. Handing it an explicit
    filesystem keeps the column projection, which is the only reason these
    fetchers are affordable: the published parquet carries 16 kHz audio and the
    text columns are a rounding error beside it.
    """
    import fsspec

    return fsspec.filesystem("http")


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences_from_contexts(rows, limit: int = 60000) -> list[str]:
    """Declarative sentences lifted from disfl_qa's Wikipedia passages.

    Templated sentences alone would teach the model a handful of shapes. These
    supply real prose with real names, dates and numbers -- exactly the material
    a retraction lands on -- at no extra licensing cost, since the passages ship
    with a dataset already being used.
    """
    seen: set[str] = set()
    out: list[str] = []
    for _dis, _flu, context in rows:
        for sent in SENT_SPLIT.split(context):
            sent = sent.strip()
            # Long encyclopaedic sentences are not dictation; short fragments
            # carry no content to preserve.
            if not (40 <= len(sent) <= 160):
                continue
            if sent.count(",") > 3 or "(" in sent:
                continue
            key = normalise(sent)
            if key in seen:
                continue
            seen.add(key)
            out.append(sent)
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# Verifier data
# ---------------------------------------------------------------------------
def corrupt(clean: str, raw: str, rng: random.Random) -> str | None:
    """Damage a correct edit in one of the ways the verifier must catch.

    Built from the *pair* rather than from scratch, so a corruption is always
    reachable from the original -- which is what makes it a hard negative
    instead of an obvious one.
    """
    words = clean.rstrip(".").split()
    if len(words) < 4:
        return None
    mode = rng.choice(["keep_wrong", "swap", "negate", "drop", "smuggle"])

    if mode == "keep_wrong":
        # The failure that matters most: the edit kept the abandoned choice.
        raw_words = set(normalise(raw).split())
        clean_words = set(normalise(clean).split())
        dropped = [w for w in raw_words - clean_words if len(w) > 3]
        if not dropped:
            return None
        victim = rng.choice([w for w in words if len(w) > 3] or words)
        return " ".join(
            rng.choice(dropped) if w == victim else w for w in words
        ) + "."

    if mode == "swap":
        idx = [i for i, w in enumerate(words) if len(w) > 3]
        if len(idx) < 2:
            return None
        i, j = rng.sample(idx, 2)
        words[i], words[j] = words[j], words[i]
        return " ".join(words) + "."

    if mode == "negate":
        for i, w in enumerate(words):
            if w.lower() in ("is", "are", "was", "were", "will", "should", "can"):
                words.insert(i + 1, "not")
                return " ".join(words) + "."
        return None

    if mode == "drop":
        if len(words) < 8:
            return None
        return " ".join(words[: len(words) // 2]) + "."

    # smuggle: a word that was never dictated, short enough to slip the
    # length guard in prompt.rs
    pool = tx.NAMES + tx.DAYS + tx.PLACES + tx.SMALL_NUMBERS
    intruder = rng.choice([w for w in pool if w not in normalise(clean)])
    i = rng.randrange(1, len(words))
    words.insert(i, intruder)
    return " ".join(words) + "."


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
# Published conversational transcripts are segmented by turn, not by sentence,
# so a row can end mid-clause or carry ", ." where a segment boundary fell. At
# ~1% of the corpus that is small, but it lands squarely on the punctuation half
# of the task: these are the rows that would teach the model to emit a dangling
# comma.
_PUNCT_FIXES = [
    (re.compile(r"\s+([,.;:!?])"), r"\1"),
    (re.compile(r",\s*\."), "."),
    (re.compile(r",\s*,+"), ","),
    (re.compile(r"\.\s*,"), "."),
    (re.compile(r"\s{2,}"), " "),
]

# Words no finished English sentence can end on, whatever punctuation follows.
# The list is deliberately short. "so", "for", "is", "was", "to" and "that" are
# all absent because they end ordinary sentences -- "I think so.", "that's what
# it was.", "I want to." -- and dropping those rows would cost more than the
# artefacts are worth.
_UNFINISHED = re.compile(
    r"\b(and|but|or|the|of|a|an|with|from|because)\s*[,.]?\s*$", re.I
)

# The weaker signal: a word that *can* end a sentence, left hanging on a comma.
# This has to be tested before `tidy_target` runs, because turning the trailing
# comma into a full stop is exactly what destroys the evidence.
_DANGLING = re.compile(r"\b(so|for|is|was|to|that|it|they|we|i|he|she)\s*,\s*$", re.I)


def target_is_usable(text: str) -> bool:
    """Whether a target is a finished sentence. Test before tidying, not after."""
    text = text.strip()
    return bool(text) and not _UNFINISHED.search(text) and not _DANGLING.search(text)


def tidy_target(text: str) -> str:
    """Repair segmentation artefacts in a target without rewording it."""
    for rx, rep in _PUNCT_FIXES:
        text = rx.sub(rep, text)
    text = text.strip()
    # A trailing comma is a segment boundary, not punctuation the speaker meant.
    return re.sub(r",+$", ".", text)


def chat(system: str, user: str, assistant: str) -> dict:
    """One training example, with the system turn omitted when it is empty.

    A fine-tune does not need the instruction: the behaviour is in the weights,
    and repeating ~450 tokens of prompt in front of a ~30-token example is most
    of the sequence length for no gain. Measured on the first fine-tune, giving
    it the prompt at inference time was actively worse -- 54/68 against 57/68 --
    because it started copying the prompt's own rules into its output.

    Omitting the turn rather than sending an empty one matters for matching what
    the app does. LFM2's chat template renders an empty system message and an
    absent one to byte-identical text, so a corpus built this way is what the
    sidecar produces when it is handed `system: ""`.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


# The instruction an Alpaca-format row carries, and the one the app sends at
# inference time. Read from a file rather than inlined because the Rust side
# `include_str!`s the same path -- a fine-tune served an instruction one word
# different from the one it was trained on degrades quietly, and a shared file
# makes that drift impossible rather than merely unlikely.
ALPACA_INSTRUCTION_PATH = Path(__file__).parent / "alpaca_instruction.txt"

# Verification is deprecated (measured net-negative against a trained editor), so
# this exists only for a --format alpaca build that also passes --editor-only off.
ALPACA_VERIFY_INSTRUCTION = (
    "Decide whether the edited version still says what the speaker meant. "
    "Reply with exactly one word: SAME or CHANGED."
)


def to_alpaca(record: dict, instruction: str) -> dict:
    """Re-shape one chat record as an Alpaca row.

    Converting at write time rather than building two parallel corpora keeps a
    single source of truth: the Alpaca files are provably the same examples as
    the chat files, not a second pipeline that can drift.
    """
    messages = record["messages"]
    return {
        "instruction": instruction,
        "input": messages[-2]["content"],
        "output": messages[-1]["content"],
    }


def alpaca_instruction_for(record: dict, editor: str) -> str:
    """Pick the instruction that matches what this row is asking for.

    A verify row is identifiable from its user turn alone -- it always opens
    `Original:` -- which is the same invariant `validate_dataset.py` asserts and
    the reason one model can serve both tasks.
    """
    user = record["messages"][-2]["content"]
    return ALPACA_VERIFY_INSTRUCTION if user.startswith("Original:") else editor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=100_000, help="editor examples to emit")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-hub", action="store_true", help="synthetic sources only")
    ap.add_argument("--short-prompt", action="store_true",
                    help="train against a terse instruction instead of the shipped prompt")
    ap.add_argument("--editor-prompt", type=Path,
                    default=Path(__file__).parent / "editor_prompt.txt")
    ap.add_argument("--verify-prompt", type=Path,
                    default=Path(__file__).parent / "verify_prompt.txt")
    ap.add_argument("--editor-only", action="store_true",
                    help="skip the verifier corpus; measured net-negative in the app")
    ap.add_argument("--system-mode", choices=("prompt", "none"), default="prompt",
                    help="'none' omits the system turn, for a fine-tune meant to be "
                         "run without one")
    ap.add_argument("--disfl-qa-cap", type=int, default=None,
                    help="max disfl_qa rows; default matches the declarative real rows")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--format", choices=("chatml", "alpaca", "both"), default="chatml",
                    help="chatml is the OpenAI messages array, for an instruct model "
                         "with a chat template; alpaca is instruction/input/output, "
                         "for a base model that has none")
    ap.add_argument("--alpaca-instruction", type=Path, default=ALPACA_INSTRUCTION_PATH,
                    help="instruction written into every Alpaca row; must match the "
                         "one the host sends at inference")
    args = ap.parse_args()

    alpaca_instruction = args.alpaca_instruction.read_text(encoding="utf-8").strip()
    if args.format in ("alpaca", "both") and not alpaca_instruction:
        raise SystemExit(f"{args.alpaca_instruction} is empty")

    rng = random.Random(args.seed)
    out = Path(args.out)
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    editor_system = (
        SHORT_EDITOR_PROMPT if args.short_prompt
        else load_prompt(args.editor_prompt, SHORT_EDITOR_PROMPT)
    )
    verify_system = (
        SHORT_VERIFY_PROMPT if args.short_prompt
        else load_prompt(args.verify_prompt, SHORT_VERIFY_PROMPT)
    )

    if args.system_mode == "none":
        editor_system = ""
        verify_system = ""

    print("holdout:")
    held = load_suite_inputs()
    print(f"  {len(held)} eval-suite inputs will be excluded")

    disfl: list[tuple[str, str, str]] = []
    nyra: list[tuple[str, str]] = []
    dsp: list[tuple[str, str]] = []
    if not args.no_hub:
        print("hub sources:")
        try:
            disfl = fetch_disfl_qa(cache)
        except Exception as e:
            print(f"  ! disfl_qa unavailable ({e}); continuing without it")
        nyra = fetch_nyra(cache)
        dsp = fetch_disfluency_speech(cache)

    sentences = sentences_from_contexts(disfl) if disfl else []
    print(f"  {len(sentences)} clean sentences lifted from passages")

    # --- editor corpus ------------------------------------------------------
    # Proportions follow the measured failure modes rather than what is easiest
    # to generate: negatives are a quarter of the corpus because over-eager
    # cutting is the dominant error, and retractions dominate the positives
    # because that is the feature nothing else in the pipeline provides.
    pairs: list[tuple[str, str, str]] = []  # (raw, clean, tag)
    seen: set[str] = set()

    def add(raw: str, clean: str, tag: str) -> None:
        raw, clean = raw.strip(), clean.strip()
        # Judge the target before repairing it: tidying rewrites a trailing
        # comma into a full stop, which is the very thing that marks a segment
        # that was cut off mid-clause.
        if not raw or not clean or not target_is_usable(clean):
            return
        clean = tidy_target(clean)
        if not clean:
            return
        key = normalise(raw)
        if not key or key in seen or key in held:
            return
        seen.add(key)
        pairs.append((raw, clean, tag))

    # Declarative real speech first, so the cap below can be measured against
    # what it is meant to balance.
    for verb, intended in nyra:
        if isinstance(verb, str) and isinstance(intended, str):
            add(to_asr(verb, rng), tx.sentence_case(intended), "nyra")

    # Declarative counterweight to disfl_qa, which is entirely questions. A
    # fine-tune on questions alone learns to answer with one: LFM2.5-350M
    # trained that way turned "we we need to to check the logs" into "What is
    # the name of the log file?".
    # The `test` split is reserved so the eval set can be built from utterances
    # training never saw, rather than from a random slice of the same rows.
    dsp_train = dsp[dsp["split"] != "test"] if len(dsp) else dsp
    for verbatim, cleaned in zip(dsp_train["transcript_a"], dsp_train["transcript_c"]) \
            if len(dsp_train) else []:
        if not (isinstance(verbatim, str) and isinstance(cleaned, str)):
            continue
        raw, clean = tidy_spoken(verbatim), tidy_spoken(cleaned)
        # An utterance the annotators left untouched is a negative, and those
        # are wanted; one that lost most of its words is a segmentation
        # artefact rather than an edit.
        if not clean or len(clean) < 0.4 * len(raw):
            continue
        add(to_asr(raw, rng), clean, "disfluency_speech")

    # Real paragraphs, rebuilt from runs of consecutive utterances.
    for raw, clean in long_form_from_speech(dsp, rng, LONG_FORM_REAL_TARGET, {"train", "validation"}):
        add(to_asr(raw, rng), clean, "long_real")

    # disfl_qa is the largest and the best-annotated real source, and it is also
    # the only one made entirely of questions. Taken whole it was 60% of the
    # corpus, and the model that produced learned to answer everything with a
    # question. Capping it at the declarative real rows keeps its hard
    # annotations without letting its sentence shape become the house style.
    declarative = len(pairs)
    cap = declarative if args.disfl_qa_cap is None else args.disfl_qa_cap
    disfl_pool = list(disfl)
    rng.shuffle(disfl_pool)
    taken = 0
    for dis, flu, _ctx in disfl_pool:
        if taken >= cap:
            break
        before = len(pairs)
        add(to_asr(dis, rng), tx.sentence_case(flu), "disfl_qa")
        # A second copy in the other surface form, so the model sees this hard,
        # human-written data both cased and uncased.
        add(dis.lower().rstrip("?").strip() + "?", tx.sentence_case(flu), "disfl_qa_lower")
        taken += len(pairs) - before
    print(f"  disfl_qa capped at {cap} rows against {declarative} declarative ones")

    quota = max(0, args.n - len(pairs))
    plan = [
        (make_retraction, int(quota * 0.28), "syn_retraction"),
        (make_double_retraction, int(quota * 0.06), "syn_double"),
        # Long-range corrections. The app never chunks, so a paragraph arrives
        # in one request, and until now nothing in the corpus was longer than a
        # single utterance.
        (lambda r: make_long_retraction(r, sentences), int(quota * 0.12), "syn_long"),
        (lambda r: make_filler_only(r, sentences), int(quota * 0.15), "syn_filler"),
        (lambda r: make_negative(r, sentences), int(quota * 0.22), "syn_negative"),
        # The four patterns the trained editor actually got wrong.
        (lambda r: make_leading_adjunct(r, sentences), int(quota * 0.05), "syn_leading"),
        (lambda r: make_false_start(r, sentences), int(quota * 0.05), "syn_false_start"),
        (lambda r: make_initial_retraction(r, sentences), int(quota * 0.04), "syn_initial"),
        (lambda r: make_slot_confusion(r, sentences), int(quota * 0.03), "syn_slot"),
    ]
    for fn, count, tag in plan:
        made, attempts = 0, 0
        while made < count and attempts < count * 6:
            attempts += 1
            try:
                raw, clean = fn(rng)
            except Exception:
                continue
            before = len(pairs)
            add(raw, clean, tag)
            made += len(pairs) - before
        print(f"  {tag}: {made}")

    rng.shuffle(pairs)
    print(f"editor corpus: {len(pairs)} examples")

    # --- verifier corpus ----------------------------------------------------
    # Free labels: a known-good pair is SAME, and the same pair with a
    # deliberate injury is CHANGED. Balanced 50/50 so the model cannot win by
    # guessing, which is exactly how the prompted version failed.
    verifier: list[dict] = []
    for raw, clean, tag in ([] if args.editor_only else pairs):
        if tag.startswith("syn_negative"):
            continue
        if len(verifier) >= args.n:
            break
        verifier.append(
            chat(verify_system, f"Original: {raw}\nEdited: {clean}", "SAME")
        )
        bad = corrupt(clean, raw, rng)
        if bad:
            verifier.append(
                chat(verify_system, f"Original: {raw}\nEdited: {bad}", "CHANGED")
            )
    rng.shuffle(verifier)
    same = sum(1 for v in verifier if v["messages"][-1]["content"] == "SAME")
    print(f"verifier corpus: {len(verifier)} examples ({same} SAME, {len(verifier)-same} CHANGED)")

    # --- eval corpus --------------------------------------------------------
    # Held out by *source*, not by slicing the finished corpus. A random slice
    # would still share the Wikipedia sentences and the template instances that
    # generated it, so it would measure memorisation. Here the real utterances
    # come from a split training never read, and the synthetic half runs on a
    # different seed with every training input excluded by hand.
    print("eval corpus:")
    eval_rng = random.Random(args.seed + 10_000)
    eval_pairs: list[tuple[str, str, str]] = []
    eval_seen: set[str] = set()

    def add_eval(raw: str, clean: str, tag: str) -> None:
        raw, clean = raw.strip(), clean.strip()
        if not raw or not clean or not target_is_usable(clean):
            return
        clean = tidy_target(clean)
        key = normalise(raw)
        # `seen` is every input the model will be trained on. Nothing that
        # appears there may appear here, or the score is inflated by recall.
        # `held` is the 68-case bench suite, excluded so the two evaluation
        # instruments stay independent and can be quoted side by side.
        if not key or not clean or key in eval_seen or key in seen or key in held:
            return
        eval_seen.add(key)
        eval_pairs.append((raw, clean, tag))

    dsp_eval = dsp[dsp["split"] == "test"] if len(dsp) else dsp
    if len(dsp_eval):
        for verbatim, cleaned in zip(dsp_eval["transcript_a"], dsp_eval["transcript_c"]):
            if not (isinstance(verbatim, str) and isinstance(cleaned, str)):
                continue
            raw, clean = tidy_spoken(verbatim), tidy_spoken(cleaned)
            if not clean or len(clean) < 0.4 * len(raw):
                continue
            add_eval(to_asr(raw, eval_rng), clean, "eval_real")
    for raw, clean in long_form_from_speech(dsp, eval_rng, 150, {"test"}):
        add_eval(to_asr(raw, eval_rng), clean, "eval_long_real")

    eval_plan = [
        (make_retraction, 500, "eval_retraction"),
        (lambda r: make_long_retraction(r, sentences), 300, "eval_long"),
        (make_double_retraction, 100, "eval_double"),
        (lambda r: make_filler_only(r, sentences), 200, "eval_filler"),
        (lambda r: make_negative(r, sentences), 400, "eval_negative"),
        # The eval has to cover the targeted patterns too, or the next run
        # cannot tell whether adding them worked.
        (lambda r: make_leading_adjunct(r, sentences), 150, "eval_leading"),
        (lambda r: make_false_start(r, sentences), 150, "eval_false_start"),
        (lambda r: make_initial_retraction(r, sentences), 150, "eval_initial"),
        (lambda r: make_slot_confusion(r, sentences), 150, "eval_slot"),
    ]
    for fn, count, tag in eval_plan:
        made, attempts = 0, 0
        while made < count and attempts < count * 20:
            attempts += 1
            try:
                raw, clean = fn(eval_rng)
            except Exception:
                continue
            before = len(eval_pairs)
            add_eval(raw, clean, tag)
            made += len(eval_pairs) - before
        print(f"  {tag}: {made}")

    eval_records = [chat(editor_system, raw, clean) for raw, clean, _ in eval_pairs]
    if not args.editor_only:
        for raw, clean, tag in eval_pairs:
            if tag.endswith("negative"):
                continue
            eval_records.append(
                chat(verify_system, f"Original: {raw}\nEdited: {clean}", "SAME")
            )
            bad = corrupt(clean, raw, eval_rng)
            if bad:
                eval_records.append(
                    chat(verify_system, f"Original: {raw}\nEdited: {bad}", "CHANGED")
                )
    eval_rng.shuffle(eval_records)

    # --- write --------------------------------------------------------------
    want_chat = args.format in ("chatml", "both")
    want_alpaca = args.format in ("alpaca", "both")

    def dump(path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        size = path.stat().st_size / 1e6
        print(f"  wrote {path}  ({len(records)} rows, {size:.1f} MB)")

    def emit(stem: str, records: list[dict]) -> None:
        """Write one logical file in every requested format.

        The Alpaca copy is derived from the same list, so the two formats cannot
        disagree about which examples the corpus contains.
        """
        if want_chat:
            dump(out / f"{stem}.jsonl", records)
        if want_alpaca:
            dump(
                out / f"{stem}_alpaca.jsonl",
                [to_alpaca(r, alpaca_instruction_for(r, alpaca_instruction))
                 for r in records],
            )

    def write(name: str, records: list[dict]) -> None:
        cut = int(len(records) * (1 - args.val_fraction))
        for split, chunk in (("train", records[:cut]), ("val", records[cut:])):
            emit(f"{name}_{split}", chunk)

    editor_records = [chat(editor_system, raw, clean) for raw, clean, _ in pairs]
    print("output:")
    write("editor", editor_records)
    if not args.editor_only:
        write("verifier", verifier)

    # A single shuffled file covering both tasks, for a model that has to edit
    # *and* judge. Only written when there are two tasks to combine -- under
    # --editor-only it would duplicate handy_editor.jsonl byte for byte.
    #
    # Also the shape Unsloth Studio wants: one local file it slices into train
    # and eval itself, rather than two uploads it has no way to relate.
    combined = editor_records + verifier
    rng.shuffle(combined)
    if not args.editor_only:
        emit("handy_combined", combined)

    # The editor task on its own, in the same single-file shape, for a run that
    # only wants to move the number the 68-case suite measures.
    editor_only = list(editor_records)
    rng.shuffle(editor_only)
    emit("handy_editor", editor_only)
    emit("handy_eval", eval_records)

    # Cheap insurance against a refactor quietly reconnecting the two sets.
    train_inputs = {normalise(r["messages"][-2]["content"]) for r in combined}
    overlap = train_inputs & {normalise(r["messages"][-2]["content"]) for r in eval_records}
    if overlap:
        raise SystemExit(f"eval leaks into train: {len(overlap)} shared inputs")
    print(f"  eval/train overlap: 0 of {len(eval_records)} rows")

    manifest = {
        "seed": args.seed,
        "editor_examples": len(editor_records),
        "verifier_examples": len(verifier),
        "held_out_suite_inputs": len(held),
        "short_prompt": args.short_prompt,
        "system_mode": args.system_mode,
        "format": args.format,
        "alpaca_instruction": alpaca_instruction if want_alpaca else None,
        "editor_only": args.editor_only,
        "editor_system_sha256": hashlib.sha256(editor_system.encode()).hexdigest()[:16],
        "verify_system_sha256": hashlib.sha256(verify_system.encode()).hexdigest()[:16],
        "composition": {tag: sum(1 for _, _, t in pairs if t == tag)
                        for tag in sorted({t for _, _, t in pairs})},
        "sources": {
            "disfl_qa": "google-research-datasets/disfl_qa (CC-BY-4.0)",
            "nyralabs": "nyralabs/disfluency_speech_english (Apache-2.0)",
            "disfluency_speech": "amaai-lab/DisfluencySpeech (Apache-2.0)",
            "synthetic": "generated by scripts/enhance-train/build_dataset.py",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {out / 'manifest.json'}")
    print("\nComposition:")
    for tag, n in sorted(manifest["composition"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:7d}  {tag}")


if __name__ == "__main__":
    main()

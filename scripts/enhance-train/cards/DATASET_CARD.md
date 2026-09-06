---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-generation
tags:
  - dictation
  - speech-disfluency
  - disfluency-removal
  - text-editing
  - alpaca
size_categories:
  - 10K<n<100K
configs:
  - config_name: alpaca
    default: true
    data_files:
      - split: train
        path: handy_editor_alpaca.jsonl
      - split: test
        path: handy_eval_alpaca.jsonl
  - config_name: chatml
    data_files:
      - split: train
        path: handy_editor.jsonl
      - split: test
        path: handy_eval.jsonl
---

# Handy dictation-editing corpus

Turns a raw dictated transcript into the text the speaker meant to write.

```
in : um so the meeting is uh moved to friday no wait thursday at three
out: The meeting is Thursday at three.
```

Three jobs at once, because they are not separable in speech: drop filler words,
repair punctuation and capitalisation, and — the hard one — when the speaker
changes their mind mid-sentence, delete the wording they abandoned and keep only
what they settled on.

Built for the on-device enhancement layer in
[Handier](https://github.com/MagicNoThief/Handier-desktop), a fork of
[Handy](https://github.com/cjpais/Handy), where a sub-500M model has to do this
in under a second on a laptop with no GPU. It is not specific to that app: it is
a plain instruction-tuning corpus.

## Format

Two views of the **same rows**, generated from one list so they cannot disagree.

**`*_alpaca.jsonl`** — for fine-tuning a **base** model, which has no chat
template of its own:

```json
{
  "instruction": "Rewrite the dictated transcript as the speaker meant to write it. Remove filler words and hesitations, repair punctuation and capitalisation, and when the speaker corrects themselves keep only the wording they settled on. Reply with the edited text and nothing else.",
  "input": "book it for four people make that six people",
  "output": "Book it for six people."
}
```

**`*.jsonl`** — the same examples as an OpenAI `messages` array, for an
**instruct** model that already has a chat template:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "book it for four people make that six people"
    },
    { "role": "assistant", "content": "Book it for six people." }
  ]
}
```

The chat view carries **no system turn**. That is deliberate: the behaviour is
meant to live in the weights, and prepending a ~450-token instruction to a
~30-token example is most of the sequence for no gain. Measured on an earlier
run, giving the fine-tune the instruction at inference was actively _worse_ —
54/68 against 57/68 — because it started copying the prompt's own rules into its
output.

### The instruction is constant

Every Alpaca row carries the same `instruction`. This is a property of the
intended use, not an oversight: the host that serves the model sends exactly this
string, and a fine-tune given an instruction a few words off the one it learned
degrades quietly. If you want instruction diversity, paraphrase it yourself —
but then serve a matching distribution.

### Serve it as one turn

Whatever you train, prompt it with the **whole Alpaca string in a single turn**.
Splitting it — instruction into a chat "system" slot, input into a "user" slot —
looks equivalent and is not. Measured on a 350M fine-tune of this corpus:

| Prompt shape                         | Result                                                                 |
| ------------------------------------ | ---------------------------------------------------------------------- |
| Instruction in system, input in user | Repeats itself until the token budget runs out; no stop token. ~620 ms |
| Whole Alpaca prompt as one turn      | Clean, terminated edits. **~100 ms**                                   |

This bites specifically because a converted base-model GGUF often still carries a
chat template, which then wraps the split prompt into a shape the model never saw
in training. One assembled turn is safe either way.

## Files

| File                            |         Rows |    Size | Use                                   |
| ------------------------------- | -----------: | ------: | ------------------------------------- |
| `handy_editor_alpaca.jsonl`     |       89,996 | 45.5 MB | **Train on this** (base model)        |
| `handy_editor.jsonl`            |       89,996 | 25.0 MB | Train on this (instruct model)        |
| `handy_eval_alpaca.jsonl`       |        2,152 |  1.1 MB | Held-out eval. **Never train on it.** |
| `handy_eval.jsonl`              |        2,152 |  0.6 MB | The same eval, chat view              |
| `editor_train/val_alpaca.jsonl` | 88.2K / 1.8K |       — | Pre-split, if you want it             |

## Context length

Measured by rendering every row as the full Alpaca sequence — preamble,
instruction, input and the response being learned — and tokenising with LFM2's
tokenizer:

|                             | median | p90 | p99 |     max |
| --------------------------- | -----: | --: | --: | ------: |
| `handy_editor_alpaca.jsonl` |    128 | 167 | 277 | **510** |
| `handy_eval_alpaca.jsonl`   |    126 | 171 | 256 |     329 |

**512 truncates nothing, by two tokens.** That is not much margin, and
truncation cuts from the right — which is where the target lives, so a clipped
row teaches the model to stop mid-sentence. 384 clips 146 rows. If your trainer
pads dynamically, 640 costs almost nothing and removes the question; use 512 only
if you are sure your renderer matches the one above.

(The chat view is shorter — max 426 — because it has no preamble.)

## Composition

89,996 training rows. **15.6% come from recordings of real speech**; the rest are
generated from templates targeting specific failure modes.

| Source              |   Rows | What it contributes                               |
| ------------------- | -----: | ------------------------------------------------- |
| `syn_retraction`    | 21,271 | Mid-sentence self-correction, the core case       |
| `syn_negative`      | 16,713 | Sentences needing **no** edit — the counterweight |
| `syn_filler`        | 11,395 | Filler words and hesitations only                 |
| `syn_long`          |  9,116 | Multi-sentence utterances                         |
| `disfl_qa`          |  6,980 | Real human-written disfluent questions            |
| `nyra`              |  4,921 | Real disfluent speech transcripts                 |
| `syn_double`        |  4,558 | Two corrections in one utterance                  |
| `syn_false_start`   |  3,798 | Abandoned pronoun left stranded                   |
| `syn_leading`       |  3,798 | Sentence-initial phrase wrongly dropped           |
| `syn_initial`       |  3,038 | Retraction that opens the utterance               |
| `syn_slot`          |  2,279 | Replacement landing on the wrong slot             |
| `long_real`         |  1,109 | Long-form real speech                             |
| `disfluency_speech` |    985 | Real read-aloud disfluent speech                  |
| `disfl_qa_lower`    |     35 | Case-normalised variants                          |

`syn_negative` at 19% is load-bearing. A corpus of nothing but corrections
teaches a model that _something_ must always be deleted, and it starts eating
sentences that were fine. The five smallest generators exist because a previous
fine-tune got exactly those patterns wrong.

## The held-out split is held out by source

`handy_eval*.jsonl` is not a random slice of the training file. A slice would
still share the source sentences and template instances that produced it, and
would measure memorisation. Instead:

- real utterances come from DisfluencySpeech's `test` split, which the training
  build never reads;
- synthetic rows use a different RNG seed, with every training input excluded by
  hand;
- the builder asserts zero input overlap and fails the build if that ever breaks.

## How it was built

`scripts/enhance-train/build_dataset.py` in
[Handier](https://github.com/MagicNoThief/Handier-desktop):

```bash
python build_dataset.py --out ./corpus --n 90000 \
    --system-mode none --editor-only --format both
python validate_dataset.py ./corpus     # gate before spending GPU time
```

`validate_dataset.py` checks JSON validity, role patterns, that no target still
contains a filler or a stutter, that targets are not malformed, and that no
eval-suite input has leaked in.

## Limitations

- **English only.**
- **Punctuation style is British-ish and opinionated** — sentence case, serial
  commas absent, full stops added. A model trained on this will impose it.
- **Short utterances dominate.** The median row is a single sentence; long-form
  dictation is ~10% of the corpus.
- **The synthetic majority is template-shaped.** It covers the failure modes it
  was aimed at, and generalises less well outside them.
- **Real-speech transcripts carry their annotators' choices**, including some
  targets that are arguably worse than what a good model would write.
- **No verification rows.** An earlier version of this corpus trained a second
  "did the meaning change?" task. Measured end to end on 400 live edits it caught
  0 of 10 bad edits and rejected 1 good one, so the whole budget went to editing.

## Sources and licensing

| Source                                                                                                     | Licence                         |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------- |
| [`google-research-datasets/disfl_qa`](https://huggingface.co/datasets/google-research-datasets/disfl_qa)   | CC-BY-4.0                       |
| [`nyralabs/disfluency_speech_english`](https://huggingface.co/datasets/nyralabs/disfluency_speech_english) | Apache-2.0                      |
| [`amaai-lab/DisfluencySpeech`](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech)                 | Apache-2.0                      |
| Synthetic rows                                                                                             | Generated by `build_dataset.py` |

Released under **CC-BY-4.0**, the most restrictive of the inbound licences.
Attribution to the three datasets above is required.

> **Check before you publish:** `disfl_qa` is derived from SQuAD, which is
> CC-BY-SA-4.0. Its own Hugging Face card declares CC-BY-4.0, and this corpus
> follows that declaration — but if you need certainty about whether a
> share-alike obligation reaches this derivative, confirm it rather than relying
> on this note.

## Citation

```bibtex
@misc{handy_dictation_editing,
  title  = {Handy dictation-editing corpus},
  author = {Handier contributors},
  year   = {2026},
  note   = {Built with scripts/enhance-train/build_dataset.py},
  url    = {https://github.com/MagicNoThief/Handier-desktop}
}
```

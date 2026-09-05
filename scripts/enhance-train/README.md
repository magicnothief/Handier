# Enhancement training corpus

Builds the fine-tuning data for the enhancement layer.

```
editor    raw transcript          -> cleaned transcript
verifier  (original, edited)      -> SAME / CHANGED   (no longer recommended)
```

The verify pass measured **net-negative** against a trained editor — it caught 0
of 10 real mistakes and destroyed a correct edit — so the default build is now
editor-only and the freed budget goes to editing. Details in
`../enhance-eval/README.md`.

## Running it

The Hub fetchers need `pandas`, `pyarrow` and `fsspec`. They are not in the app's
toolchain, so keep them in their own environment — on this machine that lives off
the system drive:

```bash
python -m venv D:/dev/handy-train-venv
D:/dev/handy-train-venv/Scripts/python.exe -m pip install pandas pyarrow fsspec aiohttp
```

Then build, and validate before spending GPU time:

```bash
D:/dev/handy-train-venv/Scripts/python.exe build_dataset.py \
    --out D:/dev/handy-train-editor --n 90000 --system-mode none --editor-only
D:/dev/handy-train-venv/Scripts/python.exe validate_dataset.py D:/dev/handy-train-editor
```

Drop `--editor-only` to also build the verifier corpus.
`--no-hub` builds from the synthetic generators alone. `--disfl-qa-cap N` overrides
the automatic cap described below. `--system-mode none` omits the system turn, for
a fine-tune meant to be run without a prompt.

### Why `--n 90000`

The Hub sources contribute a fixed ~12K rows; everything above that is synthetic
template output, so raising `--n` buys template repetition rather than coverage.
The combined build used 45K to keep the real share at 27% and both halves under
the upload size. Editor-only spends the whole budget on one task, so 90K gives the
same number of editing rows the 45K combined build gave _plus_ the verifier's
share, in a 25 MB file. Real data is 13% of it; the added rows are targeted
generators, not more of the same.

## Training in Unsloth Studio

Studio reads `CSV`, `JSONL`, `JSON` and `Parquet` from the Hub, a local file, or
S3. Its `chatml` format is the OpenAI-style `messages` array and its `alpaca`
format is `{instruction, input, output}` — this builder emits both, so the files
load as-is with format on `auto`.

Upload **one** file. Studio splits train/eval itself from a single dataset rather
than relating two uploads, which is why the builder also writes these.

**Editor-only build** (`D:/dev/handy-train-alpaca` with `--format both`):

| File                            |         Rows |        Size | Use                                      |
| ------------------------------- | -----------: | ----------: | ---------------------------------------- |
| **`handy_editor_alpaca.jsonl`** |    **90.0K** | **45.5 MB** | **Train a base model on this.**          |
| **`handy_editor.jsonl`**        |    **90.0K** | **25.0 MB** | **Train an instruct model on this.**     |
| `handy_eval_alpaca.jsonl`       |         2.2K |      1.1 MB | Held-out eval. **Never train on this.**  |
| `handy_eval.jsonl`              |         2.2K |      0.6 MB | The same eval, chat view.                |
| `editor_train/val[_alpaca]`     | 88.2K / 1.8K |           — | Pre-split, for the Python/notebook path. |

Match the file to the checkpoint: a **base** model has no chat template, so it
needs the Alpaca file; an **instruct** model has one, so it needs the chat file.
Crossing them trains the model on a format nothing will ever send it.

Built with:

```bash
build_dataset.py --out D:/dev/handy-train-alpaca --n 90000     --system-mode none --editor-only --format both
```

The verifier half is dropped because it measured net-negative in the app — see
`../enhance-eval/README.md`. Its budget went to editing, so this corpus has twice
the editor rows of the combined build, plus four generators aimed at the patterns
the trained editor actually got wrong:

| Generator         | Rows | Fixes                                                                          |
| ----------------- | ---: | ------------------------------------------------------------------------------ |
| `syn_leading`     | 3.8K | Dropped a sentence-initial phrase (`Between Bingen and Bonn,`)                 |
| `syn_false_start` | 3.8K | Kept an abandoned pronoun (`We you'd have to just sit and wait`)               |
| `syn_initial`     | 3.0K | Retraction that opens the utterance (`no wait not friday…`)                    |
| `syn_slot`        | 2.3K | Replacement landing on the wrong slot (`the audit report to the finance team`) |

The eval carries 150 of each, so the next run can tell whether they worked.

**Combined build** (`D:/dev/handy-train-v2`) is still there if you want one model
doing both jobs: `handy_combined.jsonl` (90.0K, 27.9 MB).

### Context length

Measured by rendering every row through LFM2's own chat template and tokenising
with LFM2's tokenizer, assistant turn included:

|                                    | median | p90 | p99 |     max |
| ---------------------------------- | -----: | --: | --: | ------: |
| `handy_editor.jsonl` (editor-only) |     44 |  83 | 193 | **426** |
| `handy_eval.jsonl` (editor-only)   |     42 |  87 | 172 |     245 |
| `handy_combined.jsonl`             |     52 |  93 | 236 |     428 |

**Set it to 512** for the chat format. That is the smallest value that truncates
nothing; on the editor-only build 384 clips 13 rows and 256 clips 299. For Alpaca
see the measurement further down -- the fixed preamble pushes the maximum to 510,
so 512 still fits but with no margin. Truncation cuts from the right, which is where the target
lives — on a verifier row that is the entire `SAME`/`CHANGED` label, three tokens
that are the whole example.

Going higher is close to free with dynamic padding, since batches pad to the
longest sequence in the batch rather than to the ceiling. The risk is one-sided,
so err upward.

(Before long-form was added the max was 162 and 192 sufficed. The long examples
are what moved it.)

### Held-out eval

`handy_eval.jsonl` is held out **by source**, not by slicing the finished corpus —
a random slice would still share the Wikipedia sentences and template instances
that produced it, and would measure memorisation. Instead:

- real utterances come from DisfluencySpeech's `test` split, which the training
  build never reads;
- synthetic rows use a different RNG seed, with every training input excluded by
  hand;
- the 68 bench-suite inputs are excluded too, so `bench.py` and this file measure
  independent things and can be quoted side by side.

The builder asserts zero overlap and exits non-zero if that ever breaks. The
editor-only eval is 2,152 edit rows including long-form and 150 of each targeted
pattern; the combined build's eval additionally carries ~2.1K verify rows.

If you do build the combined corpus, note that the sidecar holds one model and
asks it both to edit and to judge, so the two tasks have to share weights. They
stay distinguishable from the user turn alone — a verify request always opens
`Original:` — and `validate_dataset.py` asserts that separation holds.

### Formats: chatml and Alpaca

`--format` picks what the builder writes. Both views are generated from the same
list of examples, so they cannot disagree about the corpus contents.

| `--format`         | Files                       | For                                              |
| ------------------ | --------------------------- | ------------------------------------------------ |
| `chatml` (default) | `handy_editor.jsonl`        | An **instruct** model, which has a chat template |
| `alpaca`           | `handy_editor_alpaca.jsonl` | A **base** model, which does not                 |
| `both`             | both                        | Publishing, or deciding later                    |

Alpaca rows are `{instruction, input, output}` with a **constant** instruction,
read from `alpaca_instruction.txt`. That file is the single source of truth: the
Rust side `include_str!`s the same path, so the string the model is trained on
and the string the host sends at inference cannot drift apart. Change the file
and both follow; change one in isolation and you get a fine-tune served a
paraphrase of its own instruction, which degrades quietly rather than failing.

```bash
build_dataset.py --out D:/dev/handy-train-alpaca --n 90000 \
    --system-mode none --editor-only --format both
```

### The corpus is promptless, and so is the app

Every chat row is `[user, assistant]` with no system turn, because that is how
the fine-tune is run. The app now matches: `PromptStyle::Tuned` on a catalog
entry — or **How to prompt this model** on a local GGUF — makes the host send
`system: ""` instead of the shipped editor prompt, and `local:<path>` model ids
let an unpublished GGUF be selected without going through Hugging Face.

**An empty system turn is not the same as no system turn.** An earlier version of
this file claimed LFM2 renders both identically. That is false as llama.cpp
applies the template, and it matters: dropping the message instead of emptying it
measured **66/68 -> 59/68**, with the model answering `SAME` and `CHANGED` to
editing requests because it no longer recognised the shape of its own input. The
sidecar passes the empty string straight through; do not "tidy" it away.

### Serving an Alpaca fine-tune: one turn, not two

`PromptStyle::Alpaca` / **Fine-tuned from a base model** assembles the _whole_
Alpaca prompt — preamble, `### Instruction:`, `### Input:`, `### Response:` — and
sends it as a single user turn, with an empty system turn.

That is not a stylistic choice, it is the difference between working and not.
The obvious alternative is to put the instruction in the system slot and the
transcript in the user slot, which is what this originally did. Measured on
`checkpoint-5000`:

| Prompt shape                              | Result                                                                            |
| ----------------------------------------- | --------------------------------------------------------------------------------- |
| Instruction in system, transcript in user | Repeats itself until the token budget runs out; never emits a stop token. ~620 ms |
| Whole Alpaca prompt as one user turn      | Clean, terminated edits. **~100 ms**                                              |

The reason is that a converted base-model GGUF often still carries a chat
template. If it does, the template wraps whatever it is handed — so a split
prompt becomes `<|im_start|>system\n{instruction}<|im_end|>…`, a shape the
fine-tune has never seen. Sending one assembled turn works either way: with a
template it is wrapped around text the model keys on anyway, and without one the
sidecar passes an empty-instruction request straight through untouched.

**Check before you trust a GGUF's provenance.** `tokenizer.chat_template` being
present does not mean the model was trained on chat turns — both Alpaca
checkpoints here carry one. Probe it: run a handful of cases under each shape and
look at the output. Runaway repetition is what a format mismatch looks like.

`bench.py --alpaca` and `eval_heldout.py --alpaca` use the same assembly, reading
`alpaca_instruction.txt` and `alpaca_preamble.txt`, so the harness cannot measure
a prompt different from the one the app will send.

**Unsloth Studio's `auto` format detection handles the Alpaca file correctly** —
verified, not assumed. A fine-tune trained through Studio on
`handy_editor_alpaca.jsonl` with the format left on `auto` answers the Alpaca
shape and collapses on the chat shapes:

| Prompt shape at inference     | Suite |
| ----------------------------- | ----: |
| Alpaca, one turn (`--alpaca`) | 61/68 |
| plain chatml, empty system    | 29/68 |
| plain chatml, no system       |  0/68 |

If detection had gone the other way the ordering would be reversed, so this is a
clean read. Worth re-running on any new checkpoint before blaming the data
pipeline for a weak model — a format fault and an undertrained model look similar
from the score alone, and they are fixed in completely different places.

Build with `--system-mode prompt` instead if you would rather train against the
shipped instruction prompt; the corpus then carries it on every row.

### Context length

Measured through LFM2's tokenizer on the rendered training sequence, response
included:

|                             | median | p90 | p99 |     max |
| --------------------------- | -----: | --: | --: | ------: |
| `handy_editor.jsonl` (chat) |     44 |  83 | 193 |     426 |
| `handy_editor_alpaca.jsonl` |    128 | 167 | 277 | **510** |

Alpaca adds a fixed ~90-token preamble and instruction to every row. **512 still
truncates nothing — by two tokens.** 384 clips 146 Alpaca rows. With dynamic
padding 640 costs almost nothing and removes the question, and the risk is
one-sided: truncation cuts from the right, which is where the target lives.

### Testing the result

Export to GGUF, then score it against the suite the app is measured on:

```bash
cd ../enhance-eval
python bench.py --model D:/dev/handy-models/<export>.gguf --no-switch --label mine
```

`--no-switch` is required for a fine-tune: without it the harness appends
`/no_think`, a token the model has never seen, into the transcript it is meant to
edit. Omit `--prompt` for a promptless model. See that directory's README for what
the current numbers are.

**Check the composition it prints.** If a Hub source reports `0 pairs`, the
fetcher failed and was swallowed — the run will still succeed and quietly produce
a corpus missing its real-speech half. That has happened: an earlier corpus was
100% synthetic templates because `pandas` was not installed.

## Sources

| Source                               | Licence    | Rows    | Shape                                                                                                                                |
| ------------------------------------ | ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `google-research-datasets/disfl_qa`  | CC-BY-4.0  | 11.8K   | Questions. Annotator-written retractions whose distractor comes from the source passage, so they are much harder than a random swap. |
| `nyralabs/disfluency_speech_english` | Apache-2.0 | 5.0K    | Declarative. Verbatim/intended pairs from real speech: fillers, repetitions, cutoffs.                                                |
| `amaai-lab/DisfluencySpeech`         | Apache-2.0 | 5.0K    | Declarative. Switchboard utterances published at three cleanup levels; `transcript_a` → `transcript_c` is exactly this task.         |
| ↳ rebuilt into passages              | Apache-2.0 | 1.1K    | **Real long-form**, see below.                                                                                                       |
| synthetic                            | —          | balance | Retractions, doubles, long-range retractions, filler injection, negatives.                                                           |

### Real long-form, not synthesised

DisfluencySpeech publishes one utterance per row, but the audio filename is a
running index over the source corpus — 94% of neighbours are consecutive — so
joining a run of adjacent rows recovers a genuine paragraph of conversational
dictation. Real hesitations, real repairs, real cleanup, rather than an
approximation of them:

```
IN : ...in the first three minutes you're talking about it, and the re-, the
     other fifty-seven minutes... innovative in dealing with, uh, the various
     students... it's great to be well rounded and hav-, be exposed to all this
     stuff... they're very, tri-, almost trivial with it.
OUT: ...and the other fifty-seven minutes... in dealing with, the various
     students... great to be well rounded and be exposed to all this stuff...
     they're almost trivial with it.
```

Runs are cut wherever the index is not contiguous. The corpus does not mark call
boundaries, so a window can change subject partway. That is harmless here and
arguably useful: the target keeps every sentence, so those become long examples
whose correct answer is to delete nothing.

No public dataset was found with long-form _disfluent/clean pairs_ directly.
Earnings-call corpora (`distil-whisper/earnings22`, `florencejiang/earnings25`)
are long-form but ship a single transcript with no cleaned counterpart, so there
is no target to train against.

`syn_long` covers what real data cannot: a retraction that reaches back several
sentences, where the abandoned wording cannot be found by looking at the
neighbouring clause.

```
IN : one more thing we ship in october the deadline is in october i want to
     publish the release hmm no we ship in may
OUT: One more thing, we ship in May. The deadline is in October. I want to
     publish the release.
```

Both dates say October; only the one the speaker restated may change.

### Why disfl_qa is capped

disfl_qa is the best-annotated source and the only one made entirely of
questions. Taken whole it was **60% of the corpus**, and a model trained that way
learns that the answer is always a question. Measured on a full fine-tune of
LFM2.5-350M trained on disfl_qa alone:

```
in : we we need to to check the logs
out: What is the name of the log file?

in : let's meet tuesday actually no let's meet thursday
out: What is the name of the program that uses the Air Force for its operations?
```

That second answer is SQuAD content the model never saw in the input. Inventing
text is the worst failure this layer can have — worse than doing nothing.

The builder now adds the declarative sources first and caps disfl_qa at their
combined size. Question-shaped targets went from 60.0% to 21.4% of the corpus.

### Why synthesise at all

No public dataset contains the negative half. Every disfluency corpus is a corpus
of things to remove, so a model trained only on them learns that editing is always
correct. Over-eager cutting was the measured dominant failure, so roughly a fifth
of the corpus is text whose correct output is itself.

Nothing public covers explicit entity-replacing self-correction at scale either
("send it to John — no wait, Jane"), which is the feature the rest of the pipeline
cannot provide. That is what `syn_retraction` and `syn_double` are for.

## Rejected sources

- **`stillerman/fdt-disfluency-synthetic`** — DailyDialog + Persona-Chat with
  word-level KEEP/DELETE tags, and a good shape for this task. Licensed
  **CC-BY-NC-SA-4.0**: the non-commercial and share-alike terms would follow the
  trained weights into anything shipped, so it is not used. Fine for evaluation.
- **`hhoangphuoc/ami-disfluency`** — gated, and holds only disfluency and
  laughter segments rather than clean/verbatim pairs.
- **Switchboard / Fisher originals** — LDC licensed. `amaai-lab/DisfluencySpeech`
  is a re-recording of Switchboard material released under Apache-2.0 and carries
  the annotations this needs.

## Corpus checks worth re-running

The builder does not enforce these; they caught real bugs.

```python
# Targets must not be question-shaped as a matter of course.
q = sum(1 for r in rows if r["messages"][2]["content"].rstrip().endswith("?"))

# A target containing a doubled word teaches the model to keep a stutter.
re.compile(r"\b(\w+)\s+\1\b", re.I)

# Fillers surviving into a target teach it to keep them.
```

The doubled-word check found templates that supplied an article to a slot bank
whose entries already carried one — `"forward it to the {slot}"` against
`"the board"` produced `"forward it to the the board"`. It is down to 0.19%, and
what remains is genuine repetition in the real-speech sources ("that that").

## Holdout

`scripts/enhance-eval/suite.py`'s 68 inputs are excluded from training by exact
match. That is not enough to call the suite held out any more: the synthetic
generators draw on the same signal vocabulary, so the suite is in-distribution
even when its exact strings are absent. **A fresh, unseen suite is needed before
any score from a fine-tuned model can be compared with the base-model numbers in
`../enhance-eval/README.md`.**
